#include "ctranslate2/layers/drama_projection.h"

#include "ctranslate2/ops/dequantize_fp8.h"

namespace ctranslate2 {
  namespace layers {

    DramaProjection::DramaProjection(const models::Model& model, const std::string& scope,
                                     const DramaExecutionOptions& options)
      : _weight(model.get_variable(scope + "/weight"))
      , _bias(model.get_variable_if_exists(scope + "/bias"))
      , _qscale(model.get_variable_if_exists(scope + "/weight_scale"))
      , _fp8_scale_inv(model.get_variable_if_exists(scope + "/weight_scale_inv"))
      , _gpu_device_index(options.device == Device::CUDA ? options.device_index : -1)
      , _gpu_compute_type(options.compute_type)
    {
      if (_weight.device() != Device::CPU || model.tensor_parallel()
          || model.get_variable_if_exists(scope + "/weight_packed")
          || model.get_variable_if_exists(scope + "/weight_zero")
          || get_default_float_type(model.effective_compute_type()) != DataType::FLOAT32)
        throw std::invalid_argument("Drama3 projections require unpacked CPU weights and float32 graph state");
      if (_fp8_scale_inv) {
        if (model.config.value("fp8_format", std::string()) != "e4m3fn_block128"
            || _weight.dtype() != DataType::INT8 || _qscale)
          throw std::invalid_argument("Block FP8 projections require explicit E4M3FN metadata");
      } else if (_gpu_device_index < 0) {
        _cpu = std::make_unique<Dense>(model, scope);
      }
    }

    void DramaProjection::prepare_gpu_weight(const StorageView& weight, const StorageView* scale) const {
      const auto types = compute_type_to_data_type(_gpu_compute_type);
      const auto weight_type = types.first, float_type = types.second;
      if (_gpu_weight.empty()) {
        StorageView uploaded;
        if (!_fp8_scale_inv && !is_float_type(weight.dtype())) {
          StorageView decoded;
          ops::Dequantize()(weight, *scale, decoded);
          uploaded = decoded.to(Device::CUDA);
        } else {
          uploaded = weight.to(Device::CUDA);
        }
        if (_fp8_scale_inv) {
          _gpu_scale_inv = _fp8_scale_inv->to(Device::CUDA);
          // Prepare half-precision experts once, just like INT8 experts, instead
          // of decoding and synchronizing their FP8 weights on every projection.
          if (weight_type == DataType::INT8 || weight_type == DataType::FLOAT16
              || weight_type == DataType::BFLOAT16) {
            StorageView decoded(DataType::FLOAT32, Device::CUDA);
            ops::DequantizeFP8()(uploaded, _gpu_scale_inv, decoded);
            uploaded = std::move(decoded);
            _gpu_scale_inv.release();
          }
        }
        if (weight_type == DataType::INT8) {
          StorageView quantized(DataType::INT8, Device::CUDA),
              scales(DataType::FLOAT32, Device::CUDA);
          ops::Quantize(ops::Quantize::ScaleType::PER_LAYER, false)(uploaded, quantized, scales);
          _gpu_scale = std::move(scales);
          uploaded = std::move(quantized);
        } else if (_gpu_scale_inv.empty() && uploaded.dtype() != float_type) {
          uploaded = uploaded.to(float_type);
        }
        _gpu_weight = std::move(uploaded);
      }
    }

    void DramaProjection::forward_gpu(const StorageView& input, const StorageView& weight,
                            const StorageView* bias, const StorageView* scale,
                            const StorageView* residual, StorageView& output) const {
      const ScopedDeviceSetter device_setter(Device::CUDA, _gpu_device_index);
      if (input.dtype() != DataType::FLOAT32 || output.dtype() != DataType::FLOAT32)
        throw std::invalid_argument("Drama3 GPU projections require float32 input/output");
      if ((input.device() == Device::CUDA && input.device_index() != _gpu_device_index)
          || (output.device() == Device::CUDA && output.device_index() != _gpu_device_index))
        throw std::invalid_argument("Drama3 GPU projection tensors must use the configured device");
      const auto types = compute_type_to_data_type(_gpu_compute_type);
      const auto weight_type = types.first, float_type = types.second;
      prepare_gpu_weight(weight, scale);
      StorageView converted_input;
      const StorageView* gpu_input = &input;
      if (input.device() != Device::CUDA) {
        converted_input = input.to(Device::CUDA);
        gpu_input = &converted_input;
      }
      if (gpu_input->dtype() != float_type) {
        converted_input = gpu_input->to(float_type);
        gpu_input = &converted_input;
      }
      StorageView gpu_bias, gpu_residual;
      if (bias)
        gpu_bias = bias->to(Device::CUDA).to(float_type);
      if (residual)
        gpu_residual = residual->to(Device::CUDA).to(float_type);
      StorageView result(float_type, Device::CUDA);
      const ops::Gemm gemm(1, 0, false, true, false, false,
                           weight_type == DataType::INT8 ? nullptr : nullptr);
      if (weight_type == DataType::INT8) {
        StorageView quantized(DataType::INT8, Device::CUDA),
            scales(DataType::FLOAT32, Device::CUDA);
        StorageView accumulator(DataType::INT32, Device::CUDA);
        ops::Quantize(ops::Quantize::ScaleType::PER_LAYER, false)(*gpu_input, quantized, scales);
        gemm(quantized, _gpu_weight, accumulator);
        _dequantize_op(accumulator, scales, _gpu_scale, false, true, result,
                       bias ? &gpu_bias : nullptr);
        if (residual)
          ops::Add()(gpu_residual, result, result);
      } else {
        StorageView decoded(DataType::FLOAT32, Device::CUDA);
        const StorageView* gpu_weight = &_gpu_weight;
        if (!_gpu_scale_inv.empty()) {
          ops::DequantizeFP8()(_gpu_weight, _gpu_scale_inv, decoded);
          if (decoded.dtype() != float_type)
            decoded = decoded.to(float_type);
          gpu_weight = &decoded;
        }
        gemm(*gpu_input, *gpu_weight, result, nullptr, bias ? &gpu_bias : nullptr,
             residual ? &gpu_residual : nullptr);
      }
      if (result.dtype() != DataType::FLOAT32)
        result = result.to(DataType::FLOAT32);
      if (output.device() == Device::CUDA) {
        output = std::move(result);
      } else {
        output.copy_from(result);
        synchronize_stream(Device::CUDA); // The following graph operations read host memory.
      }
    }

    DataType DramaProjection::output_type() const {
      return DataType::FLOAT32;
    }

    dim_t DramaProjection::output_size() const {
      return _partial_weight.empty() ? _weight.dim(0) : _partial_weight.dim(0);
    }

    void DramaProjection::select_weights(const StorageView* index, const StorageView* extra_bias) {
      release_device_weights();
      if (index && _fp8_scale_inv)
        throw std::invalid_argument("Selecting block FP8 projection weights is unsupported");
      if (_cpu)
        _cpu->select_weights(index, extra_bias);
      if (index) {
        ops::Gather()(_weight, *index, _partial_weight);
        if (_bias) {
          ops::Gather()(*_bias, *index, _partial_bias);
          if (extra_bias)
            ops::Add()(_partial_bias, *extra_bias, _partial_bias);
        } else if (extra_bias) {
          _partial_bias = *extra_bias;
        }
        if (_qscale && !_qscale->is_scalar())
          ops::Gather()(*_qscale, *index, _partial_qscale);
      } else {
        _partial_weight.clear();
        _partial_bias.clear();
        _partial_qscale.clear();
      }
    }

    void DramaProjection::operator()(const StorageView& input, StorageView& output,
                                     const StorageView* residual) const {
      if (_cpu) {
        (*_cpu)(input, output, residual);
        return;
      }
      const auto& weight = _partial_weight.empty() ? _weight : _partial_weight;
      const auto* bias = _partial_bias.empty() ? _bias : &_partial_bias;
      const auto* scale = _partial_qscale.empty() ? _qscale : &_partial_qscale;
      if (_gpu_device_index >= 0) {
        forward_gpu(input, weight, bias, scale, residual, output);
      } else {
        if (input.device() != Device::CPU || input.dtype() != DataType::FLOAT32)
          throw std::invalid_argument("Block FP8 projections require CPU float32 input");
        StorageView decoded;
        ops::DequantizeFP8()(weight, *_fp8_scale_inv, decoded);
        ops::Gemm(1, 0, false, true)(input, decoded, output, nullptr, bias, residual);
      }
    }

    DramaProjection::DeviceWeights DramaProjection::device_weights() const {
      if (_gpu_device_index < 0)
        throw std::invalid_argument("Device weights require GPU execution");
      const ScopedDeviceSetter setter(Device::CUDA, _gpu_device_index);
      prepare_gpu_weight(_partial_weight.empty() ? _weight : _partial_weight,
                         _partial_qscale.empty() ? _qscale : &_partial_qscale);
      return {_gpu_weight, _gpu_scale};
    }

    void DramaProjection::release_device_weights() const {
      _gpu_weight.release();
      _gpu_scale.release();
      _gpu_scale_inv.release();
    }

    size_t DramaProjection::device_weight_bytes() const {
      return _gpu_weight.reserved_memory() + _gpu_scale.reserved_memory()
             + _gpu_scale_inv.reserved_memory();
    }

    size_t DramaProjection::estimated_device_weight_bytes() const {
      const auto type = compute_type_to_data_type(_gpu_compute_type).first;
      if (type == DataType::INT8)
        return _weight.size() + _weight.dim(0) * sizeof(float);
      if (type == DataType::FLOAT32 && _fp8_scale_inv)
        return _weight.size() + _fp8_scale_inv->size() * sizeof(float);
      return _weight.size() * (type == DataType::FLOAT32 ? 4 : 2);
    }

  }
}
