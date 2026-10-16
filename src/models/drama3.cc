#include "ctranslate2/models/drama3.h"
#include <regex>
#include <stdexcept>
namespace ctranslate2 { namespace models {
  namespace {
    std::string quantization_policy(const nlohmann::json& config) {
      const auto policy = config.value("quantization_policy", std::string("all_matrices_int8"));
      if (policy != "all_matrices_int8" && policy != "experts_int8_sensitive_float32")
        throw std::invalid_argument("Unknown Drama3 quantization policy: " + policy);
      return policy;
    }
  }
  bool Drama3Model::is_quantizable(const std::string& name) const {
    // Model::set_compute_type runs before initialize. Enforce this policy here
    // so loading INT8 experts cannot silently re-quantize preserved F32 matrices.
    const auto policy = quantization_policy(config);
    if (name.compare(0,6,"codec/") == 0 || name.size() < 7
        || name.compare(name.size()-7,7,"/weight") != 0) return false;
    const auto* variable = get_variable_if_exists(name);
    if (!variable || variable->rank() != 2) return false;
    if (policy == "experts_int8_sensitive_float32") {
      static const std::regex routed("backbone/layers_[0-9]+/mlp/experts_[0-9]+/(gate_proj|up_proj|down_proj)/weight");
      return std::regex_match(name, routed);
    }
    return true;
  }
  bool Drama3Model::is_linear_weight(const std::string& name) const {
    return is_quantizable(name) && name.find("embedding") == std::string::npos;
  }
  void Drama3Model::initialize(ModelReader&) {
    quantization_policy(config);
    if (device() != Device::CPU)
      throw std::invalid_argument("Drama3 currently supports CPU execution only");
    if (effective_compute_type() != ComputeType::FLOAT32
        && effective_compute_type() != ComputeType::INT8_FLOAT32)
      throw std::invalid_argument("Drama3 requires float32 or int8_float32 compute");
    if (!config.is_object() || config.value("model_type",std::string()) != "drama3"
        || config.value("ct2_layout_version",0) != 1
        || !config.contains("source_config") || !config["source_config"].is_object())
      throw std::invalid_argument("Drama3 requires verified layout-v1 source configuration");
  }
  std::unique_ptr<Model> Drama3Model::clone() const {
    return std::make_unique<Drama3Model>(*this);
  }
}}
