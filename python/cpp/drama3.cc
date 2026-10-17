#include "module.h"
#include <pybind11/numpy.h>
#include <ctranslate2/models/drama3.h>

namespace ctranslate2 { namespace python {
  void register_drama3(py::module& m) {
    py::class_<models::Drama3>(m,"Drama3")
      .def(py::init<const std::string&,size_t,const std::string&,const std::string&,int>(),
           py::arg("model_path"),py::arg("intra_threads")=8,
           py::arg("compute_type")="default",
           py::arg("device")="cpu",py::arg("device_index")=0,
           py::call_guard<py::gil_scoped_release>())
      .def("generate",[](models::Drama3& model,
                         py::array_t<int32_t,py::array::c_style> prompt,
                         dim_t max_frames,dim_t prefill_chunk_size, float temperature,
                         dim_t top_k,float top_p,uint64_t seed,bool greedy) {
        if (prompt.ndim() != 2 || prompt.shape(1) != 11)
          throw std::invalid_argument("Drama3 prompt must have shape [rows,11]");
        StorageView input({prompt.shape(0),prompt.shape(1)},
                          std::vector<int32_t>(prompt.data(),prompt.data()+prompt.size()));
        models::Drama3Result result;
        {
          py::gil_scoped_release release;
          result = model.generate(input,max_frames,prefill_chunk_size,temperature,top_k,top_p,seed,greedy);
        }
        py::array_t<float> audio(result.audio.size());
        std::copy_n(result.audio.data<float>(),result.audio.size(),audio.mutable_data());
        py::array_t<int32_t> codes(std::vector<py::ssize_t>{result.codes.dim(0),10});
        std::copy_n(result.codes.data<int32_t>(),result.codes.size(),codes.mutable_data());
        py::dict output;
        output["audio"] = std::move(audio);
        output["codes"] = std::move(codes);
        output["sample_rate"] = result.sample_rate;
        output["finish_reason"] = result.finish_reason;
        py::dict timings;
        timings["prefill_seconds"] = result.timings.prefill_seconds;
        timings["backbone_decode_seconds"] = result.timings.backbone_decode_seconds;
        timings["tower_decode_seconds"] = result.timings.tower_decode_seconds;
        timings["codec_seconds"] = result.timings.codec_seconds;
        timings["total_native_seconds"] = result.timings.total_native_seconds;
        output["timings"] = std::move(timings);
        return output;
      },py::arg("prompt_rows"),py::arg("max_frames")=512,
         py::arg("prefill_chunk_size")=16,
         py::arg("temperature")=.8f,py::arg("top_k")=30,py::arg("top_p")=.8f,
         py::arg("seed")=0,py::arg("greedy")=false,
         "Generate offline speech with seeded sampling or diagnostic greedy decoding.");
  }
}}
