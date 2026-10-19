#pragma once

#include "ctranslate2/models/model.h"
#include "env.h"

namespace ctranslate2 {
  namespace layers {

    inline bool drama_gpu_resident(const models::Model& model) {
      return model.config.contains("drama3_gpu_compute_type")
             && model.config.value("drama3_gpu_resident_backbone",
                                   read_bool_from_env("CT2_DRAMA_GPU_RESIDENT_BACKBONE", true));
    }

  }
}
