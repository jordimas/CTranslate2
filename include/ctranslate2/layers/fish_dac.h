#pragma once

#include "common.h"

namespace ctranslate2 {
  namespace layers {
    // Converts upsampled codec latents into audio waveform samples.
    // Offline waveform portion of Fish DAC. Input is the already upsampled
    // latent [1,C,T]; codebooks/post-transformer/ConvNeXt precede this layer.
    // CPU float32 weights, materialized weight normalization, no packing.
    class FishDACWaveformDecoder {
    public:
      FishDACWaveformDecoder(const models::Model& model, const std::string& scope,
                            std::vector<dim_t> strides = {8, 8, 4, 2});
      void operator()(const StorageView& input, StorageView& output) const;
    private:
      // References the weights and optional bias of a waveform convolution.
      struct Convolution {
        const StorageView* weight;
        const StorageView* bias;
      };
      // Groups the activations and convolutions of a waveform residual unit.
      struct Residual {
        const StorageView* alpha0;
        Convolution conv0;
        const StorageView* alpha1;
        Convolution conv1;
      };
      // Groups one waveform upsampling stage and its residual units.
      struct Upsample {
        const StorageView* alpha;
        Convolution transpose;
        std::vector<Residual> residuals;
        dim_t stride;
      };
      static void causal(const StorageView& input, const Convolution& conv,
                         dim_t dilation, StorageView& output);
      Convolution _initial, _final;
      const StorageView* _final_alpha;
      std::vector<Upsample> _blocks;
      std::vector<const StorageView*> _parameters;
    };
  }
}
