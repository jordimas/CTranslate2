"""Gemma2 attention softcapping uses a versioned runtime operation."""

import numpy as np
import pytest
import test_utils

import ctranslate2


@test_utils.only_on_linux
@pytest.mark.parametrize("cap", [None, 0.0, 0.8])
def test_gemma2_attention_softcapping(cap, tmp_dir):
    import torch
    import transformers
    from ctranslate2.converters.transformers import Gemma2Loader

    torch.manual_seed(42)
    config = transformers.Gemma2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        query_pre_attn_scalar=8,
        sliding_window=64,
        attn_logit_softcapping=None if cap == 0 else cap,
        final_logit_softcapping=None,
    )
    config._attn_implementation = "eager"
    model = transformers.Gemma2ForCausalLM(config).eval()
    ids = [2, 5, 9, 14, 11]
    with torch.no_grad():
        expected = model(torch.tensor([ids]), use_cache=False).logits.numpy()
        config.attn_logit_softcapping = cap
        loader = Gemma2Loader()
        spec = loader.get_model_spec(model)
    tokens = ["<pad>", "<eos>", "<bos>", "<unk>"] + [str(i) for i in range(4, 64)]
    loader.set_vocabulary(spec, tokens)
    spec.config.bos_token = "<bos>"
    spec.config.eos_token = "<eos>"
    spec.config.unk_token = "<unk>"
    spec.config.layer_norm_epsilon = config.rms_norm_eps
    assert spec.revision == (9 if cap else 8)
    spec.validate()
    spec.optimize(quantization="float32")
    spec.save(str(tmp_dir))
    generator = ctranslate2.Generator(
        str(tmp_dir), device="cpu", compute_type="float32", intra_threads=1
    )
    actual = np.asarray(generator.forward_batch([[tokens[i] for i in ids]]))
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
