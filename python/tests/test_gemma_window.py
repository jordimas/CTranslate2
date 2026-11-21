"""Parity at window boundaries, including mixed attention and padded batches."""

import numpy as np
import pytest
import test_utils

import ctranslate2


@test_utils.only_on_linux
@pytest.mark.parametrize("architecture", ["Gemma2", "Gemma3", "Mistral"])
@pytest.mark.parametrize("num_key_value_heads", [1, 2, 4])
@pytest.mark.parametrize("window", [5, 8])
def test_windowed_scoring(architecture, num_key_value_heads, window, tmp_dir):
    import torch
    import transformers
    from ctranslate2.converters import transformers as converters

    torch.manual_seed(42)
    loader = getattr(converters, architecture + "Loader")()
    model_class = getattr(transformers, loader.architecture_name)
    config_class = (
        transformers.Gemma3TextConfig
        if architecture == "Gemma3"
        else model_class.config_class
    )
    config = config_class(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=num_key_value_heads,
        head_dim=8,
        query_pre_attn_scalar=9,
        sliding_window=window,
        max_position_embeddings=256,
        attn_logit_softcapping=None,
        final_logit_softcapping=None,
    )
    if architecture == "Gemma3":
        config.layer_types = ["sliding_attention", "full_attention"] * 2
    config._attn_implementation = "eager"
    model = model_class(config).eval()
    sequences = [
        [2] + torch.randint(4, 64, (length - 1,)).tolist()
        for length in (5, 8, 9, 10, 17, 24, 40)
    ]
    expected = []
    with torch.no_grad():
        for ids in sequences:
            logits = model(input_ids=torch.tensor([ids]), use_cache=False).logits
            expected.append(
                logits[0, :-1]
                .log_softmax(-1)
                .gather(1, torch.tensor(ids[1:])[:, None])[:, 0]
                .numpy()
            )
        spec = loader.get_model_spec(model)
    tokens = ["<pad>", "<eos>", "<bos>", "<unk>"] + [str(i) for i in range(4, 64)]
    loader.set_vocabulary(spec, tokens)
    spec.config.bos_token = "<bos>"
    spec.config.eos_token = "<eos>"
    spec.config.unk_token = "<unk>"
    spec.config.layer_norm_epsilon = config.rms_norm_eps
    spec.validate()
    spec.optimize(quantization="float32")
    spec.save(str(tmp_dir))
    generator = ctranslate2.Generator(
        str(tmp_dir), device="cpu", compute_type="float32", intra_threads=1
    )
    for batch in [[i] for i in range(len(sequences))] + [[6, 0], [3, 1, 0]]:
        actual = generator.score_batch(
            [[tokens[token] for token in sequences[i]] for i in batch],
            max_input_length=0,
        )
        for i, result in zip(batch, actual):
            np.testing.assert_allclose(
                result.log_probs, expected[i], atol=2e-5, rtol=2e-5
            )


@test_utils.only_on_linux
def test_encoder_window_is_not_causal(tmp_dir):
    import torch
    import transformers
    from ctranslate2.converters.transformers import T5GemmaLoader

    torch.manual_seed(42)
    module = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        query_pre_attn_scalar=8,
        sliding_window=64,
        attn_logit_softcapping=None,
        final_logit_softcapping=None,
    )
    config = transformers.T5GemmaConfig(
        encoder=dict(module), decoder=dict(module, is_decoder=True), vocab_size=64
    )
    config._attn_implementation = "eager"
    model = transformers.T5GemmaForConditionalGeneration(config).eval()
    source, target = [2, 4, 5, 6], [7, 8, 9]
    with torch.no_grad():
        logits = model(
            input_ids=torch.tensor([source]),
            decoder_input_ids=torch.tensor([[2] + target]),
            use_cache=False,
        ).logits[0]
        expected = (
            logits.log_softmax(-1)
            .gather(1, torch.tensor(target + [1])[:, None])[:, 0]
            .numpy()
        )
        loader = T5GemmaLoader()
        spec = loader.get_model_spec(model)
    # A decoder's causal window must not be applied to bidirectional encoders.
    for layer in spec.encoder.layer:
        layer.self_attention.sliding_window = np.int32(2)
    tokens = ["<pad>", "<eos>", "<bos>", "<unk>"] + [str(i) for i in range(4, 64)]
    loader.set_vocabulary(spec, tokens)
    spec.config.bos_token = "<bos>"
    spec.config.eos_token = "<eos>"
    spec.config.unk_token = "<unk>"
    spec.config.decoder_start_token = "<bos>"
    spec.config.layer_norm_epsilon = config.decoder.rms_norm_eps
    spec.validate()
    spec.optimize(quantization="float32")
    spec.save(str(tmp_dir))
    translator = ctranslate2.Translator(
        str(tmp_dir), device="cpu", compute_type="float32", intra_threads=1
    )
    result = translator.score_batch(
        [[tokens[i] for i in source]], [[tokens[i] for i in target]], max_input_length=0
    )[0]
    np.testing.assert_allclose(result.log_probs, expected, atol=2e-5, rtol=2e-5)
