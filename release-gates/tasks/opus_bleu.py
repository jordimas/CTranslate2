"""Translation BLEU task.

Model: Helsinki-NLP/opus-mt-en-de (converted on the fly).
Data:  sacrebleu wmt14 en-de, optionally capped via --limit.
Metric: BLEU + tokens_per_sec.
"""

import os
import tempfile
import time

MODEL_NAME = "Helsinki-NLP/opus-mt-en-de"
TEST_SET = "wmt14"
LANG_PAIR = "en-de"


def _convert_model(output_dir):
    from ctranslate2.converters.transformers import TransformersConverter
    TransformersConverter(MODEL_NAME).convert(output_dir, force=True)


def _load_test_set(limit):
    import sacrebleu
    src = sacrebleu.utils.download_test_set(TEST_SET, LANG_PAIR)
    # sacrebleu returns a dict of file paths keyed by role; we need src + ref.
    src_lang, tgt_lang = LANG_PAIR.split("-")
    src_path = sacrebleu.utils.get_source_file(TEST_SET, LANG_PAIR)
    ref_paths = sacrebleu.utils.get_reference_files(TEST_SET, LANG_PAIR)
    with open(src_path, encoding="utf-8") as f:
        sources = [line.rstrip("\n") for line in f]
    refs = []
    for ref_path in ref_paths:
        with open(ref_path, encoding="utf-8") as f:
            refs.append([line.rstrip("\n") for line in f])
    if limit is not None:
        sources = sources[:limit]
        refs = [r[:limit] for r in refs]
    return sources, refs


def run(limit=None, device="cpu"):
    import ctranslate2
    import sacrebleu
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    sources, refs = _load_test_set(limit)

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = os.path.join(tmp, "ct2")
        _convert_model(model_dir)

        translator = ctranslate2.Translator(model_dir, device=device)

        tokenized = [tokenizer.convert_ids_to_tokens(tokenizer.encode(s)) for s in sources]
        total_tokens = sum(len(t) for t in tokenized)

        start = time.perf_counter()
        results = translator.translate_batch(tokenized, beam_size=4, max_batch_size=16)
        elapsed = time.perf_counter() - start

    hypotheses = []
    for r in results:
        out_tokens = r.hypotheses[0]
        text = tokenizer.decode(
            tokenizer.convert_tokens_to_ids(out_tokens),
            skip_special_tokens=True,
        )
        hypotheses.append(text)

    bleu = sacrebleu.corpus_bleu(hypotheses, refs).score

    return {
        "bleu": bleu,
        "tokens_per_sec": total_tokens / elapsed if elapsed > 0 else 0.0,
        "num_sentences": len(sources),
        "elapsed_sec": elapsed,
    }
