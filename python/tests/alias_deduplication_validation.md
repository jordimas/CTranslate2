# Alias deduplication conversion validation

Compared on 2026-10-20 at branch commit `524b6bf634d6e1fb85811569567ce2602aca934c`.
The baseline used `_alias_variables` from master commit
`0495cfa02ef28e069c5b3c867703a83828aafa05`; all other conversion code was kept
identical to isolate the deduplication change.

Environment: macOS arm64, Python 3.13.2, PyTorch 2.14.1, Transformers 4.57.1,
NumPy 2.5.4, safetensors 0.8.0.

Each checkpoint was loaded once using `TransformersConverter._load()`. For each
precision, independent deep copies were converted with master and branch alias
implementations. Compared alias maps, all resolved tensor values, output sizes,
and SHA-256 hashes of `model.bin`.

| Model | Precision | Bytes (both versions) | Aliases (both versions) | Result |
| --- | --- | ---: | ---: | --- |
| sshleifer/tiny-gpt2 | float32 | 413,001 | 15 | Byte-identical |
| sshleifer/tiny-gpt2 | int8 | 312,863 | 15 | Byte-identical |
| sshleifer/bart-tiny-random | float32 | 5,096,240 | 43 | Byte-identical |
| sshleifer/bart-tiny-random | int8 | 1,631,832 | 43 | Byte-identical |

## Checkpoint revisions

- `sshleifer/bart-tiny-random`: `69bce9237e4fa10ea015446395ec0108067890cf`
- `sshleifer/tiny-gpt2`: `5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be`

## Matching output hashes

- `sshleifer/tiny-gpt2` / `float32`: `f440e3533406c0aea809992660f6eb3062cb96d16ffc42aac9bba17ef6e06dcb`
- `sshleifer/tiny-gpt2` / `int8`: `514a54d99f081c5dad56320e13699f9445c9a54cdd772ba4a73f0b9266de7d47`
- `sshleifer/bart-tiny-random` / `float32`: `5a8bc997ce44084c99e19dee1b45efb74a467f5f49122a61f2027fbf99f536df`
- `sshleifer/bart-tiny-random` / `int8`: `273b64464b50b95f7a73ac6479ae7774c76701bc5f29b8d0c7b32015fa63c929`

All four comparisons had identical alias maps and no resolved tensor differences.
These small GPT-2 and BART checkpoints provide compatibility evidence for two
architectures, not a guarantee for every converter. Equal-valued tensors with
different variable classes or quantization eligibility can still produce different
alias decisions under the new implementation. No inference or speed benchmark
was performed. Downloaded weights and generated model binaries are not included.
