# Qwen3.8 Flash Next FP8 support plan

Status: implementation plan (2026-09-12)

- task_id: `RR-Q38-FP8-01`
- objective: Register and fail-closed preflight the official blockwise-FP8 donor without changing BF16 evidence.
- input files and hashes: official model revision `236dfdf285828023ca3bcd3f37366c58a3469b13`, official config quantization metadata, and official 131-shard weight index.
- expected outputs: donor/config contract, FP8 templates, quantization and index-layout validation, tests, and documentation.
- definition of done: model ID, immutable revision, qwen4_exp architecture, dynamic activation scheme, 128x128 block weights, per-expert weight and scale keys, and separate run identity are enforced.
- validation command: `UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv run pytest -q tests/test_config.py tests/test_model_preflight.py tests/test_extraction.py`
- estimated GPU hours: `0`.
- estimated storage: `< 1 MB` repository changes; remote metadata inspection used one temporary 17 MB index only.
- dependencies: existing Qwen3.8 BF16 contract.
- failure behavior: reject missing/mismatched FP8 scales or metadata before downloading weights.

- task_id: `RR-Q38-FP8-02`
- objective: Preserve FP8 expert weights and inverse-scale tensors losslessly during extraction.
- input files and hashes: pinned FP8 weight index, passed candidate manifest, exact source shards.
- expected outputs: six byte-verified tensors per selected expert: gate/up/down weights and their inverse scales.
- definition_of_done: no cast, dequantization, fusion, or scale loss occurs and independent reload matches source bytes.
- validation command: `UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv run pytest -q tests/test_extraction.py`
- estimated GPU hours: `0` for implementation; extraction runtime is separately budgeted.
- estimated storage: proportional to selected expert tensors and declared at runtime.
- dependencies: `RR-Q38-FP8-01`.
- failure behavior: fail the extraction if any weight or scale tensor is absent or differs.
