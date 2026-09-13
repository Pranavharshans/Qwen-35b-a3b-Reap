# GLM-5.3-Flash-BF16 support plan

- task_id: `RR-GLM53-01`
- objective: Add an exact, fail-closed donor contract for the official BF16 GLM checkpoint.
- input files and hashes: official `config.json` and weight index at pinned revision; current donor, config, and preflight code.
- expected outputs: donor registry entry, schema/config templates, sparse-layer-aware index validation, and tests.
- definition of done: the exact model ID, revision, 45 decoder layers with layers 0-2 dense, 42 sparse layers, 288 routed experts, top-8 routing, and per-expert BF16 tensor keys are enforced.
- validation command: `uv run pytest -q tests/test_config.py tests/test_model_preflight.py tests/test_extraction.py tests/test_qwen35.py`
- estimated GPU hours: `0`
- estimated storage: `<1 MiB` in Git; metadata-only tests use temporary fixtures.
- dependencies: fresh human GLM approval; official Hugging Face metadata.
- failure behavior: reject revision, architecture, sparse-layer set, or tensor-layout drift before weight download.

- task_id: `RR-GLM53-02`
- objective: Make the fused runtime adapter preserve GLM's absolute sparse layer identities.
- input files and hashes: passed `RR-GLM53-01`; Transformers GLM5 Next module structure.
- expected outputs: sparse-layer-aware architecture inspection, telemetry, interventions, and extraction.
- definition of done: runtime hooks operate only on sparse layers 3-44, emit absolute layer IDs, reproduce GLM's clamped SwiGLU side path, and keep Qwen behavior unchanged.
- validation command: `uv run pytest -q tests/test_qwen35.py tests/test_instrumentation.py tests/test_runtime.py`
- estimated GPU hours: `0`
- estimated storage: `<1 MiB`.
- dependencies: `RR-GLM53-01`.
- failure behavior: fail closed on a dense/sparse mismatch or unsupported expert module shape; require an exact-checkpoint Gate A GPU probe before calibration.
