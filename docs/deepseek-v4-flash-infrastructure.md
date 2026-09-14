# DeepSeek V4 Flash launch infrastructure

This repository contains an infrastructure-only launcher for the dedicated
`deepseek-ai/DeepSeek-V4-Flash-0731` checkpoint. It provides two transports for
one serving workload:

| Transport | Use when | State-changing flag |
| --- | --- | --- |
| `direct` | The operator already has local or cloud GPU access | `--execute` |
| `fau_slurm` | The operator is on FAU Alex and needs the native Slurm scheduler | `--submit` |

The default command is a read-only preview. `--check` is also read-only. The
launcher does not provision a machine, inspect scheduler availability, download
weights, run a GPU process, or submit a job during implementation. A real
`--execute` or `--submit` invocation is a separate, explicitly authorized
operator action.

## Scope and provenance

This is a separate DeepSeek track. It does not modify the Qwen v0 donor,
experiment schemas, datasets, run state, attribution logic, or extraction
artifacts. The Qwen v0 scientific contract remains governed by `AGENTS.md`,
`prd.md`, and `roadmap.md`.

The scope and authorization record for this slice is [the DeepSeek
infrastructure decision](decisions/2026-09-14-deepseek-v4-flash-infrastructure.md).

The model and revision are fixed in both example configurations:

```text
model: deepseek-ai/DeepSeek-V4-Flash-0731
revision: 7872f01b1d1fe23eabc4c98b48bffcef5a386062
official pin captured: 2026-09-14
```

The pin is an immutable source identity, not evidence that this repository has
run the checkpoint. The implementation has no exact-checkpoint GPU, vLLM,
Slurm, throughput, memory, or quality validation. Perform a bounded hardware
preflight and a smoke request before any larger workload.

At the 2026-09-14 metadata preflight, the [pinned `config.json`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json)
declared:

- `architectures: [DeepseekV4ForCausalLM]` and `model_type: deepseek_v4`;
- 43 transformer layers;
- 256 routed experts per MoE layer with top-6 routing;
- `max_position_embeddings: 1,048,576`; and
- FP8 quantization metadata (`quant_method: fp8`, `fmt: e4m3`) alongside
  `expert_dtype: fp4`.

These are declarations from pinned configuration metadata only. They are not a
runtime claim: the repository has not loaded the exact checkpoint or proven
that the declared context, FP8 path, expert layout, or memory requirement works
on any particular GPU.

The [0731 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
describes a dedicated DeepSeek V4 encoding path. Its tokenizer
metadata does not supply a Jinja chat template, so the canonical vLLM command
selects the dedicated `deepseek_v4` tokenizer, reasoning, and tool-call parsers.
The command also includes `--trust-remote-code`, which is required for
checkpoint-specific code. Enabling it executes model-repository Python code;
review the pinned revision and use an isolated environment.

## Installation and prerequisites

The launcher itself is hardware-free Python code. The checked-in project
requires Python 3.12 or newer, `uv`, Pydantic 2, and PyYAML. Install the
development environment from the lockfile:

```bash
UV_CACHE_DIR=/tmp/reverse-reap-uv-cache uv sync --frozen --extra dev
```

The serving host additionally needs a CUDA/PyTorch environment with a vLLM
release satisfying the configuration's `vllm_min_version` (`0.20.0` in the
examples), plus eight compatible GPUs for the conservative TP8/PP1 example.
Choose a vLLM/PyTorch/CUDA combination supported by the target host and verify
it before execution. No vLLM installation or GPU validation was performed as
part of this implementation.

Do not put Hugging Face tokens, provider credentials, FAU account identifiers,
private paths, or model files in the repository. Authenticate through the
operator's existing environment only after reviewing the generated command.

## Configuration files

The two reviewable examples are:

```text
configs/deepseek-v4-flash-direct.yaml
configs/deepseek-v4-flash-fau-slurm.yaml
```

They intentionally contain absolute placeholder paths, example run IDs, and,
for FAU, placeholder site values. This makes the examples safe to inspect but
not accidentally launchable. Replace every unresolved value only in a private
working copy or through a reviewed local edit.

### Field reference

| Field | Meaning and constraints |
| --- | --- |
| `schema_version` | Must be integer `1`. Unknown fields are rejected. |
| `run_id` | Filesystem-safe, operator-visible identifier; a suffix such as `-example` is treated as unresolved for state-changing actions. |
| `mode` | Exactly `direct` or `fau_slurm`; selects the permitted state-changing action. |
| `model.id` | Exactly `deepseek-ai/DeepSeek-V4-Flash-0731`. |
| `model.revision` | Exactly the 40-character SHA above; other revisions fail closed. |
| `model.trust_remote_code` | Must remain `true` for the dedicated model code. |
| `serving.vllm_min_version` | Informational minimum; examples use `0.20.0`. It does not install vLLM. |
| `serving.tensor_parallel_size` | TP degree; examples use `8`. |
| `serving.pipeline_parallel_size` | PP degree; examples use `1`. |
| `serving.gpu_count` | Optional declared GPU count; when present it must equal TP × PP and FAU allocation. |
| `serving.max_model_len` | Explicit context setting; examples use `32768`, not the metadata-declared 1M limit. |
| `serving.gpu_memory_utilization` | Explicit vLLM GPU memory fraction; examples use `0.90`. |
| `serving.host` / `serving.port` | Explicit bind address and port; examples use `127.0.0.1:8000`. |
| `serving.extra_args` | Additional vLLM tokens. Canonical model, parser, parallelism, context, bind, memory, cache, and revision flags are protected. |
| `paths.model_cache_dir` | Passed to vLLM as `--download-dir`; the model/cache root. |
| `paths.run_dir` | Reserved per-run artifact directory; created by an explicit launch, not passed to vLLM. |
| `paths.output_dir` | Reserved output/artifact directory; launchers do not redirect stdout here. |
| `paths.cache_dir` | Reserved auxiliary cache directory; it is not a vLLM cache flag. |
| `paths.log_dir` | Reserved log directory; direct mode inherits process stdout/stderr, while FAU Slurm logs use `fau_slurm.output_path` and `error_path`. |
| `direct.working_directory` | Optional direct-process cwd; it is not a model or cache path. |
| `fau_slurm.job_name` | Slurm-safe job name. |
| `fau_slurm.account` / `partition` / `constraint` | Operator-supplied single-token FAU site values; spaces and shell metacharacters are rejected, and placeholders are rejected for submission. |
| `fau_slurm.nodes` / `tasks_per_node` | Exactly `1` until a tested distributed vLLM wrapper exists. |
| `fau_slurm.gpus_per_node` | Must match the declared serving GPU count; examples use `8`. |
| `fau_slurm.cpus_per_task` / `memory_gb` / `time_limit` | Explicit Slurm resource directives. They are requests, not validated availability. |
| `fau_slurm.output_path` / `error_path` | Slurm stdout/stderr files. Their parent directories must already exist on FAU before `--submit`. |
| `fau_slurm.module_commands` | Simple inert commands beginning with `module `, emitted in order; shell metacharacters are rejected. |
| `fau_slurm.environment_activation` | Optional command beginning with `source ` or `conda activate `; shell metacharacters are rejected. |

All config models are frozen Pydantic models with `extra="forbid"` and strict
scalar validation. Paths are absolute and newline/control characters are
rejected. The renderer uses argv lists for direct mode and shell quoting for
the generated Slurm script.

## Canonical workload

Both transports render the same workload, including:

```text
vllm serve deepseek-ai/DeepSeek-V4-Flash-0731
  --revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062
  --trust-remote-code
  --tokenizer-mode deepseek_v4
  --reasoning-parser deepseek_v4
  --tool-call-parser deepseek_v4
  --enable-auto-tool-choice
  --kv-cache-dtype fp8
  --tensor-parallel-size 8
  --pipeline-parallel-size 1
  --max-model-len 32768
  --host 127.0.0.1
  --port 8000
  --gpu-memory-utilization 0.9
  --download-dir /absolute/path/to/deepseek-v4-flash/model-cache
```

The actual rendered command is an argv list in direct mode and a shell-quoted
`exec` line in FAU mode. `extra_args` cannot replace any canonical flag,
including the model revision, parser selections, TP/PP, context, bind,
memory, or download directory.

## Preview and check

Run these read-only commands before changing any path or site value:

```bash
uv run reverse-reap deepseek-launch \
  configs/deepseek-v4-flash-direct.yaml

uv run reverse-reap deepseek-launch \
  configs/deepseek-v4-flash-direct.yaml --check

uv run reverse-reap deepseek-launch \
  configs/deepseek-v4-flash-fau-slurm.yaml

uv run reverse-reap deepseek-launch \
  configs/deepseek-v4-flash-fau-slurm.yaml --check
```

The default preview prints JSON containing the direct argv/command or the
complete generated Slurm script. `--check` prints the hardware-free preflight.
Placeholder values, including the checked-in example run ID, appear as warnings
in both modes. Wrong model IDs, revisions, types, unknown fields, protected
overrides, and unsafe control characters fail closed. Neither operation creates
directories or invokes `vllm`, `sbatch`, a network client, or a GPU process.

## Direct/cloud execution

`--execute` is accepted only for `mode: direct`. Before requesting fresh
authorization, replace the example run ID and every placeholder path, confirm
the target GPU/CUDA/PyTorch/vLLM stack, confirm the exact revision is available,
and review the previewed argv. Then run the explicit command:

```bash
uv run reverse-reap deepseek-launch \
  /private/path/to/reviewed-direct.yaml --execute
```

The launcher creates the configured model-cache, run, output, auxiliary-cache,
and reserved-log directories, then invokes `subprocess.run` with the canonical
argv, `shell=False`, and the optional direct working directory. It does not
redirect stdout/stderr into `output_dir` or `log_dir`; capture those streams at
the process supervisor/operator layer if desired. If the model is not already
cached, vLLM may download it into `model_cache_dir` as part of this explicit
operator action.

Do not use `--execute` with the FAU example. The launcher rejects that transport
mismatch before any subprocess call.

## FAU Slurm execution

`--submit` is accepted only for `mode: fau_slurm`. The wrapper is deliberately
single-node and single-task (`nodes: 1`, `tasks_per_node: 1`) because a tested
multi-node vLLM orchestration path is not part of this infrastructure.

Slurm opens `--output` and `--error` before executing the script body. Therefore,
on the FAU host, create both parent directories first and verify their exact
paths. This is a separate state-changing shell action and requires the same
fresh operator authorization as submission. The launcher never creates those
external FAU log parents locally.

After replacing the example run ID and all site/path placeholders, adding only
cluster-confirmed commands beginning with `module ` and `source ` or
`conda activate `, and creating the log parents,
preview again, run `--check`, and obtain fresh explicit authorization for
submission:

```bash
uv run reverse-reap deepseek-launch \
  /private/path/to/reviewed-fau-slurm.yaml

uv run reverse-reap deepseek-launch \
  /private/path/to/reviewed-fau-slurm.yaml --check

uv run reverse-reap deepseek-launch \
  /private/path/to/reviewed-fau-slurm.yaml --submit
```

The submit path renders one deterministic script and invokes exactly `sbatch`
with that script on standard input. It does not call `srun`, probe scheduler
state, submit a second job, or create the Slurm output/error parents. A missing
or non-directory output/error parent is a warning in preview/check and a hard
error before `sbatch` during `--submit`.

The generated script contains:

1. a Bash shebang and `#SBATCH` job name, account, partition, constraint,
   node/task/GPU, CPU, memory, time, output, and error directives;
2. `set -euo pipefail`;
3. `mkdir -p` for the model-cache, run, output, auxiliary-cache, and reserved
   log directories (not the externally opened Slurm output/error parents);
4. configured module commands in order;
5. the optional environment activation command; and
6. one shell-quoted `exec` line containing the same canonical vLLM workload as
   direct mode.

The checked-in FAU example leaves `module_commands` empty and uses a placeholder
activation path. Add commands only after checking the target FAU environment;
the launcher does not guess or install site modules.

Do not use `--submit` with the direct example. The launcher rejects the mismatch
before invoking `sbatch`.

## Readiness and health probes

The following probes are operator checks after the explicitly authorized server
is running. They are not run by the launcher and do not prove exact-checkpoint
quality or compatibility:

```bash
export DEEPSEEK_BASE_URL=http://127.0.0.1:8000

curl -fsS "$DEEPSEEK_BASE_URL/health"
curl -fsS "$DEEPSEEK_BASE_URL/v1/models"

curl -fsS "$DEEPSEEK_BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V4-Flash-0731","messages":[{"role":"user","content":"Reply with exactly: ready"}],"max_tokens":16,"temperature":0}'
```

For a remote cloud or FAU node, use an already authorized network route or SSH
tunnel and set `DEEPSEEK_BASE_URL` accordingly. Keep the model ID in the request
exactly pinned; do not infer it from a different `/v1/models` result.

## Troubleshooting

| Symptom | Safe interpretation and next step |
| --- | --- |
| `placeholder values require operator replacement` | Expected for the checked-in examples. Edit a private copy, preview again, and do not bypass the guard. |
| Revision/model validation failure | Stop. Re-check the exact 0731 ID and SHA; do not substitute a nearby DeepSeek checkpoint. |
| `extra_args cannot override protected...` | Remove the override and change only the dedicated serving fields in a reviewed config. |
| Missing FAU output/error parent | Create both parents on the FAU host before submission; preview/check warn, submit blocks. |
| `sbatch` rejects a directive | Re-check account, partition, constraint, resource syntax, and site policy. Availability is not validated locally. |
| `vllm` is missing or too old | Install/activate a compatible target-host environment meeting `vllm_min_version`; no installation is performed by this launcher. |
| Port already in use | Stop the conflicting authorized process or choose a reviewed bind/port change; re-preview before execution. |
| CUDA OOM or unsupported FP8 path | Stop the run, preserve logs, and perform a bounded hardware/runtime probe. Do not silently lower context, change precision, or switch checkpoints after observing a result. |
| `/health` works but chat fails | Inspect vLLM logs, tokenizer/parser support, model availability, and request model ID. A 200 health response is not model validation. |
| Declared 1M context is tempting | It is metadata only. The examples intentionally launch with 32K until exact runtime behavior and memory are measured. |

## Sources (accessed 2026-09-14)

The model identity and metadata statements above are tied to these public
sources. The vLLM links describe support/recipe surfaces; they do not replace
an exact-checkpoint smoke test on the operator's hardware.

- [DeepSeek-V4-Flash-0731 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- [Pinned `config.json` at revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json)
- [Official vLLM DeepSeek-V4 recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash)
- [vLLM supported-model registry](https://docs.vllm.ai/en/latest/models/supported_models/)

## Authorization and delivery boundary

Preview and check are read-only. `--execute`, `--submit`, creating FAU log
parents, downloading the model, and provisioning or paying for GPU capacity are
state-changing actions that require fresh explicit authorization in the
operator's environment. This implementation performed none of them: no GPU
process, model download, cloud instance, or Slurm job occurred.

“Push to Hub” for this repository means delivering the source/config/docs to the
configured GitHub `origin` repository. It does not mean uploading the DeepSeek
checkpoint, extracted tensors, datasets, credentials, or provider artifacts to
Hugging Face. No model-weight publication is performed automatically.
