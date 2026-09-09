# Bridge-training v1

Bridge-training v1 is an opt-in follow-on to the Reverse-REAP donor handoff. It
does not change the v0 causal label or donor tensors. The host is the frozen,
post-trained `Qwen/Qwen3.5-2B` checkpoint, selected because its verified hidden
size is 2,048. The exact host revision must be pinned before a run.

## Contract

The bridge is a parallel sidecar attached at an explicitly configured host MLP
layer for each donor layer-expert pair:

```text
host MLP hidden state
  -> trainable 2048 -> bottleneck -> 2048 input adapter
  -> exact frozen extracted donor SwiGLU expert
  -> trainable 2048 -> bottleneck -> 2048 output adapter
  -> capped sigmoid gate (initial bias -6)
  -> residual added to the host MLP path
```

The final output-adapter projection is zero-initialized, making the initial
sidecar an identity while the finite gate bias preserves a gradient path. The
host and extracted expert parameters are always frozen. Only the adapters and
gate are saved in `bridge.safetensors`; host weights and donor checkpoint files
are never copied into a bridge artifact.

The donor-to-host layer map is required in `configs/bridge-qwen35-2b.yaml`.
The implementation never guesses a map from layer counts. The example map is a
template and must be reviewed and frozen as part of a bridge generation.

## Data preparation

The donor VM handoff must be validated first:

```bash
reverse-reap validate-target-handoff /path/to/handoff-manifest.json
```

Host hidden states are captured from the frozen Qwen host into a small
`bridge-host-states` manifest. Its `records.jsonl` contains one record per
`(sample_content_sha256, token_position, host_layer)` and its safetensors file
contains the corresponding hidden vectors under the declared tensor key. The
capture hook is attached to the host decoder layer's MLP input, which is the
parallel-sidecar insertion point (not a post-block approximation). The host
state manifest must include hashes for both files and a manifest hash.

The capture helper uses the donor handoff's exact teacher-forced token rows and
fails on any host/donor token-ID mismatch:

```bash
reverse-reap capture-host-states /path/to/Qwen3.5-2B /path/to/tokenizer \
  /path/to/handoff-manifest.json /path/to/capture-manifest.json \
  /path/to/host-states \
  --host-revision <40-64-char-host-sha> --run-id <bridge-run-id> \
  --allow-observational-coverage-incomplete \
  --mapping 3:26:2 --mapping 7:18:4 --mapping 35:239:18 --mapping 37:5:20
```

Repair and freeze the sample-grouped training manifest after both artifacts
exist:

```bash
reverse-reap repair-bridge-manifest-raw \
  /path/to/handoff-manifest.json /path/to/host-states-manifest.json \
  /path/to/bridge-training-manifest.json \
  --allow-observational-coverage-incomplete --min-events-per-cell 32 \
  --mapping 3:26:2 --mapping 7:18:4 --mapping 35:239:18 --mapping 37:5:20
```

Repair uses deterministic iterative multilabel stratification over
`sample_content_sha256` groups. No content hash can cross train, validation, or
test. Every donor-expert/domain cell must have the configured minimum number of
sample groups and event rows in all three splits or the command fails closed.
The incomplete-capture flag explicitly acknowledges the preserved coverage
failure for observational bridge development; it does not upgrade the experts
to causal or coding-critical status. Per-sample caps,
missing host states, and any other excluded source rows remain in
`unused_rows` with a reason; they are not silently dropped.

The trainer rechecks the handoff, extraction, host-state, and training-manifest
hashes. It also checks target and host token IDs for every row before training.
The current trainer is vector-supervision only: it learns against precomputed
host MLP vectors and captured donor expert vectors. It does not compute a
language-model loss, fine-tune the host, or claim full host end-to-end quality.
After this repair step, copy the emitted training-manifest hash into the pinned
training config. The config-based `repair-bridge-manifest` command is available
when a provisional config already contains the required 64-character hashes.

## Controls

The implemented evaluation conditions are `disabled`, `untrained`, `trained`,
`shuffled-pair`, and `equal-parameter-adapter`. `random-expert` is intentionally
unavailable until a separately verified random extraction artifact exists; an
absent control is recorded as unavailable rather than fabricated. Training
does not claim these controls were evaluated: the report records each as
`evaluated: false` until a separate end-to-end evaluation run executes it.

Validation loss drives early stopping. Test rows are loaded only after the best
validation checkpoint is selected and are never used for gradients or stopping.

The budget block is mandatory and includes a provider rate, hard GPU-hour and
storage ceilings, and a timezone-aware deadline. The checked-in example uses
an expired deadline on purpose; replace it only in a separately approved,
hash-bound run configuration.

## CPU-only validation and later GPU gate

Configuration and split repair do not require Torch. Model construction and
training require the optional GPU dependencies. A later machine may run the
read-only B1 memory estimate without loading the host:

```bash
reverse-reap bridge-preflight $((16 * 1024 * 1024 * 1024)) \
  --host-parameter-count 2000000000 --sequence-tokens 1024
```

The estimate requires batch size 1 and applies a 92% VRAM ceiling. It is a
preflight only; the first actual run must measure peak VRAM and stop on OOM or a
peak above the declared ceiling.

## Launch sequence

The template is deliberately unlaunchable until all `REQUIRED_*` values are
replaced with hashes from the immutable run bundle and the host revision is a
40–64 character commit SHA:

```bash
reverse-reap validate-bridge-config configs/bridge-qwen35-2b.yaml
reverse-reap train-bridge /path/to/pinned-bridge-config.yaml
```

No command in this feature downloads models or provisions paid hardware.
