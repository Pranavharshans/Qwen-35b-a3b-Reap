# Targeted donor bridge-capture hand-off

This is an opt-in follow-on to the Qwen3.5 v0 run. It does not alter the
existing REAP telemetry or causal plans. The four selected layer-expert pairs
remain labelled `observational-candidates`; Gate C is not causal proof.

## Data flow

1. Freeze `capture-manifest.json` from an already approved normalized dataset.
   The manifest embeds the normalized samples, exact teacher-forced token
   counts, source-manifest hash, tokenizer fingerprint, donor revision, and
   the immutable 200,000-token target / 500,000-token ceiling.
2. Run `capture-targets` on the donor BF16 checkpoint. The targeted hook
   replays only the four selected experts on a detached side path and always
   returns the native fused output. It records expert input, output, and
   router-weighted output vectors in atomic BF16 safetensors shards. Every
   completed batch is finalized before `capture-state.json` is atomically
   replaced, so an interrupted process can validate existing shards and resume
   at the next sample without overwriting them.
3. Validate every shard and build `handoff-manifest.json`. Copy this manifest,
   all shards, and the lossless extraction directory to durable storage before
   destroying the donor VM.

## Coverage rule

Capture stops when every selected expert has at least 5,000 coding routed
events and 1,000 control routed events. If that does not happen, it stops at
500,000 analysed tokens and reports incomplete coverage. The ceiling is never
raised after observing routing results.

## Commands

```bash
reverse-reap freeze-bridge-manifest \
  datasets/manifests/full-lengthmatched.jsonl /models/qwen \
  runs/bridge-capture/<RUN_ID>/capture-manifest.json \
  --config configs/pinned-pro6000-bf16-bridge-capture.yaml \
  --candidate-manifest runs/bridge-capture/inputs/candidate-manifest.json \
  --model-revision 59d61f3ce65a6d9863b86d2e96597125219dc754 \
  --run-id <RUN_ID> \
  --target-tokens 200000 --hard-token-ceiling 500000 \
  --max-input-tokens 1024 \
  --report runs/bridge-capture/<RUN_ID>/manifest-freeze-report.json

reverse-reap capture-targets \
  configs/pinned-pro6000-bf16-bridge-capture.yaml /models/qwen \
  runs/bridge-capture/<RUN_ID>/capture-manifest.json \
  runs/bridge-capture/<RUN_ID>/targets \
  --batch-size 4 --shard-max-records 4096 --run-id <RUN_ID> \
  --report runs/bridge-capture/<RUN_ID>/capture-report.json

# Resume with the same command, run ID, manifest, and batch settings after an
# interruption. Existing completed shards are validated before continuation.

reverse-reap extract \
  configs/pinned-pro6000-bf16-bridge-capture.yaml /models/qwen \
  runs/bridge-capture/inputs/candidate-manifest.json \
  runs/bridge-capture/<RUN_ID>/extraction

reverse-reap build-target-handoff \
  runs/bridge-capture/<RUN_ID>/targets \
  runs/bridge-capture/<RUN_ID>/capture-manifest.json \
  runs/bridge-capture/<RUN_ID>/handoff-manifest.json \
  --extraction-dir runs/bridge-capture/<RUN_ID>/extraction

reverse-reap validate-target-handoff \
  runs/bridge-capture/<RUN_ID>/handoff-manifest.json
```

The donor checkpoint is not copied into the hand-off. Its immutable revision
and source hashes are recorded. Bridge training needs a separate host model and
is not launched by this plan.

The execution plan deliberately keeps the bulk capture batch at the currently
pinned B4 default. Before a paid launch, the lead must review
`batch-probe.json` and amend the config and plan together if a different batch
is selected. This avoids silently consuming a runtime-generated batch choice.
