# Run 1B multi-model benchmark rerun design

## Status and name

`Run 1B` is user shorthand for the strict real-model capability/compliance
sweep. No preserved artifact uses that literal run name. It is an evaluation
run, not the historical Run B GRPO run and not the RunB2 SFT/GRPO lineage.

The historical July 24 sweep is evidence only. It ran on older source and is
not allowed to qualify the current contribution branch.

## Objective

Rerun the causal resource-server benchmark on the current contribution source
with at least two locally served learned models under the same prompt, tool,
sampling, and environment contracts. Produce a graph and a complete receipt
package without ranking a model that fails the engineering gate.

The immediately available, no-credential ladder is Qwen3-1.7B and Qwen3-8B.
Neither is called a frontier model. Thinking Machines Inkling is out of scope
for this run because no compatible local artifact or approved endpoint is
available.

## Frozen experiment contract

- Source starts from contribution commit
  `a3897fc9382a5ec67c3f077f4fc19a9af0bd7ae6` plus the reviewed benchmark
  hardening commit produced by this work.
- Backend is causal deterministic `replay`, never `dataset_replay`.
- Reward, dynamics, renderer, task rows, and candidate tools are fixed by the
  checked-in current contribution and recorded by SHA-256.
- Five one-hot regimes are covered.
- The full profile is 500 deterministic prompts by 16 responses per prompt,
  or 8,000 episodes per policy.
- Models receive identical messages, tool schemas, horizon, temperature,
  top-p, maximum output tokens, required single-tool-call policy, and
  pair-derived request seeds.
- vLLM is launched with its framework defaults instead of mutable model
  `generation_config.json` overrides; engine and parser settings are recorded.
- Scripted anchors and learned models use the same prompt/repeat support.

## Gates

The launch sequence is fail-closed:

1. Local CPU tests pass on the exact source to transfer.
2. Model servers report the expected model IDs and revisions.
3. A one-prompt/two-response tool-call smoke completes for both models with no
   infrastructure, parse, or invalid-call failures.
4. A five-prompt/two-response benchmark smoke passes every scripted anchor
   constraint and both model engineering gates.
5. Only then may the 500 by 16 profile run.

The full engineering gate requires the complete planned pair support for every
model and zero infrastructure errors, parse failures, and invalid calls. A
model that misses this gate is `NOT_EVALUABLE`; its raw operational mean may be
shown diagnostically but it cannot be ranked as model quality.

The environment gate requires a positive mean and positive 95% lower bound for
each declared anchor comparison: relief over noop, relief over random-valid,
noop over catastrophic, and random-valid over catastrophic.

The model-quality gate applies only after the engineering gate. Every adjacent
higher-ranked model must have a positive paired prompt-cluster mean delta and a
positive 95% lower bound.

## Statistical unit

The sixteen responses for one prompt share the same state and therefore are
not independent experimental units. Inference uses prompt clusters:

1. Pair responses by `(prompt_index, response_index)`.
2. Compute stronger-minus-weaker deltas for every paired response.
3. Average those deltas within each prompt.
4. Resample prompt clusters within each regime with a fixed seed.
5. Use 50,000 bootstrap draws for the compliance profile.

Report cluster count, response-pair count, mean/median delta, 95% interval, and
prompt-level win/tie/loss. The same clustered method applies to anchors.

## Artifacts

The package is immutable after completion and contains:

```text
run1b_<UTC>_<shortcommit>/
  benchmark_contract.json
  models.json
  raw_report.json
  episodes.jsonl
  summary.json
  summary.csv
  paired_deltas.csv
  benchmark.png
  benchmark.svg
  environment.json
  run.log
  SHA256SUMS
```

`environment.json` records source and dirty status, exact commands and exit
codes, host/GPU identity, model and tokenizer revisions or hashes, container or
runtime identities, decoding settings, seeds, reward/dynamics identity, and
start/end UTC. Credentials are never written.

The graph consumes only validated report data. Failed models are visibly
marked `NOT EVALUABLE`, never converted to score zero. Historical T1, Run B,
and RunB2 values are not mixed into the same quantitative panel because they
use different manifests and estimands.

## Container boundary

The R46 digest-pinned container is the runtime design for RunB2 training. Its
formal copied-venv predecessor becomes an optional audit artifact, not a launch
blocker. The Run 1B evaluation may use separately digest-pinned model-serving
containers, but it must not imply that resource-native SFT or GRPO ran.
