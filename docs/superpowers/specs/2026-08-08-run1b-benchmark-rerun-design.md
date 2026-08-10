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

The sweep implementation is model-count agnostic; the current prelaunch
manifest contains four ordered model specs. The locally available
Qwen3-1.7B/Qwen3-8B pair remains a useful no-credential subset, but neither is
called a frontier model. Thinking Machines Inkling is out of scope for this
run because no compatible local artifact or approved endpoint is available.

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
- Every named profile sends `chat_template_kwargs={"enable_thinking": false}`
  natively to every declared model endpoint. Generic sweeps may omit this
  field, but a Run 1B engineering smoke, benchmark smoke, or compliance run
  may not. The H100 tool-path smoke showed that thinking can exhaust the
  512-token tool-only budget before a native call, so omission is not
  equivalent to the frozen contract.
- vLLM is launched with its framework defaults instead of mutable model
  `generation_config.json` overrides; engine and parser settings are recorded.
- Scripted anchors and learned models use the same prompt/repeat support.
- The engineering and benchmark smokes require a request-capture path. For
  these named Run 1B profiles, exact unauthenticated loopback endpoints and
  the frozen synthetic inputs make the captured URL and JSON payload
  credential-free by contract. Full compliance forbids request capture so the
  512,000-request lattice cannot become an in-memory launch hazard; its receipt
  binds the passed prelaunch captures instead.

## Gates

The launch sequence is fail-closed:

1. Local CPU tests pass on the exact source to transfer.
2. Model servers report the expected model IDs and revisions.
3. A one-prompt/two-response tool-call smoke completes for every declared
   model with no infrastructure, parse, or invalid-call failures. Its request
   capture must contain the complete expected lattice.
4. Request-capture validation confirms identical sampling, seed derivation,
   and explicit `enable_thinking=false` chat-template kwargs across models.
5. A five-prompt/two-response benchmark smoke passes every scripted anchor
   constraint and every model engineering gate, again with a complete request
   capture.
6. Only then may the 500 by 16 profile run.

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

## Request-capture contract

`--request-capture-out` is mandatory for `--engineering-smoke` and
`--benchmark-smoke`. It remains optional for generic exploratory sweeps and is
rejected with `--compliance-profile`. The destination's parent receipt
directory must already exist. The resolved capture and report paths must be
different. A same-directory `<destination>.partial` file is opened with
exclusive creation before model traffic, and the final destination must not
already exist. Either existing path fails the run instead of being overwritten.

After the complete lattice validates, the implementation writes, flushes,
fsyncs, and closes the partial before atomically publishing it through a
same-directory hard link that cannot replace a racing final path. It then
removes the partial name. On any `BaseException` before successful publication,
the implementation publishes no new final, closes and retains the clearly
named partial as incomplete evidence, and leaves any pre-existing or racing
final untouched.

The capture records every payload actually constructed for a model request,
not one representative payload per model. With the current four-model
manifest and the fixed 16-step horizon, the exact cardinalities are:

- engineering: `4 models x 1 prompt x 2 responses x 16 steps = 128` rows;
- benchmark: `4 models x 5 prompts x 2 responses x 16 steps = 640` rows.

Rows are serialized in deterministic model-spec, prompt, response, then step
order regardless of asynchronous completion order. Each row contains schema
version, model label/served ID/capability rank, all three integer coordinates,
the request URL, the exact constructed JSON payload, and the SHA-256 of that
payload's canonical JSON (`sort_keys=true`, compact separators, UTF-8,
non-finite numbers rejected). HTTP headers are never captured. Generic
exploratory capture does not promise that caller-supplied URLs or payloads are
secret-free, so callers must not place secrets there. Every named Run 1B
profile is stricter: every model must have no `api_key_env` and must use exactly
`http://127.0.0.1:<port>/v1`, with a decimal port from 1 through 65535. This
excludes URL user information, query strings, fragments, non-loopback hosts,
and TLS endpoints from the named profile. A duplicate, out-of-range, missing,
or unexpected coordinate fails the run; validation happens before a complete
capture is finalized.

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
start/end UTC. Under the named Run 1B endpoint and frozen-input contract, no
credential is written to the package.

The strict reporter package above stays unchanged. The final custody handoff
wraps it with the raw full receipt and both prelaunch receipts:

```text
run1b_qwen4_handoff/
  prelaunch/
    engineering_smoke/       # includes request_capture.jsonl + SHA256SUMS
    benchmark_smoke/         # includes request_capture.jsonl + SHA256SUMS
  full_profile/              # raw compliance receipt; no request capture
  validation/                # verifier output and validator digest
  package_inputs/metadata.json
  package/                   # strict reporter output shown above
  HANDOFF_MANIFEST.json
  OUTER_SHA256SUMS
```

A full-profile request capture is forbidden rather than a launch gate. The
full verifier requires both passed prelaunch capture receipts and proves that
they share the same source, model configuration, serving image, prompt/tool,
sampling, and environment identities before the outer archive is sealed.

The graph consumes only validated report data. Failed models are visibly
marked `NOT EVALUABLE`, never converted to score zero. Historical T1, Run B,
and RunB2 values are not mixed into the same quantitative panel because they
use different manifests and estimands.

## Container boundary

The R46 digest-pinned container is the runtime design for RunB2 training. Its
formal copied-venv predecessor becomes an optional audit artifact, not a launch
blocker. The Run 1B evaluation may use separately digest-pinned model-serving
containers, but it must not imply that resource-native SFT or GRPO ran.
