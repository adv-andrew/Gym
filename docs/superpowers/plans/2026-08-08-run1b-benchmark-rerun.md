# Run 1B multi-model benchmark rerun implementation plan

## Task 1: Correct the statistical unit in the sweep

Files:

- Modify `resources_servers/openair_congestion/model_sweep.py`
- Modify `resources_servers/openair_congestion/tests/test_model_sweep.py`

Use TDD to prove that repeated responses for one prompt collapse to one
cluster and that stratified prompt-cluster resampling, not response-level
resampling, drives the interval. Preserve strict pair-support and failure
checks. Record bootstrap method, seed, draws, prompt clusters, response pairs,
and prompt-level win/tie/loss in every comparison.

## Task 2: Pin the model request contract

Files:

- Modify `resources_servers/openair_congestion/model_sweep.py`
- Modify `resources_servers/openair_congestion/sweep_models.example.json`
- Modify `resources_servers/openair_congestion/tests/test_model_sweep.py`

Add explicit `top_p` and a deterministic request seed derived from prompt,
response, and step. Validate finite/range-safe sampling parameters. Confirm the
mock endpoint receives the exact same sampling contract for every model.

## Task 3: Build deterministic reporting and custody packaging

Files:

- Create `resources_servers/openair_congestion/model_sweep_report.py`
- Create `resources_servers/openair_congestion/tests/test_model_sweep_report.py`

Under tests, validate a raw report, flatten complete episode records, write
summary JSON/CSV and prompt-cluster paired-delta CSV, render deterministic PNG
and SVG graphs, record supplied environment metadata, and write sorted
SHA-256 custody entries. Reject missing, duplicate, non-finite, or inconsistent
records. Mark failed learned models `NOT_EVALUABLE` in summaries and graphs.

## Task 4: Document the executable contract

Files:

- Modify `resources_servers/openair_congestion/README.md`

Document the Run 1B naming caveat, smoke/full commands, fixed sampling and
clustered inference, receipt schema, model-server requirements, and the rule
that the benchmark is evaluation rather than GRPO training.

## Task 5: Local verification and independent review

Run the focused tests first, then all OpenAir resource-server tests from the
isolated worktree with its own source paths. Run Ruff or the repository's
equivalent lint on changed Python. Have an independent read-only reviewer audit
the diff against the design and fix every P0/P1 finding.

## Task 6: Transfer and remote engineering gates

The parent agent alone will push/transfer exact reviewed commits and perform
all LaunchPad actions serially. Capture the new LaunchPad's runtime, GPU, disk,
container, model-cache, and source identities. Start the two model servers
with pinned launch settings; verify model IDs and revisions; run the one-prompt
tool smoke and then the five-prompt benchmark smoke.

## Task 7: Full profile and handoff

Only if all smoke gates pass, run the 500 by 16 profile. Generate the receipt
package and graph, verify every hash, copy the package locally, independently
recompute its statistics, and write an evidence-bounded verdict. A failed gate
is a result to diagnose, not permission to weaken the contract or silently
drop rows.
