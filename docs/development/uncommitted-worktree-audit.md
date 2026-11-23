# Remaining worktree changes

Audit after restoring the non-split supervisor and module-entry loading.
The source worktree is on `feat/split-agentops-agent-config`. Its uncommitted
files have not been discarded, staged, or changed by the restoration.

## Independent fixes still not ported

- `src/agent_development/agent_evaluation/evaluation/custom_scorers.py`: forward
  the configured judge model and increase its response limit when selected.
- `src/agent_development/agent_evaluation/evaluation/scorer_loader.py`: resolve
  the helper before scorer threads, pass the judge model, and provide bounded
  native tool/retrieval evidence to numeric domain rubrics.
- `src/framework/evaluation/evaluation_pipeline.py`: render a missing metric as
  `missing` while continuing to reject the failed quality gate.
- `tests/integration/test_supervisor_evaluation_support.py`: regression tests
  for those fixes.

These fixes were part of the historical supervisor FEVM working tree, so that
run does not establish evaluation equivalence on the non-split branch.

## Documentation and tooling still not ported

- `docs/configuration/`: configuration manual, parameter indexes, examples,
  separation design and migration notes. Some content describes split configs
  and requires review before publishing on the non-split branch.
- `scripts/docs/generate_config_parameter_index.py`: index generator.
- `docs/diagrams/agentops-request-flows/`: printable PDF and generator.
- `examples/multi_agent_fe/scripts/fevm_remote_usage_smoke.py`: explicit-profile
  remote-worker usage validation script. Its presence is not a test result.

## Split experiment intentionally excluded

- `src/framework/configuration.py` and split-aware edits in `team_config.py`,
  `team_runtime.py`, `notebook_bootstrap.py`, and `observability/tracing.py`.
- Split-config edits in registration, pre/post evaluation, production monitoring
  and the serving wrapper. Registration and wrapper files also contain independent
  module-loading changes; those were ported selectively, not copied wholesale.
- `src/teams/platform-intelligence-split/`, `test_split_configuration.py`,
  `docs/development/split-configuration-validation.md`, and README changes.

## Restored on the non-split branch

- Original supervisor team, workers, rubric YAMLs, combined config and bundle.
- `target_loader.py`, registration entry detection, serving loader and tests.
- Supervisor uses `inference.target_module` without `Agent.py`; an existing
  `Agent.py` wins for other teams.

The old supervisor and loader files still appear untracked in the source worktree
because that branch was left untouched. Their presence there does not mean the
restored versions are absent from the non-split branch. Local integration validation
passed 176 tests, excluding the separate live-endpoint module. No new FEVM deployment
was performed during restoration or module-loading validation.
