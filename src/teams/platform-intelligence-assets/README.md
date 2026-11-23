# Nested prompt-packaging validation fixture

This isolated FEVM test team preserves the clean supervisor and worker source
contents while placing the supervisor below `agent/supervisors/`. The supervisor
still reads `../configs/prompts.json` relative to its implementation directory.
`model.asset_mappings` supplies that path during registration without changing
its code. The original sibling configs remain packaged as before.

This team calls existing platform and Genie workers. It does not deploy workers.
Use the normal combined config and shared pipeline in `deployment/databricks.yml`.
See `docs/development/non-split-asset-mapping-validation.md` for evidence and limits.

Worker connectors live in `agent/workers/platform_engineering.py` and
`agent/workers/cost_intelligence.py`. The package `__init__.py` only exports
`call_platform` and `call_genie`; `_responses.py` shares response text extraction.

Domain evaluation definitions are in `scorers/domain/cost_grounding.yaml` and
`scorers/domain/platform_evidence.yaml`. These use the framework's existing
Guidelines scorer type (pass/fail rules); `config.yaml` controls enablement
and thresholds. They are not numeric rubric scorers. Built-in correctness,
relevance and safety remain configured under `evaluation.builtin_scorers`.
