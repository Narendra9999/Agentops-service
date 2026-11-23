# Nested prompt-packaging validation fixture

This isolated FEVM test team preserves the clean supervisor and worker source
contents while placing the supervisor below `agent/supervisors/`. The supervisor
still reads `../configs/prompts.json` relative to its implementation directory.
`model.asset_mappings` supplies that path during registration without changing
its code. The original sibling configs remain packaged as before.

This team calls existing platform and Genie workers. It does not deploy workers.
Use the normal combined config and shared pipeline in `deployment/databricks.yml`.
See `docs/development/non-split-asset-mapping-validation.md` for evidence and limits.
