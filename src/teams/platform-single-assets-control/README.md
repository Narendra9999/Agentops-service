# Default-packaging control fixture

This isolated FEVM test team copies the existing platform-engineering-test agent
without implementation changes and omits `model.asset_mappings`. Resource names
are distinct so the shared pipeline can validate default packaging without
replacing an existing endpoint.

See `docs/development/non-split-asset-mapping-validation.md` for results.
