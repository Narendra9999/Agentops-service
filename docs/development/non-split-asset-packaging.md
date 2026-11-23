# Non-split agent packaging without customer code changes

Branch: `feat/platform-intelligence-supervisor`.
The split experiment remains untouched in its original working directory. This
change uses the combined team `config.yaml` and existing registration pipeline.

## Contract

AgentOPS preserves the customer's existing `agent/` subtree and packages the
conventional sibling `configs/` directory alongside it. Existing subpackage import
aliases and configuration-file staging are retained. A single shared helper now
constructs these paths for both registration paths, replacing duplicated notebook
logic. There is no new asset API, manifest format, runtime path monkeypatch or
requirement for teams to read `context.artifacts`.

For example:

```text
Source team/                      MLflow code/
  agent/                            agent/
    Agent.py                          Agent.py
    workers/                          workers/
      supervisor.py                     supervisor.py
      assets/prompts.json               assets/prompts.json
  configs/prompts.json               configs/prompts.json
  config.yaml                        config.yaml
```

A supervisor directly in `agent/` can continue reading
`Path(__file__).parent.parent / 'configs/prompts.json'` unchanged. A nested package
can continue reading its own `Path(__file__).parent / 'assets/prompts.json'`.
The same files remain in the same relative positions after model staging.

## Explicit mappings for nested implementations

When nested code still reads `Path(__file__).parent.parent / "configs/prompts.json"`,
its expected location is `agent/configs`, rather than the conventional sibling
`configs/`. Supply an explicit registration mapping in the existing combined
team `config.yaml`:

```yaml
model:
  asset_mappings:
    - source: configs
      destination: agent/configs
```

AgentOPS copies the existing code paths into a private temporary staging tree,
then copies `team/configs` to `stage/agent/configs`. MLflow packages that staged
copy. The original team files and the supervisor's prompt-reading code remain
unchanged. Both registration branches use the same resolver. The original
`code/configs` directory and import aliases are retained; mapped asset directories
do not introduce additional aliases.

Sources are relative to the team directory and must exist. Destinations must be
new paths below `agent/`. File and directory sources are supported. Absolute
paths, `..`, escaping symlinks, malformed mappings, and overwriting any existing
destination are rejected. Mappings run in order; conflicting destinations fail
registration. Mapping a directory copies the entire directory, so teams should
keep credentials outside packaged asset directories.

Omitting `asset_mappings`, or setting it to `[]`, returns precisely the existing
code paths without staging. AgentOPS cannot infer arbitrary file access paths;
a mapping must describe the location the unchanged implementation expects.
Absolute machine paths and assets outside the team directory are unsupported.
Mappings apply to team-agent registration, not the shared fallback agent.

## Validation

See [FEVM asset-mapping validation](non-split-asset-mapping-validation.md) for
local regression results and the isolated FEVM supervisor and single-agent runs.
The original split experiment and existing FEVM endpoints are not modified.
