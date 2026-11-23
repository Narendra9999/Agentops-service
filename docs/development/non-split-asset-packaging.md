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

If a developer moves a source file while retaining an expression that no longer
locates its asset in the source tree, the source layout is already broken before
AgentOPS packages it. AgentOPS cannot infer the intended asset or safely rewrite
arbitrary file access. Keep the working customer tree intact when integrating it.
Absolute machine paths and assets outside the packaged directories are outside this
contract. Support for additional asset roots can be considered separately.

## Validation

Real MLflow save/load tests package unchanged customer source, check staged source
bytes, delete the original team checkout, and invoke the loaded model. They cover
both sibling configs and nested package assets. A third test checks optional assets
and preservation of the legacy subpackage aliases. These are local packaging tests,
not a new FEVM deployment. The tests use the real AgentOPS target resolver with a
small MLflow loader to isolate the packaging behavior.

Local result: all 143 integration tests on this branch passed, including the three
new real-packaging cases. The live-serving test module was excluded. Registration
notebook Python syntax and CRLF-aware whitespace checks passed. No deployment or
customer agent source changes were made.
