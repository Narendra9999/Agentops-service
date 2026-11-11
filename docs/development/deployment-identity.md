# Deployment identities: Jenkins and FEVM

The customer root `databricks.yml` uses `run_as.service_principal_name: !sp_id`.
Jenkins renders `!sp_id` from `dab-config.json`, along with the workspace host.
The authenticated Jenkins caller deploys the bundle; the configured principal
executes its jobs. Changing `run_as` does not authenticate Model Serving as that
principal or inject the Jenkins caller's credentials into model containers.

The FEVM team bundles mirror this pattern with
`run_as.service_principal_name: ${var.service_principal_id}`. Each FEVM target
binds the existing `fe-agentops-multi-runtime` application ID and the workspace
host. Override the variable when deploying to another target. The workspace host
is a target literal because the CLI resolves authentication before ordinary
variable interpolation; customer Jenkins already renders its host before CLI use.
The original six notebook stages remain unchanged.

## Required grants

The bundle deployer must have **Service Principal User** (`servicePrincipal.user`)
on the selected principal. The principal also needs source-file access and the
resource permissions required by registration, evaluation and deployment:
configured UC schemas/models/volumes/tables, experiments, secret references,
Genie and its downstream resources, documentation index, model services, serving
endpoints and Lakebase where used. Existing inference-time grants alone do not
prove the principal can execute the complete pipeline.

## FEVM attempt

Both bundles passed `bundle validate --strict -t fevm --profile fevm-naren`.
Deployment of both jobs was rejected with HTTP 403: the current deployer lacks
`servicePrincipal.user` on `fe-agentops-multi-runtime`
(application ID `e3bcf75e-7b34-4026-8860-d5e4423ea2b7`). Source files were uploaded,
but the job updates failed. No pipeline run or serving update was started for
this identity change. An administrator must grant that role to the deployer
before deployment can be retried. Full pipeline execution as this principal is
not yet validated.

## Runtime authentication remains separate

Do not remove serving OAuth settings solely because a job uses `run_as`.
Databricks documents that Model Serving automatic authentication passthrough
**does not support Unity Gateway model services**. The supervisor currently uses
`system.ai.*` through that route, so its explicit runtime credentials remain.
Its Lakebase Autoscaling path must also be validated before changing identity.
The single-agent runtime configuration is retained pending a separate managed-auth
validation. No signing-key, memory or App configuration changes are included.

Reference: [Model Serving agent authentication](https://docs.databricks.com/aws/en/agents/custom-agents/model-serving/agent-authentication-model-serving).

After the administrator grant, deploy each team bundle and run its existing
`agentops_pipeline` using the explicitly selected `fevm-naren` profile. Validate
both pipelines and downstream resource access as the new job identity before
calling the migration complete.
