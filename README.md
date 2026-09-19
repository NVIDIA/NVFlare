<img src="https://raw.githubusercontent.com/NVIDIA/NVFlare/main/docs/resources/nvidia_eye.wwPt122j.png" alt="NVIDIA Logo" width="200">

# NVIDIA FLARE

Federate familiar training code. Validate it locally, then deploy the same workload securely across sites.

[Documentation](https://nvflare.readthedocs.io/en/main/) |
[Quick Start](https://nvflare.readthedocs.io/en/main/quickstart.html) |
[Examples](https://nvidia.github.io/NVFlare/catalog/) |
[Discussions](https://github.com/NVIDIA/NVFlare/discussions)

[![Blossom-CI](https://github.com/NVIDIA/nvflare/workflows/Blossom-CI/badge.svg?branch=main)](https://github.com/NVIDIA/nvflare/actions)
[![documentation](https://readthedocs.org/projects/nvflare/badge/?version=main)](https://nvflare.readthedocs.io/en/main/?badge=main)
[![pypi](https://badge.fury.io/py/nvflare.svg)](https://badge.fury.io/py/nvflare)
[![license](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](./LICENSE)

NVIDIA FLARE is an open-source SDK for federated learning and federated compute. It lets data scientists adapt
PyTorch, TensorFlow, scikit-learn, XGBoost, and other Python workflows while each participating site keeps control of
its data and local execution. The same application can move from laptop simulation to a provisioned multi-site
deployment.

## Run a meaningful two-client federation

Install NVFLARE with its PyTorch dependencies, retrieve the example that matches the installed package revision, and
run it with focused progress output:

```bash
python -m pip install "nvflare[PT]"
nvflare examples get hello-pt
cd hello-pt
python job.py --log_config progress
```

The example needs no dataset download or GPU. Two simulated clients train independently on deterministic,
site-specific synthetic images for three federated rounds. The persisted final global model is then evaluated on
separate client evaluation data.

A validated run produced this result; elapsed time varies by machine, and these values demonstrate the example
rather than benchmark NVFLARE:

```text
============================= RUN SUMMARY ==============================

  NVIDIA FLARE · hello-pt
  Simulation · 2 clients

  ✓ Completed                                                   30.6s

  Training · aggregated client metrics

  Round        accuracy  accuracy_after_local_training
  1                   1                             20
  2                  30                             55
  3                  70                             65

  Model evaluation · accuracy

  Client  SRV_FL_global_model.pt  SRV_best_FL_global_model.pt
  site-1                      75                           70
  site-2                      77                           70

  Models    server/simulate_job/app_server/
  Metrics   server/simulate_job/metrics/
  Evaluation server/simulate_job/cross_site_val/cross_val_results.json
  Logs      server/log.txt · site-1/log.txt · site-2/log.txt
  Results   /tmp/nvflare/simulation/hello-pt
```

See the [Hello PyTorch guide](./examples/hello-world/hello-pt/README.md) for the complete output, deterministic data
design, artifact semantics, command options, CIFAR-10 continuation, and troubleshooting.

## Understand what happened

1. Two client tasks trained the same model on distinct local datasets.
2. Raw samples stayed in the client processes; clients returned model parameters, metrics, and aggregation weights.
3. FedAvg combined the client updates and persisted a new global model after each round.
4. Both clients evaluated the final persisted global model on data excluded from their training partitions.

The simulator runs the federation locally, but it exercises the same Recipe and client application structure used by
the POC and production environments.

## Adapt existing training code

Most of the example client remains ordinary PyTorch. The federated integration surrounds its existing model,
training, and evaluation code with the Client API exchange loop. This is a shortened excerpt; review
[`client.py`](./examples/hello-world/hello-pt/client.py) for the complete runnable implementation, including metrics
and aggregation-weight metadata.

```diff
+ import nvflare.client as flare

+ flare.init()
+ while flare.is_running():
+     input_model = flare.receive()
+     model.load_state_dict(input_model.params)

      # Existing local evaluation and training loop

+     last_params = {
+         name: param.detach().cpu().clone()
+         for name, param in model.state_dict().items()
+     }
+     output_model = flare.FLModel(
+         params=last_params,
+         metrics={"accuracy": accuracy_before_training},
+         meta={"NUM_STEPS_CURRENT_ROUND": steps},
+     )
+     flare.send(output_model)
```

For a guided manual conversion, start with the
[Client API guide](https://nvflare.readthedocs.io/en/main/user_guide/data_scientist_guide/client_api_usage.html) and
[Job Recipe guide](https://nvflare.readthedocs.io/en/main/user_guide/data_scientist_guide/job_recipe.html).

You can also use the maintained Agent Skills workflow from an existing PyTorch project in Codex or Claude Code:

```text
I have an existing PyTorch training project in ./source. Convert it to
federated learning using FedAvg and validate it locally with 2 clients and 2
rounds of training. You may download any required public model artifacts,
including tokenizer and configuration files, if they are not already cached.
Proceed without asking for additional confirmation.
```

The validated workflow produces reviewable Client API and Recipe code, runs the generated two-client simulation,
and reports its metrics and artifacts. You remain responsible for reviewing the generated code and its data and
model assumptions. See the [Agent Skills guide](https://nvflare.readthedocs.io/en/main/user_guide/agent_skills/index.html)
for installation, supported conversion workflows, validation, and limitations.

## Move from simulation to a real federation

The application remains the workload as its execution environment changes:

| Stage | Execution environment | What changes |
|---|---|---|
| Local validation | `SimEnv` | Server and clients run locally for fast iteration. |
| Process-level proof | `PocEnv` | Separate local services exercise deployment-like boundaries. |
| Deployed federation | `ProdEnv` | Provisioned identities, startup kits, networks, policies, and real sites participate. |

Continue with the tested
[Hello PyTorch environment guide](./examples/advanced/hello-pt-environments/README.md) to run the same model, data,
client code, and Recipe in POC or connect it to an already-running production system. Provisioning, identity,
authorization, networking, and site operations are covered by the
[deployment guide](https://nvflare.readthedocs.io/en/main/user_guide/admin_guide/deployment/overview.html).

## Choose your next path

After the first successful run:

1. **Adapt this example:** replace the model and data ownership points in
   [`client.py`](./examples/hello-world/hello-pt/client.py), then validate the changed job locally.
2. **Adapt existing code:** [choose an API](https://nvflare.readthedocs.io/en/main/user_guide/data_scientist_guide/api_selection.html),
   then use the manual Client API route or the optional Agent Skills workflow.
3. **Move to a real federation:** follow the
   [environment guide](./examples/advanced/hello-pt-environments/README.md), then continue to provisioned deployment.

Other maintained paths include the [example catalog](https://nvidia.github.io/NVFlare/catalog/),
[Collaboration API](https://nvflare.readthedocs.io/en/main/user_guide/data_scientist_guide/collab_api.html),
[federated LLM guide](https://nvflare.readthedocs.io/en/main/programming_guide/llm_fine_tuning.html),
[research implementations](./research/README.md), and
[security overview](https://nvflare.readthedocs.io/en/main/system_architecture/security_overview.html).

Project and community resources:

- Read [What's New](https://nvflare.readthedocs.io/en/main/whats_new.html) and the
  [talks and publications](https://nvflare.readthedocs.io/en/main/publications_and_talks.html).
- Ask questions and share ideas in [GitHub Discussions](https://github.com/NVIDIA/NVFlare/discussions).
- Review the [contributing guide](./CONTRIBUTING.md) and open
  [good first issues](https://github.com/NVIDIA/NVFlare/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
- Cite the [NVIDIA FLARE paper](https://arxiv.org/abs/2210.13291) when the project supports your work.

NVIDIA FLARE is released under the [Apache 2.0 license](./LICENSE).
