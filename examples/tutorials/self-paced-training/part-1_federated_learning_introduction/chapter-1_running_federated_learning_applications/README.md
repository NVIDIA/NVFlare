# Running Federated Learning Applications

This development revision targets **NVFlare 2.10** and uses the **Recipe API throughout**. Start with the [introduction](01.0_introduction/introduction.ipynb), then [setup](01.1_running_federated_learning_job/setup.ipynb) and the [first run](01.1_running_federated_learning_job/running_pytorch_fl_job.ipynb).

This chapter has been validated against the NVFlare 2.10 development branch and is ready for independent review. Other chapters are being migrated and reviewed separately. The [archived 2.6 course](https://github.com/NVIDIA/NVFlare/tree/2.6.0/examples/tutorials/self-paced-training) preserves the original notebooks and videos.

## Environment: development and published editions

Use Python 3.10–3.14 on Linux or macOS. The exercises run on CPU with two simulated clients; they do not require a GPU or dataset download. Installation requires internet access unless the dependencies are already cached.

Obtain the repository version containing these notebooks with Git **or extract its source ZIP**. Keep the chapter's notebooks and `code` directory together. A Git clone is not required. In a terminal at the repository root:

```bash
python3 -m venv .venv-chapter1
source .venv-chapter1/bin/activate
python -m pip install --upgrade pip
```

Choose the FLARE installation matching your notebook edition:

```bash
# Development / before publication: install this checkout or extracted source tree.
NVFL_BASE_VERSION=2.10.0 python -m pip install -e '.[pt]'
```

`NVFL_BASE_VERSION` supplies package metadata when a development ZIP has no Git version tag. It does not select or download a code version; use the source archive containing this chapter.

```bash
# Published 2.10 edition: use matching release notebooks and the PyPI package.
python -m pip install 'nvflare[pt]>=2.10.0,<2.11'
```

Then install the notebook tools and select the matching kernel:

```bash
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user --name nvflare-chapter1 --display-name 'NVFlare Self-Paced Training'
python -m jupyterlab
```

Use **NVFlare Self-Paced Training** in Jupyter. Run training commands in a terminal with the same environment activated. Notebook cells inspect source and results; run each lesson's terminal commands before its result-inspection cells.

## One project for all lessons

All executable examples are in [code](code), shared across the notebooks:

```text
chapter-1_running_federated_learning_applications/
├── code/                         # Shared runnable project
│   ├── job.py
│   ├── client.py
│   ├── model.py
│   └── ...
├── 01.1_running_federated_learning_job/
│   ├── setup.ipynb
│   ├── running_pytorch_fl_job.ipynb
│   └── code/README.md             # Links to the shared project
└── ...
```

Open `code` directly under the chapter directory. The former lesson-specific `code/src` directories may remain locally because of ignored Jupyter checkpoints.

| File | Lesson |
| --- | --- |
| `job.py` | Construct `FedAvgRecipe`, add final evaluation, and execute with `SimEnv` |
| `train.py` | Ordinary local PyTorch training |
| `client_train_only.py` | Minimal training-only Client API conversion |
| `client.py` | Training, evaluation, and optional analytics |
| `job_custom_server.py` | Recipe stopping, model selection, and persistence settings |
| `job_custom_clients.py` | `set_per_site_config()` for different client arguments |
| `training.py` | Shared local training and evaluation functions |
| `model.py`, `data.py` | Seeded CNN and site-specific synthetic images |

The synthetic data uses the same design as Hello PyTorch: 200 training and 100 evaluation samples per site, with different site/split seeds and a shared class distribution. It is an offline workflow exercise, not a benchmark. Local SGD state is reset every round. The client reports actual optimizer-step counts for aggregation and evaluates received weights before training; final evaluation measures the persisted checkpoint separately.

From `code`, run:

```bash
python train.py
python job.py
python job.py --client-script client_train_only.py
python job_custom_server.py
python job_custom_clients.py
python job.py --export --export-dir ./exported
python job_custom_clients.py --export --export-dir ./exported
```

Recipe workspaces default to `/tmp/nvflare/simulation/<recipe-name>`. Set `NVFLARE_SIMULATOR_WORKSPACE_ROOT` to a fresh directory to preserve separate experiments, and give the notebook kernel the same setting. The default and minimal-client jobs both use `chapter1-none`, so use separate roots to retain both runs.

Optional tracking labs:

```bash
python -m pip install tensorboard mlflow
python job.py --tracking tensorboard
python job.py --tracking mlflow
```

The TensorBoard and MLflow notebooks explain how to open the local UIs in a separate terminal. Neither UI is started automatically by a notebook cell. MLflow uses `code/mlflow.db`; repeated runs remain in that store.

## Completion and remaining edition work

Complete the [recap exercise](01.8_recap/recap.ipynb): configure a site, inspect the exported recipe, run it, and explain the saved model's evaluation using its artifacts and logs.

The [existing videos](video.md) teach the original Job API edition. They require new recordings to match this Recipe API revision. Updates to other chapters, full-course validation, and learner review proceed separately and do not block review of this chapter.
