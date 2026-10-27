# Chapter 1: Running federated learning applications

This development revision targets **NVFlare 2.10** and uses the **Recipe API throughout**. Start with the [introduction](01.0_introduction/introduction.ipynb), then [setup](01.1_running_federated_learning_job/setup.ipynb) and the [first run](01.1_running_federated_learning_job/running_pytorch_fl_job.ipynb).

The remaining chapters still describe the 2.6 course. Keep this revision on the course development branch until the new edition is ready as a whole. The [original 2.6 course](https://github.com/NVIDIA/NVFlare/tree/2.6.0/examples/tutorials/self-paced-training) remains available with its original code and videos.

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
python -m ipykernel install --user --name nvflare-chapter1 --display-name 'NVFlare Chapter 1'
python -m jupyterlab
```

Use **NVFlare Chapter 1** in Jupyter. Run training commands in a terminal with the same environment activated. Notebook cells inspect source and results; run each lesson's terminal commands before its result-inspection cells.

## One project for all lessons

All executable examples are in [code](code), shared across the notebooks:

| File | Lesson |
| --- | --- |
| `fl_job.py` | Construct `FedAvgRecipe`, add final evaluation, and execute with `SimEnv` |
| `src/client_origin.py` | Ordinary local PyTorch training |
| `src/client_v1.py` | Minimal training-only Client API conversion |
| `src/client.py` | Training, evaluation, and optional analytics |
| `custom_server_job.py` | Recipe stopping, model selection, and persistence settings |
| `custom_client_job.py` | `set_per_site_config()` for different client arguments |
| `src/training.py` | Shared local training and evaluation functions |
| `src/network.py`, `src/data.py` | Seeded CNN and site-specific synthetic images |

The synthetic data uses the same design as Hello PyTorch: 200 training and 100 evaluation samples per site, with different site/split seeds and a shared class distribution. It is an offline workflow exercise, not a benchmark. Local SGD state is reset every round. The client reports actual optimizer-step counts for aggregation and evaluates received weights before training; final evaluation measures the persisted checkpoint separately.

From `code`, run:

```bash
python -m src.client_origin
python fl_job.py
python fl_job.py --client-script src/client_v1.py
python custom_server_job.py
python custom_client_job.py
python fl_job.py --export --export-dir ./exported
python custom_client_job.py --export --export-dir ./exported
```

Recipe workspaces default to `/tmp/nvflare/simulation/<recipe-name>`. Set `NVFLARE_SIMULATOR_WORKSPACE_ROOT` to a fresh directory to preserve separate experiments, and give the notebook kernel the same setting. The default and minimal-client jobs both use `chapter1-none`, so use separate roots to retain both runs.

Optional tracking labs:

```bash
python -m pip install tensorboard mlflow
python fl_job.py --tracking tensorboard
python fl_job.py --tracking mlflow
```

The TensorBoard and MLflow notebooks explain how to open the local UIs in a separate terminal. Neither UI is started automatically by a notebook cell. MLflow uses `code/mlflow.db`; repeated runs remain in that store.

## Completion and remaining edition work

Complete the [recap exercise](01.8_recap/recap.ipynb): configure a site, inspect the exported recipe, run it, and explain the saved model's evaluation using its artifacts and logs.

The [existing videos](video.md) teach the original Job API edition. They require new recordings to match this Recipe API revision. The new course edition also needs the remaining chapters, full-book validation, and learner review before publication; this chapter does not mark the whole course as updated.
