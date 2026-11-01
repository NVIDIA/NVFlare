# Running Federated Learning Applications

These lessons use the **Recipe API** and require **NVFlare 2.10**. Start with the [introduction](01.0_introduction/introduction.ipynb), then [setup](01.1_running_federated_learning_job/setup.ipynb) and the [first run](01.1_running_federated_learning_job/running_pytorch_fl_job.ipynb).

> **Video notice:** All [linked videos](video.md) show the earlier Job API workflow and do not match these Recipe API notebooks. Follow the notebooks for setup, commands, and code; see the video page for the matching archived course.

Other chapters are being migrated separately.

## Environment: development and published editions

Use Python 3.10–3.14 on Linux or macOS. The exercises run on CPU with two simulated clients, or three in the server aggregation lesson; they do not require a GPU or dataset download. Installation requires internet access unless the dependencies are already cached.

Obtain the repository version containing these notebooks with Git **or extract its source ZIP**. Keep each section's notebook and its own `code` directory together. A Git clone is not required. In a terminal at the repository root:

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

Use **NVFlare Self-Paced Training** in Jupyter and run each notebook from top to bottom. Executable cells launch training or export jobs, show their output, and then inspect the results. Each notebook's initialization cell puts the kernel's environment first on `PATH`, so `!python job.py` uses that environment and matches the terminal command `python job.py`. You can also run the example scripts from a terminal with that environment activated.

## Run each section's example

Each hands-on section contains its own runnable `code/` directory beside the notebook. Every project uses `job.py`, `client.py`, and `model.py`, with a local dataset module and training functions directly in each client script. Changes in one example do not change another section's code.

```text
chapter-1_running_federated_learning_applications/
├── 01.1_running_federated_learning_job/
│   ├── running_pytorch_fl_job.ipynb
│   └── code/
│       ├── job.py
│       ├── client.py
│       ├── model.py
│       └── ...
├── 01.3_server_side_customization/
│   ├── customize_server_logics.ipynb
│   └── code/
│       ├── job.py
│       ├── client.py
│       ├── model.py
│       └── ...
└── ...
```

| Section | Local example and variation |
| --- | --- |
| [First run](01.1_running_federated_learning_job/code) | Define and run a FedAvg recipe; read its evaluation summary |
| [Convert local training](01.2_convert_deep_learning_to_federated_learning/code) | `train.py`, a training-only `client.py`, and `client_evaluate.py` |
| [Server customization](01.3_server_side_customization/code) | Custom median aggregation, stopping, model selection, and persistence |
| [Client customization](01.4_client_side_customization/code) | Different training arguments per site |
| [TensorBoard](01.5_experiment_tracking/01.5.1_experiment_tracking_with_tensorboard/code) | `SummaryWriter` and the matching server receiver |
| [MLflow](01.5_experiment_tracking/01.5.2_experiment_tracking_with_mlflow/code) | `MLflowWriter` and a local SQLite store |
| [Job structure](01.6_job_structure_and_configuration/code) | Standard and per-site exports from `job.py` |
| [Logging](01.7_logging/code) | Client log messages and console logging modes |
| [Recap](01.8_recap/code) | A local project to modify, export, run, and explain |

To run the first example in Jupyter, open its notebook and run all cells. For an optional terminal run from this chapter directory:

```bash
cd 01.1_running_federated_learning_job/code
python job.py
```

All section jobs use progress logging by default. For detailed console output, run `python job.py --log_config full`. The final run summary shows evaluation results and output locations in either mode.

When moving to another section, open its notebook and run its cells in order. The notebook selects that section's `code` directory and identifies the result paths. Its README also gives terminal commands. The introduction and experiment-tracking overview link to the hands-on sections.

The examples use 200 training and 100 evaluation samples per site, with different site/split seeds and a shared class distribution. They are offline workflow exercises, not benchmarks. Local SGD state is reset every round. Clients report actual optimizer-step counts for built-in FedAvg aggregation; the server customization lesson instead gives each site one vote in a coordinate-wise median. Clients evaluate received weights before training; final evaluation measures the persisted checkpoint separately.

Each example has a distinct recipe name and workspace under `/tmp/nvflare/simulation`. To preserve repeated experiments, set `NVFLARE_SIMULATOR_WORKSPACE_ROOT` to a fresh directory in the notebook environment before executing the run cell. The launched job inherits that setting. The conversion and job-structure variations also have distinct names.

Both tracking labs run 20 rounds to provide 20 points per metric per site. Each notebook starts its UI directly and provides a browser link; stop the UI with the notebook's Stop/Interrupt button when finished. MLflow uses `mlflow.db` in its own section's `code` directory; repeated runs remain in that store.

## Completion and remaining edition work

Complete the [recap exercise](01.8_recap/recap.ipynb): configure a site, inspect the exported recipe, run it, and explain the final model's evaluation in the run summary.

Updates to other chapters, full-course validation, and learner review proceed separately and do not block review of this chapter.
