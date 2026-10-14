# Research Project Title

Replace this heading with the full title of the research contribution.

Use this template for an implementation hosted under `research/`. To contribute
only a link to an external NVFlare example, follow the
[research contribution guide](../CONTRIBUTING.md); no local project directory or
copy of this template is required.

## Abstract

Provide a short abstract or summary of the work. This should explain the
federated learning problem, the proposed method or implementation, and the main
result or expected use case.

## Papers and Links

- Paper: [Paper title](https://example.com)
- Preprint: [arXiv link](https://arxiv.org)
- Project page or upstream code: [Repository or project link](https://example.com)

A paper or preprint is encouraged but not required. If the work has not been
published, state that clearly and link to any available technical report,
benchmark description, or project page. A substantive README describing the
method, implementation, and validation qualifies as public documentation.

## Objective

Describe the main objective of the contribution in two or three sentences. Make
it clear why this research implementation belongs in NVIDIA FLARE and what a
reader can learn or reproduce from it.

## Method Summary

Summarize the core method, algorithm, model, dataset setting, and federated
learning scenario. Include the parts that are important for reproduction, such
as horizontal or vertical FL, client heterogeneity, privacy mechanisms,
aggregation strategy, or personalization method.

## Repository Layout

Describe the files and folders included in the project. Use the structure that
best fits the research implementation. For example, a project might include:

```text
sample-research/
|-- README.md
|-- requirements.txt
|-- job.py
|-- jobs/
|-- src/
|-- notebooks/
|-- scripts/
`-- figs/
```

Not every project needs every folder. Use `job.py`, `jobs/`, scripts, notebooks,
or links to external reference code as appropriate, but document how each piece
is used.

## Setup

List the software, hardware, and data prerequisites. Include Python and package
installation commands, and mention GPU, memory, or framework requirements when
they matter.

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If the project uses a `pyproject.toml`, conda environment file, task-specific
requirements file, or unreleased NVFlare version, document the exact install
steps here.

## Data Preparation

Explain how to obtain, generate, or prepare the data. Include download links,
expected directory structure, preprocessing commands, synthetic data generation,
and any license or access restrictions.

If public data cannot be redistributed, describe the expected input format and
provide a small synthetic or mock-data path when possible.

## Run Instructions

Show the commands needed to reproduce the main experiment. Use the launch method
that matches the implementation, such as an NVFlare simulator command, a Recipe
`job.py`, a script, or a notebook.

```bash
# Example only; replace with the command for this project.
python job.py --num_clients 2 --num_rounds 5
```

Also document where logs, workspaces, checkpoints, metrics, and generated
figures are written.

## Validation performed

Describe a successful run you performed that exercises the relevant NVFlare
workflow and produces an identifiable result. A short representative run is
sufficient for functional validation; installation or process startup alone is
not. Record:

- **Revision:** tested project commit or release.
- **Environment:** NVFlare version, relevant dependencies and hardware, and the
  NVFlare commit when testing unreleased features.
- **Command:** the command and configuration used, including the data or
  synthetic/mock-data input.
- **Result:** the observed output, such as a metric, checkpoint, or completed
  workflow result, with a concise summary or link to a representative log or
  artifact.
- **Limitations:** what this run validates and what remains untested.

Distinguish evidence that the integration works from evidence supporting any
performance, privacy, or robustness claims, as described in the
[research contribution guide](../CONTRIBUTING.md#quality-and-validation).

## Expected Results

Describe the expected metrics, figures, checkpoints, or qualitative outputs.
Include example tables or images when they help readers confirm that the
experiment ran correctly.

## License

State the license for the contribution and call out any third-party code,
models, data, or assets with separate license terms. Contributions in this
repository must be compatible with the repository license requirements.

## Requirements

List dependencies in the most appropriate project file, such as
`requirements.txt`, `pyproject.toml`, or an environment file. Include the
NVFlare version used by the research implementation. On the `main` branch, this
may be an upcoming NVFlare version when the example depends on unreleased
features; in that case, tell users to install NVFlare from this repository until
the package is published.

## Citation

If users should cite this work, provide the preferred citation and BibTeX entry.

```bibtex
@article{sample2026research,
  title = {Research Project Title},
  author = {Author One and Author Two},
  journal = {arXiv preprint arXiv:0000.00000},
  year = {2026}
}
```
