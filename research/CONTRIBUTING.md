# Research Directory

The `research` directory hosts community research work that uses NVIDIA FLARE.
Research projects are reviewed through the pull request process, but the code is
not maintained by the NVIDIA FLARE team after it is contributed.

## License

By contributing code to this repository, you agree that the contribution can be
released under the Apache v2 License or another compatible open source license.
Call out any third-party code, models, datasets, or assets that have separate
license terms.

## Project Directory

Each research project should create a subdirectory under `research/`.

- Prefer a lowercase, ASCII, kebab-case directory name no longer than 35
  characters for new projects.
- Existing research folders may preserve their published or historical names.
- Keep project-specific code, configs, notebooks, scripts, figures, and docs
  inside the project directory unless the code is intentionally shared by
  multiple research projects.

## README Expectations

Every project should include a `README.md`. Start from the
[sample research template](./sample-research/README.md) and adapt it to the
project. A good research README includes:

- Title and abstract.
- Links to published papers, preprints, project pages, and upstream code when
  available.
- Objective and method summary.
- Repository layout.
- Setup instructions.
- Data download, generation, or preparation instructions.
- Steps to run the code.
- Expected results, metrics, figures, or checkpoints.
- License notes.
- Requirements and the NVFlare version used.
- Citation or BibTeX entry when applicable.

The section names do not need to be identical to the template, but the README
should give readers enough information to understand and reproduce the work.

## Implementation Layout

Use the layout that best fits the research contribution. For example, a project
may use one or more of the following:

- `job.py` for a Recipe-based experiment.
- `jobs/` for exported NVFlare job configurations and custom code.
- `src/` for project-specific Python modules.
- `scripts/` for setup, preprocessing, launch, or plotting commands.
- `notebooks/` for exploratory or tutorial workflows.
- `figs/` or `assets/` for result images.
- Links to external reference implementations when the full code is maintained
  elsewhere.

The project does not need all of these paths. Whichever layout you choose, keep
the code runnable, document the launch command, and explain where logs, metrics,
workspaces, checkpoints, and generated artifacts are written.

## Requirements

List dependencies in the project file that best matches the implementation, such
as `requirements.txt`, `pyproject.toml`, a conda environment file, or
task-specific requirements files. Include the NVFlare version used.

On the `main` branch, a research project may depend on an upcoming NVFlare
version when it uses unreleased features. In that case, keep the intended version
pin and document that users should install NVFlare from this repository until the
package is published.

## Setup

To run a research project, we recommend using a virtual environment unless the
project README specifies another environment.

```bash
python3 -m pip install --user --upgrade pip
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

Then follow the setup instructions in the project README, such as installing
`requirements.txt`, installing from `pyproject.toml`, creating a conda
environment, or preparing task-specific dependencies.
