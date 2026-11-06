# Research Directory

The `research/` directory showcases published or publicly documented methods and
examples that use NVIDIA FLARE. A paper or preprint is encouraged but not required.
A substantive project README describing the method, implementation, and validation
qualifies as public documentation.

Contributions should provide a meaningful, working NVFlare example with evidence
that the contributor has exercised the relevant workflow successfully.
Contributors are responsible for understanding, reviewing, and validating their
code, documentation, and claims, including material produced with AI tools.
Untested code, placeholder projects, and unsupported claims are not ready for
inclusion.

Research contributions are reviewed through the pull request process.
Contributors retain responsibility for their research projects, whether linked
externally or hosted here, and should be able to explain the implementation,
answer review questions, and address issues. Research code contributed to this
repository is not maintained by the NVIDIA FLARE team after contribution;
external projects likewise remain maintained by their authors.

## Quality and Validation

Both external listings and hosted implementations should provide:

- A clear explanation of the problem, approach, and NVFlare functionality
  demonstrated.
- Setup instructions, dependencies, data requirements, and a concrete command
  for running the relevant NVFlare workflow.
- Evidence of a successful representative run performed by the contributor.
  A short run that exercises the integration and produces an identifiable result
  is sufficient to demonstrate functionality. Installation success or a
  process-startup log alone is not sufficient.
- The tested project commit or release, relevant environment, NVFlare version,
  run command, and observed result. Include the NVFlare commit when testing
  unreleased features. A concise result summary, representative log, or output
  artifact can document the result.
- Known limitations and a distinction between demonstrated behavior and proposed
  capabilities.

Functional validation establishes that an integration works. Claims about
performance, privacy, or robustness require evidence appropriate to those claims;
not every contribution needs a benchmark campaign. A publication does not replace
validation of the implementation.

Reviewers may request additional evidence when warranted. Independent
reproduction by an NVFlare maintainer is not a prerequisite for every listing.

## Contribution Paths

### External Project Listings

If your project already has its own author-maintained repository, we encourage
you to maintain the implementation and evaluation there and contribute a link
to the [research README](./README.md). This keeps development with the project's
authors and helps NVFlare users discover the work.

Open a small documentation PR adding an entry to the README's project table with
the project name, a direct link to its NVFlare example or integration, and a short
description identifying it as externally maintained. Keep the supporting
documentation and validation evidence in the upstream project or link to them
from the PR.

A link-only contribution does not require a local project subdirectory, copied
code, or a local copy of the sample research template.

### Implementations Hosted in NVFlare

Full implementations contributed under `research/` are also welcome. Follow the
sections below and the repository's [contribution guidelines](../CONTRIBUTING.md),
including the applicable code-review and testing requirements. The local
directory and template requirements below apply to hosted implementations.

## License

By contributing code to this repository, you agree that the contribution can be
released under the Apache v2 License or another compatible open source license.
Call out any third-party code, models, datasets, or assets that have separate
license terms.

## Project Directory

Each hosted research implementation should create a subdirectory under `research/`.

- Prefer a lowercase, ASCII, kebab-case directory name no longer than 35
  characters for new projects.
- Existing research folders may preserve their published or historical names.
- Keep project-specific code, configs, notebooks, scripts, figures, and docs
  inside the project directory unless the code is intentionally shared by
  multiple research projects.

## README Expectations

Each hosted implementation should include a `README.md`. Start from the
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
- Validation performed, including the tested revisions, environment, command,
  observed result, and validation limitations.
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
