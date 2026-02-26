# NVFlare Agent Notes

- To trigger CI/CD from a PR review thread, post a single-line comment exactly: `/build`.
- Prefer `rg` and `rg --files` for fast codebase search.
- Keep edits scoped to the task; do not modify unrelated files in a dirty worktree.
- Start with targeted tests for changed files, then run broader checks as needed.

## Fast Commands

- `./runtest.sh` runs license/style/tests with coverage.
- `./runtest.sh -s` runs style checks (flake8, black, isort).
- `./runtest.sh -f` auto-fixes style where possible.
- `./runtest.sh -u` runs unit tests.
- `python3 -m pytest tests/unit_test/path/to/test_file.py -v` runs one test file.
- `python3 -m pytest --numprocesses=8 -v tests/unit_test` runs unit tests in parallel.
- `./build_doc.sh --html` builds docs.
- `./build_doc.sh --clean` cleans docs build artifacts.

## Style and Testing Conventions

- Format/lint stack: black (line length 120), flake8, isort (black profile).
- Python support targets: 3.9, 3.10, 3.11, 3.12.
- Add the standard NVIDIA Apache-2.0 license header to new Python source files.
- Unit tests live in `tests/unit_test/`; integration tests live in `tests/integration_test/`.
- Test file names follow `[module_name]_test.py`.

## Quick Package Map

- `nvflare/apis/`: core interfaces (Controller, Executor, Task, Shareable, FLContext).
- `nvflare/app_common/`: common algorithms and utilities.
- `nvflare/app_opt/`: optional integrations/dependencies.
- `nvflare/client/`: client-side APIs.
- `nvflare/job_config/`: FedJob/job configuration.
- `nvflare/private/`: internal implementations.
- `nvflare/fuel/`: shared infrastructure utilities.
