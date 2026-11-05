# NVIDIA Flare Test


This file introduces how the tests in NVIDIA FLARE are organized.

We divide tests into unit tests and integration tests.

```commandline
tests:
  - unit_test
  - integration_test
```

## Unit Tests

### Structure

The structure of unit test is organized as parallel directories of the production code.

Each directory in `test/unit_test` maps to its counterpart in `nvflare`.

For example, we have `test/unit_test/app_common/job_schedulers/job_scheduler_test.py`
that tests `nvflare/app_common/job_schedulers/job_scheduler.py`.

### Run

To run unit test: `./runtest.sh`.

### Develop a test case

We use pytest to run our unit tests.
So please follow pytest test case style.

## Integration Tests

Please refer to [integration tests README](./integration_test/README.md).

The offline CoCo manifest launcher workflow runs in
`integration_test/fast/coco_existing_cluster_launch_test.py`. It starts real
Bash/Python processes with fake `kubectl` commands and requires Bash 4.4+ and
Python 3.11+, but no cluster, CVM build, or lab hardware. Premerge runs it serially
outside the unit-test xdist pool on the same OS/Python matrix; the coverage and
Jenkins premerge jobs also collect it. Product readiness/API deadline decisions
remain deterministic unit tests. The shell workflow has a separate 60-second
completion watchdog with process-group cleanup and command-progress diagnostics.

## CVM Builder Tests

`unit_test/lighter/cc/image_builder/` contains the standalone CVM Builder contracts,
fixtures and policy evaluator. Focused tests are grouped under `build`, `runtime`,
`host`, `trustee`, `artifacts` and `common`; cross-package contracts remain at the
test root. The regular unit suite invokes these contracts in
a separate process on Linux through
[`cvm_builder_test.py`](unit_test/lighter/cvm_builder_test.py).

To run the contracts directly from the repository root on Linux:

```sh
PYTHONPATH="$PWD/nvflare/lighter/cc/image_builder${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m unittest discover -s tests/unit_test/lighter/cc/image_builder -v
```

The opt-in storage, HTTPS and hardware tests and lab helpers live in
`integration_test/lighter/cc/image_builder/`. See the
[CVM Builder validation guide](../nvflare/lighter/cc/image_builder/VALIDATION.md#reproducing-tests)
for their setup and commands.
