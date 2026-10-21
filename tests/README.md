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

## Concurrency and deadline contracts

A test must establish the state it claims to exercise. Use events at the fake
request handler or I/O boundary before triggering abort, close, or failure. A
`Timer` or `sleep` before an action cannot establish that another thread reached
that boundary. For lock exclusion, `tests.timing_utils.ObservedLock` reports an
actual failed acquisition; an event before a method call still leaves a scheduling
gap. Require both that contention checkpoint and completion after release.

Keep policy time separate from scheduling time. Use `ManualClock` or
`isolate_time(monkeypatch, subject_module)` before patching clock functions.
Patching `subject_module.time.sleep` without isolation changes the shared stdlib
module used by logging, networking, and unrelated workers. Use real bounded waits
for readiness and cleanup; their limits are failure guards, not speed assertions.

Own every background worker through failure paths. Use `ThreadGroup` to release
all gates and join all threads in `finally`, including when a readiness assertion
fails. Propagate worker exceptions to the test thread. Multiprocessing fixtures
must join, terminate, then kill remaining owned children and fail if survivors
remain (`stop_processes`). Do not stop cleanup after the first failed child.

Real interpreter startup, process trees, live sockets/mTLS, and distributed ranks
belong in `integration_test/fast/`. Keep deadline decisions and configuration rules
in unit tests using controlled clocks and process doubles. In integration tests,
first observe output/readiness when the assertion requires it, then trigger the
policy event. Keep a separate unit case proving that real startup time counts
against deadlines. Never increase a timer to hide a missing precondition.

Premerge runs the extracted process/transport suites serially on every existing
unit-test OS/Python combination. Moving a suite must preserve its CI execution. The shared
`ci/run_process_contracts.py` list also runs with `--cov-append` in the coverage
job so this separation does not discard measured paths.
`python3 ci/check_unit_test_contracts.py` rejects timer-driven test interleavings
and unisolated clock patches. This focused check cannot prove thread safety;
review concurrency tests against delayed startup, missing locks, and failed setup.
