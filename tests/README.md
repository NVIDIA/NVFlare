# NVIDIA FLARE Tests

These rules apply to both human and AI contributors adding, changing, or moving
tests. Classify each case by the behavior and boundaries it verifies. An existing
file's location is not evidence that new cases belong there.

## Choosing the test layer

| Location | What belongs here | Examples |
| --- | --- | --- |
| `tests/unit_test/` | Component behavior and policy in one interpreter, with external boundaries controlled by the test. | Deadline decisions with a fake clock/process; config validation; aggregation math; serialization; cancellation after a fake request handler has entered. |
| `tests/integration_test/fast/` | Small, bounded interactions across real runtime boundaries, suitable for premerge CI. | Interpreter/CLI startup and exit; process trees and reaping; IPC; live Cells, sockets, or mTLS; distributed ranks; a minimal workflow across components. |
| `tests/integration_test/slow/` | Integration scenarios whose duration or resource/dependency needs require nightly or dedicated CI. | Extended federated training; large models/data; expensive provisioning; GPU scenarios; scenarios requiring downloads or specialized environments. |

Short runtime does not make a process or transport lifecycle test a unit test.
Conversely, temporary files, small CPU tensor operations, serialization, and
synchronized local threads can be valid unit-test tools. Importing a real library
does not by itself make a test integration-level; executing its distributed or
external runtime does. Choose fast versus slow integration by actual cost and
resources, not just whether a test uses the simulator or POC mode.

Split mixed suites: keep decisions/configuration in unit tests and verify real
startup, transport, and cleanup in integration tests. A fake boundary proves how
the component uses that boundary; it cannot prove the real runtime works.

Config-driven system scenarios use `tests/integration_test/system_test.py`, configs
under `data/test_configs/`, and selection through `test_configs.yml`. That YAML is
not a registry for direct pytest suites. See the
[integration guide](integration_test/README.md) for setup and commands.

The existing lightweight CI-checker suites in `tests/ci/` must keep working in
their declared minimal environment. Shared test fixtures must not eagerly import
NVFlare or optional ML dependencies into such checks.

## Unit test structure and commands

Mirror the production package under `tests/unit_test/`. For example,
`tests/unit_test/app_common/job_schedulers/job_scheduler_test.py` tests
`nvflare/app_common/job_schedulers/job_scheduler.py`. Use pytest conventions and
the `[module_name]_test.py` filename pattern for new unit modules.

Run from the repository root:

```bash
./runtest.sh -u
python3 -m pytest -q tests/unit_test/app_common/job_schedulers/job_scheduler_test.py
```

## Meaningful assertions

State the behavior being verified and what wrong implementation the test must
reject. Assert observable results, state transitions, error handling, protocol
payloads, or required ordering. Mock-call assertions are appropriate when the
boundary invocation is itself the contract. Merely returning a value configured
on a mock, instantiating a component, or completing without an exception does not
prove an unrelated behavior. A smoke test should name the limited contract it
actually checks.

Cover distinct lifecycle outcomes explicitly: successful completion, local
cancellation/shutdown, and remote or application failure can require different
behavior. A test for one outcome does not establish correctness for the others.

## Concurrency and deadline contracts

A test must establish the state it claims to exercise. Use events at the fake
request handler or I/O boundary before triggering abort, close, or failure. A
`Timer` or `sleep` before an action cannot establish that another thread reached
that boundary. For lock exclusion, `tests.timing_utils.ObservedLock` reports an
actual failed acquisition; an event before a method call still leaves a scheduling
gap. Require both that contention checkpoint and completion after release.

Keep intentionally blocked callbacks gated until the test releases them in
`finally`. A callback's own timeout can free a worker and falsely pass a test
intended to detect blocking. Capture callback exceptions separately and assert
them in the test thread, especially when production code catches callback errors.

Keep policy time separate from scheduling time. Use `ManualClock` or
`isolate_time(monkeypatch, subject_module)` before patching clock functions.
Patching `subject_module.time.sleep` without isolation changes the shared stdlib
module used by logging, networking, and unrelated workers. Match the clock fake
to the functions the subject uses: `ManualClock` controls `time` and `monotonic`,
while other functions such as `perf_counter` and `time_ns` remain real.

Use real bounded waits for readiness and cleanup. Ensure the guarded regression
also fails within a bound: a frozen-clock poll loop needs a finite poll budget or
an independent real-time watchdog. A policy clock that stops advancing must not
hang CI. Where prompt cancellation is a requirement, retain a response bound
measured from the actual abort/cancel trigger, excluding readiness/setup time.

Own every background worker through failure paths, including partial startup.
Register cleanup before launching work. Use `ThreadGroup` to release all gates
and join all threads in `finally`, including when a readiness assertion fails.
Propagate worker exceptions to the test thread. Multiprocessing fixtures must
join, terminate, then kill remaining owned children and fail if survivors remain
(`stop_processes`). Do not stop cleanup after the first failed child. Register
shared pytest fixtures under one name so alias exports do not wrap/clean up twice.

In integration tests, observe output/readiness when the assertion requires it,
then trigger the policy event. Keep a separate unit case proving that startup
time counts against deadlines. Never increase a timer to hide a missing
precondition or delete a required latency assertion to make a test pass.

## CI execution and coverage

Before moving or adding a suite, check the actual workflow/runner command that
collects it. Directory placement alone does not schedule a test. Preserve intended
OS/Python coverage, dependency setup, and coverage reporting when splitting suites.

Premerge runs the extracted process/transport suites serially in a fresh pytest
invocation on every existing unit-test OS/Python combination, outside the unit
xdist run. Their shared selection is `ci/run_process_contracts.py`; the coverage
job uses the same list with `--cov-append`. Other integration suites have their
own CI selections; do not add every fast suite to this specific runner. Keep
conditional skip reasons visible in CI logs, for example with pytest `-rs`.

`python3 ci/check_unit_test_contracts.py` rejects timer-driven test interleavings
and unisolated clock patches. This focused check cannot prove thread safety or
find every indirect patch; review the behavioral contract even when it passes.

## Test review checklist

- What observable behavior is verified, and which concrete regression fails it?
- Does the chosen layer match the real boundaries and resources exercised?
- Are the asserted preconditions observed before the action that depends on them?
- Would the wrong implementation fail promptly, rather than hang or pass after a gate times out?
- Are clocks, fixtures, and worker errors isolated, and all workers cleaned up after assertion/startup failure?
- Does the actual CI command collect the case with the intended matrix, dependencies, coverage, and visible skips?
