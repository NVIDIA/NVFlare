# Simple asynchronous FL with the Collab API

This single-file example teaches `collab.clients(blocking=False).train(...)`
using two clients and a tiny NumPy linear-regression model. It runs on a CPU
with synthetic local data; no dataset download or extra ML framework is needed.
Read [job.py](job.py) from the client `Trainer.train()` to the server
`AsyncServer.run()` and finally the recipe in `main()`.

## Run

From the NVFlare repository root, install this checkout, then run the example:

```bash
python -m pip install -e .
cd examples/advanced/collab/simple_async
python job.py
```

NumPy is included in NVFlare's dependencies. To accept more updates or choose
another output directory:

```bash
python job.py --num-updates 20 --workspace-root /tmp/collab-simple-async
```

The default is **eight total accepted updates**, with two clients active at
startup. `site-1` simulates 0.1 seconds of training and `site-2` simulates 1 second.
Actual completion order also depends on runtime and transport overhead.

## The nonblocking call

Call options belong on `clients`, before the method name:

```python
results = collab.clients(blocking=False).train(weights, version)
# The server can do other work here while the clients train.
for site, result in results:
    print(site, result)  # waits only until the next successful result arrives
for site, error in results.failures.items():
    print(site, error)
```

The result stream is single-pass. Exhausting it waits for every outcome and
makes the failure set complete. Converting it to `dict(results)` immediately
would wait for the whole group before you could process the collected results.
The snippet above demonstrates one dispatch; continuous asynchronous training
also needs to give finished clients more work.

## The asynchronous training loop

The runnable example keeps the server in charge of scheduling:

1. Dispatch the initial model to both clients with `blocking=False`.
2. Receive successful responses through `process_resp_cb`, which puts each
   `(site, result)` into a thread-safe queue.
3. Take the next response from that queue and immediately mix its weights into
   the global model: `weights = 0.5 * weights + 0.5 * local_weights`.
4. Send the new model only to the client that finished. The other client keeps
   training on its original model; there is no round barrier.
5. Stop submitting work once the update budget is reached, accept all remaining
   submitted updates, and drain the result streams before returning.

The callback only queues results. All model changes and new dispatches happen
in the main server loop, so aggregation needs no lock. Each result includes the
version used for training; the server logs its **staleness**, the number of
model updates made since that version was dispatched. For example, an update
with `base_version=0, staleness=3` arrived after three other updates had already
changed the global model.

A failed or timed-out call does not invoke the success callback. The loop checks
`results.failures` and fails the run explicitly; it also checks for an abort.
This teaching example intentionally has no retries, client selection policy,
staleness weighting, or buffered aggregation. Faster clients can contribute more
updates, and stale weights are mixed with the same fixed factor as fresh ones.
The finite list of call streams is suitable for this small demo, rather than an
unbounded training service.

## What to observe

In the server log, look for a client receiving a newer model version before the
other client's earlier work finishes. `staleness > 0` shows an update trained on
an older model. There should be exactly `--num-updates` accepted-update lines.
The synthetic regression target is `[2.0, -1.0]`; the final weights depend on
arrival order and the chosen update budget.

The default workspace is `/tmp/nvflare/collab/simple_async`. The server run
directory contains `final_model.npy`, and the script prints its full path plus
the job status and result location.

## Next steps

- [In-time aggregation](../async_aggregation/avg_intime.py) uses a response
  callback within **blocking, round-based** group calls.
- [CIFAR-10 FedBuff](../pt_async_cifar10/README.md) adds buffered aggregation,
  scheduling controls, evaluation, and checkpointing for a real training task.
- [All Collab examples](../README.md) provides the broader learning path.
