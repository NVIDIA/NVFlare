# Simple asynchronous MNIST training with the Collab API

This single-file example teaches `collab.clients(blocking=False).train(...)`
with two clients training an MNIST classifier on a CPU. It uses ordinary
module-level functions, with **no class definitions**:

- `@collab.publish` exposes the client `train(weights, version)` function.
- `@collab.main` marks the server `run()` function.
- `CollabRecipe()` discovers those functions in the calling module when
  `server` and `client` are omitted.

Read [job.py](job.py) from `make_model()` and `train()` to `run()`, then the
recipe and launch code at the bottom.

## Run

With NVFlare 2.9 or later installed, download and run the standalone example:

```bash
nvflare examples get collab-simple-async
cd collab-simple-async
python -m pip install -r requirements.txt
python job.py
```

Alternatively, from the NVFlare repository root:

```bash
python -m pip install -e .
cd examples/advanced/collab/simple_async
python -m pip install -r requirements.txt
python job.py
```

The script downloads MNIST once before starting the simulator. PyTorch and
torchvision are the only additional dependencies, listed in
[requirements.txt](requirements.txt). No GPU is required.

The default is **20 total accepted updates**, with two clients active at startup.
Each update trains for one local epoch. `site-2` adds a one-second delay to
illustrate a slower client. Actual completion order also depends on runtime,
training, evaluation, and transport overhead.

Settings are module-level constants near the top of [job.py](job.py):

```python
NUM_UPDATES = 20
DATA_ROOT = "/tmp/nvflare/data"
WORKSPACE_ROOT = "/tmp/nvflare/collab/simple_async"
```

Edit `NUM_UPDATES` to run longer (for example, 40 updates), `DATA_ROOT` to move
the MNIST cache, or `WORKSPACE_ROOT` to choose another output directory. Then
run `python job.py`. Each worker imports these values from the example source;
there are no command-line options or server/client property lookups for them.

## Model and data

The model is an `nn.Sequential` network: flatten the 28-by-28 image, apply a
128-unit hidden layer with ReLU, and predict ten digit classes. The server
initializes one model and every assignment starts from a snapshot of its weights.
Clients use SGD with learning rate 0.1, batch size 128, and one local epoch.
The optimizer is recreated for each assignment.

The 60,000 training images are split into two disjoint, equally sized shards
using alternating dataset indices. Each simulated site trains only on its
30,000-image shard. A single local download supplies both simulated sites;
for distributed deployment, prepare the corresponding data at each site.
Each client caches its training loader in a Collab app property, keeping it
local and reusable across updates for the lifetime of the job.
The server evaluates each global model on the separate 10,000-image MNIST test
set. That test data is never used for client training.

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
would wait for the whole group before processing the collected results.
This snippet shows one dispatch; the runnable example continuously gives
finished clients more work.

## The asynchronous training loop

1. Dispatch the initial model to both clients with `blocking=False`.
2. Receive successful responses through `process_resp_cb`, which places each
   `(site, result)` into a thread-safe queue.
3. Take the next response and mix each parameter into the current model using
   `0.5 * global_parameter + 0.5 * client_parameter`.
4. Send the new model only to the client that finished. Other clients keep
   training on their original snapshots; there is no round barrier.
5. Evaluate the new model after refill so the finished client can already train
   again. Stop submitting at the update budget, accept the remaining submitted
   updates, and drain the call streams before returning.

Only the main server loop changes the global model. Each response includes its
base model version, so the log can report **staleness**: the number of global
updates since that version was dispatched. Model snapshots are independent of
the model used for evaluation.

A failed or timed-out call does not invoke the success callback. The loop checks
`results.failures` and fails explicitly; it also checks for an abort. To keep the
example small, there are no retries, client selection, staleness weighting, or
buffers. Faster clients can contribute more updates, and stale models receive
the same mixing weight as fresh ones. Loss can temporarily increase when a
stale model arrives. The finite list of call streams is intended for this
bounded demo, not an unbounded training service.

## Results

The log includes initial test metrics, per-update loss and accuracy, staleness,
and the time spent on training and evaluation. Server-side evaluation costs time
and can affect arrival order; these timings describe the complete teaching loop,
not isolated communication throughput or an async-versus-sync benchmark.

A local macOS CPU validation run with cached MNIST, PyTorch 2.14.1,
torchvision 0.29.1, and one PyTorch intra-op thread per process produced:

| Accepted updates | Test accuracy | Test cross-entropy loss | Training + evaluation | End-to-end runtime |
|---|---:|---:|---:|---:|
| 0 (initial model) | 10.43% | 2.3041 | — | — |
| 20 (default) | 94.19% | 0.1994 | 8.7 s | 27.8 s |
| 40 | 95.89% | 0.1346 | 17.1 s | 37.0 s |

The 20- and 40-update rows are separate runs from the same initial seed.
Training/evaluation time starts at initial dispatch and ends after the last
update is evaluated and the call streams are drained. End-to-end time also
includes Python startup, job setup, initial evaluation, and simulator teardown;
it excludes the first MNIST download in these cached-data runs. Timings and
accuracy can vary with hardware and arrival order. This demonstrates learning
on MNIST with the simple mixing rule, rather than a claim of optimal accuracy
or a speedup over synchronous training.

The default workspace is `/tmp/nvflare/collab/simple_async`. The server run
directory contains:

- `final_model.pt`: the final PyTorch state dictionary.
- `metrics.json`: initial and per-update test loss, accuracy, elapsed seconds,
  and total training/evaluation time.

The script prints the full result location and job status. To load the model:

```python
import torch
from job import make_model

model = make_model()
model.load_state_dict(torch.load("/path/to/final_model.pt", weights_only=True))
model.eval()
```

## Next steps

- [Response callbacks](https://github.com/NVIDIA/NVFlare/blob/main/examples/advanced/collab/response_callbacks/avg_intime.py) uses a response
  callback within **blocking, round-based** group calls.
- [CIFAR-10 FedBuff](https://github.com/NVIDIA/NVFlare/blob/main/examples/advanced/collab/pt_async_cifar10/README.md) adds buffered aggregation,
  scheduling controls, evaluation, and checkpointing.
- [All Collab examples](https://github.com/NVIDIA/NVFlare/blob/main/examples/advanced/collab/README.md) lists the other available examples.
