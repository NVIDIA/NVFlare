# Hello Log Streaming

This example demonstrates how to stream live job logs from clients to the server
in real time using NVFlare's `JobLogStreamer` and `JobLogReceiver` widgets.
It is based on [hello-numpy](../hello-numpy) and adds log streaming with minimal
changes to the job definition.

## Installation

```bash
pip install nvflare
```

Clone the repository and navigate to this directory:

```bash
git clone https://github.com/NVIDIA/NVFlare.git
cd examples/hello-world/hello-log-streaming
pip install -r requirements.txt
```

## Code Structure

```
hello-log-streaming
├── client.py          # client local training script (identical to hello-numpy)
├── job.py             # FL job with JobLogStreamer and JobLogReceiver added
├── requirements.txt   # dependencies
└── README.md
```

## Run the Example

```bash
python job.py
```

## How to Add Log Streaming to Any Job

Log streaming requires two components: one on the client side that tails the
log file and sends it to the server, and one on the server side that receives
and stores the stream.

### Step 1 — Add `JobLogReceiver` to the server

`JobLogReceiver` registers a stream handler at system start.  It receives chunks
from the client and writes them directly to a file named
`{job_id}_{client_name}_{log_file_name}` inside *dest_dir*, so the log can be
followed with `tail -f` while the job is running.  When the stream closes the
file is handed to the job manager for storage alongside other job artifacts.

```python
from nvflare.app_common.logging.job_log_receiver import JobLogReceiver

job.to_server(JobLogReceiver())
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dest_dir` | system temp dir | Directory where incoming log files are written. |
| `idle_timeout` | `30.0` | Seconds without any message before the receiver declares the sender dead and closes the stream. Set to `0` to disable. |

### Step 2 — Add `JobLogStreamer` to the clients

`JobLogStreamer` starts tailing the job's log files when the job begins
(`START_RUN` event) and stops cleanly when the job ends or is aborted
(`ABOUT_TO_END_RUN`).  It must be placed in the **job-level**
configuration (i.e. added to clients, not the system-level resources).

```python
from nvflare.app_common.logging.job_log_streamer import JobLogStreamer

job.to_clients(JobLogStreamer())
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `log_file_name` | `"log.txt"` | Base name of the log file to stream. Add one `JobLogStreamer` per file to stream multiple logs. |
| `liveness_interval` | `10.0` | Seconds of log silence before the client sends a heartbeat to the server. Must be less than the receiver's `idle_timeout`. |
| `poll_interval` | `0.5` | Seconds between polls when no new data has been written to the log. |

### Tuning the timeouts

The liveness / idle-timeout pair controls how quickly a dead client is detected:

```
liveness_interval < idle_timeout
```

With the defaults (10 s and 30 s) the client sends a heartbeat every 10 s when
idle, so the server's 30 s watchdog always resets before it fires.  If you need
faster detection, lower both values while keeping `liveness_interval` strictly
less than `idle_timeout`.

### Complete example

```python
from nvflare.app_common.logging.job_log_receiver import JobLogReceiver
from nvflare.app_common.logging.job_log_streamer import JobLogStreamer

# ... build your recipe / job as usual ...

job.to_clients(JobLogStreamer(liveness_interval=5.0, poll_interval=0.2))
job.to_server(JobLogReceiver(idle_timeout=15.0))
```

### Using the Recipe API

When using a recipe (e.g. `NumpyFedAvgRecipe`), access the underlying job via
`recipe.job`:

```python
recipe = NumpyFedAvgRecipe(...)
recipe.job.to_clients(JobLogStreamer())
recipe.job.to_server(JobLogReceiver())
```

## How It Works

1. **Job starts** — `JobLogStreamer` opens the client's `log.txt` and begins
   tailing it in a background thread, sending 64 KB chunks to the server as
   they are written.
2. **Idle periods** — when no new data is available, the client sends a
   heartbeat every `liveness_interval` seconds so the server knows it is
   still alive.
3. **Server side** — `JobLogReceiver` writes each chunk directly to
   `{dest_dir}/{job_id}_{client_name}_{log_file_name}`, flushing after every
   chunk so the file is always up to date.  Use `tail -f` on that path to
   follow the log in real time.  When the stream closes (clean EOF, job abort,
   or idle timeout), the file is handed to the job manager for storage
   alongside other job artifacts.
4. **Job ends** — `JobLogStreamer` drains any remaining bytes and sends EOF
   before the job process exits.
