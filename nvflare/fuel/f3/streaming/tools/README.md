# Streaming API Testing Tools

The following command line tools are not part of the F3 streaming library. 
They are provided to test the streaming API.

## Sending a memory buffer

sender.py/receiver.py can be used to send a memory buffer to the remote cell. 
The memory buffer size must fit in the memory.

Starting receiver side first:

     python receiver.py grpc://0:1234

Starting sender on the same or different machine:

     python sender.py grpc://192.168.1.2:1234 -s 100000000

where `-s` is the buffer size in bytes.

## Sending a file

The old file_sender.py/file_receiver.py files are removed because they use 
deprecated API.

Please use this NVFlare example job to test file streaming. This example tests full stack APIs of NVFlare,
and it's a more realistic test

     NVFlare/tests/tools/file_streaming

## Receive-task cleanup diagnostics

Completed reliable streams remain available for duplicate ACKs during
`streaming_retry_timeout + streaming_retry_wait` (65 seconds by default).
The receiver also honors longer validated sender retry windows. One process-wide
deadline worker expires these records without allocating a thread per stream.
Non-finite or negative local retry-window totals produce a warning and fall back
to the default 65 seconds. Large finite deadlines are preserved; individual waits
are bounded by the platform's threading timeout limit.
Failed non-reliable streams keep their existing five-second tombstones;
successful non-reliable streams are removed immediately.

For an in-process diagnostic snapshot:

```python
from nvflare.fuel.f3.streaming.byte_receiver import RxTask

print(RxTask.get_cleanup_stats())
```

`retained_tasks` counts all entries in the receive-task map, including active
streams. `pending` counts scheduled cleanups; `queue_depth` also includes cancelled
entries awaiting pruning. `expired` and `cancelled` are lifetime counters.
`max_cleanup_lag` is the maximum observed delay in seconds between a deadline and
the start of its cleanup. `worker_alive` and `stopped` describe scheduler lifecycle.
Queue gauges can differ briefly while cleanup is running. Process shutdown drains
the streaming pools, clears the receive map and queue, and joins the deadline worker.

