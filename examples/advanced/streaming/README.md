# Object Streaming

## Overview
The examples here demonstrate how to use object streamers to send large objects in a memory-efficient manner.

Current default setting is to send and receive large objects in full, so extra memory will be needed and allocated to hold the received message. 
This works fine when the message is small, but can become a limit when model size is large, e.g. for large language models.

To save on memory usage, we can stream the message send / receive: when sending large objects (e.g. a dict),
streamer sends containers entry by entry (e.g. one dict item each time); further, if we save the object to a file, 
streamer can send the file by chunks (default chunk size is 1MB).

Thus, the memory demand can be reduced to the size of the largest entry for container streaming; while nearly no extra memory is needed for file
streaming. For example, if sending a dict with 10 1GB entries, without streaming, it will take 10GB extra space to send the dict. 
With container streaming, it only requires extra 1GB; and if saved to a file before sending, it only requires 1MB extra space to send the file.

All examples are run with NVFlare Simulator via [JobAPI](https://nvflare.readthedocs.io/en/main/programming_guide/fed_job_api.html).
## Concepts

### Object Streamer
ObjectStreamer is the base class to stream an object piece by piece. The `StreamableEngine` built in the NVFlare can
stream any implementations of ObjectSteamer

The following implementations are included in NVFlare,

* `ContainerStreamer`: This class is used to stream a container entry by entry. Currently, dict, list and set are supported
* `FileStreamer`: This class is used to stream a file

Note that the container streamer split the stream by the top level entries. All the sub entries of a top entry are expected to be
sent as a whole, therefore the memory is determined by the largest entry at top level.

### Object Retriever
Building upon the streamers, `ObjectRetriever` is designed for easier integration with existing code: to request an object to be streamed from a remote site. It automatically sets up the streaming
on both ends and handles the coordination.

Similarly, the following implementations are available,

* `ContainerRetriever`: This class is used to retrieve a container from remote site using `ContainerStreamer`.
* `FileRetriever`: This class is used to retrieve a file from remote site using `FileStreamer`.

Note that to use ContainerRetriever, the container must be given a name and added on the sending site,
```
ContainerRetriever.add_container("model", model_dict)
```

## Simple Examples
First, we demonstrate how to use the Streamer directly without Retriever:
```commandline
python simple_file_streaming_job.py
```
Note that in this example, the file streaming is relatively "standalone", as the `FileReceiver` and `FileSender`
are used directly as components, and no training workflow is used - as executor is required by NVFlare, here we used 
a dummy executor.

Although the file streaming is simple, it is not very practical for real-world applications, because 
in most cases, rather than standalone, we need to send an object when it is generated at certain point in the workflow. In such cases, 
Retriever is more convenient to use:
```commandline
python simple_dict_streaming_job.py
```
In this second example, the `ContainerRetriever` is setup in both server and client, and will automatically handle the streaming.
It couples closely with the workflow, and is easier to define what to send and where to retrieve.

## Full-scale Examples and Comparisons
The above two simple examples illustrated the basic usage of streaming with random small messages. In the following, 
we will demonstrate how to use the streamer with Retriever in a workflow with real large language model object, 
and compare the memory usage with and without streaming. To track the memory usage, we use a simple script `utils/log_memory.sh`. 
Note that the tracked usage is not fully accurate, but it is sufficient to give us a rough idea.

All three settings: regular, container streaming, and file streaming, are integrated in the same script to avoid extra variabilities.
To run the examples:
```commandline
bash regular_transmission.sh
```
```commandline
bash container_stream.sh
```
```commandline
bash file_stream.sh
```

We then examine the memory usage by comparing the peak memory usage of the three settings. The results are shown below,
note that the numbers here are the results of one experiment on one machine, and can be highly variable depending on the system and the environment.

| Setting | Peak Memory Usage (MB) | Job Finishing Time (s) |
| --- | --- | --- |
| Regular Transmission | 42,427 | 47
| Container Streaming | 23,265 | 50
| File Streaming | 19,176 | 170

As shown, the memory usage is significantly reduced by using streaming, especially for file streaming, 
while file streaming takes much longer time to finish the job.







