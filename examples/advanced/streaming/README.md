# Object Streaming Examples

## Overview
The examples here demonstrate how to use object streamers to send large file/objects memory efficiently.

The object streamer uses less memory because it sends files by chunks (default chunk size is 1MB) and 
it sends containers entry by entry.

For example, if you have a dict with 10 1GB entries, it will take 10GB extra space to send the dict without
streaming. It only requires extra 1GB to serialize the entry using streaming.

## Concepts

### Object Streamer

ObjectStreamer is a base class to stream an object piece by piece. The `StreamableEngine` built in the NVFlare can
stream any implementations of ObjectSteamer

Following implementations are included in NVFlare,

* `FileStreamer`: It can be used to stream a file
* `ContainerStreamer`: This class can stream a container entry by entry. Currently, dict, list and set are supported

The container streamer can only stream the top level entries. All the sub entries of a top entry are sent at once with
the top entry.

### Object Retriever

`ObjectRetriever` is designed to request an object to be streamed from a remote site. It automatically sets up the streaming
on both ends and handles the coordination.

Currently, following implementations are available,

* `FileRetriever`: It's used to retrieve a file from remote site using FileStreamer.
* `ContainerRetriever`: This class can be used to retrieve a container from remote site using ContainerStreamer.

To use ContainerRetriever, the container must be given a name and added on the sending site,

```
ContainerRetriever.add("model", model_dict)
```

## Example Jobs

### file_streaming job

This job uses the FileStreamer object to send a large file from server to client. 

It demonstrates following mechanisms:
1. It uses components to handle the file transferring. No training workflow is used. 
   Since executor is required by NVFlare, a dummy executor is created.
2. It shows how to use the streamer directly without an object retriever.

The job creates a temporary file to test. You can run the job in POC or using simulator as follows,

```
nvflare simulator -n 1 -t 1 jobs/file_streaming
```
### dict_streaming job

This job demonstrate how to send a dict from server to client using object retriever.

It creates a task called "retrieve_dict" to tell client to get ready for the streaming.

The example can be run in simulator like this,
```
nvflare simulator -n 1 -t 1 jobs/dict_streaming
```
