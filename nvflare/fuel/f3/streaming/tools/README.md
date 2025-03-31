# Streaming API Testing Tools

The following command line tools are provided to test the streaming API.

There are 2 sets of tools:
1. sender.py/receiver.py: Those can be used to send a memory buffer to the remote cell. The memory buffer size must
   fit in the memory.
2. file_sender.py/file_receiver.py: This uses StreamCell to send/receive files. This can be used to send 
   files of any size, as long as receiving end has enough disk space.


## Sending a memory buffer

Starting receiver side first:

     python receiver.py grpc://0:1234

Starting sender on the same or different machine:

     python sender.py grpc://192.168.1.2:1234 -s 100000000

## Sending a file

The old file_sender.py/file_receiver.py files are removed because they use 
deprecated API.

Please use this NVFlare example job to test file streaming. This example tests full stack APIs of NVFlare,
and it's a more realistic test

     NVFlare/tests/tools/file_streaming

