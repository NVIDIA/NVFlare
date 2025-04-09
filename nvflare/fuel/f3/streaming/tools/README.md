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

