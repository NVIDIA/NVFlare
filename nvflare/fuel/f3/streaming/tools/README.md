# Streaming API Testing Tools

The following command line tools are provided to test the streaming API.

There are 2 sets of tools:
1. sender.py/receiver.py: Those can be used to send a memory buffer to the remote cell. The memory buffer size must
   fit in the memory.
2. file_sender.py/file_receiver.py: This uses StreamCell to send/receive files. This can be used to send 
   files of any size, as long as receiving end has enough disk space.


## Sending a memory buffer

Starting receiver side first:

     python receeiver.py grpc://0:1234

Starting sender on the same or different machine:

     python sender.py grpc://192.168.1.2:1234 -s 100000000

## Send a file

Starting file receiver first:

     python file_receeiver.py grpc://0:1234 /tmp

Send the file using file sender:

     python file_sender.py grpc://192.168.1.2:1234 /home/user/big_file
