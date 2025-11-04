.. _extra_270:

Extra Features in v2.7.0
==============================

Memory Management Improvements
------------------------------

There are two main issues with sending large messages:

- A large memory space is required to serialize the message into bytes before sending it. Once memory is saturated, everything becomes very slow.
- A large byte array sent as one single message could cause the network to be saturated, which could also slow down the overall processing.

These issues exist regardless of whether the model is sent directly or via streaming. We have developed a few different ways to address these issues.

Another issue with LLM streaming is that it is limited by memory size; the model size must fit into the memory. File-based streaming is not limited by the memory size.

We introduced FileStreamer in the previous release :ref:`file_streaming`. We are now introducing FileDownloader.

Push vs. Pull
^^^^^^^^^^^^^

There are two ways to get the file sent from one place to other places: push and pull.
With push, the file owner sends the file to recipient(s). The push process is somewhat strict in that if the file is
sent to multiple recipients, all recipients must process the same chunks at the same time. If any one of them fails,
then the whole sending process fails. Hence, in practice, it is most useful when sending the file to a single recipient.

The “push” method is implemented with the **FileStreamer** class ( Released in 2.6.0)

File Streaming
^^^^^^^^^^^^^^

File streaming is a function that allows a file to be shared with one or more receivers.
The file owner could be the FL Server or any FL Client. File streaming could be a very effective alternative to sending
large amounts of data with messages.

File streaming, on the other hand, sends the big file with many small messages,
each containing a chunk of file data. The big file is never loaded into memory completely.
Since only small messages are sent over the network, it is less likely to completely bog down the network.


With pull, the file owner first prepares the file and gets the Reference ID (RID) for the file. It then sends the RID to all recipients in whatever way it wants (e.g., broadcast). Once the RID is received, each recipient then pulls the file chunk by chunk until the whole file is received.

As you can see, pulling is much more relaxed in that recipients are not synchronized in any way.
Each recipient can pull the file at its own pace. This is very useful when sharing a file with multiple recipients.

The “pull” method is implemented with the **FileDownloader** class.


FileDownloader
^^^^^^^^^^^^^^
The file downloading process requires three steps:

1. The data owner prepares the file(s) to be shared with recipients, and obtains one reference ID (RID) for each file.
2. The data owner sends the RID(s) to all recipients. This is usually done with a broadcast message.
3. Recipients download the files one by one with received RIDs.


Pre-Install CLI Command
-----------------------

In cases where custom code or dynamic code is not allowed to be deployed, we need to pre-install the application to the
host. Although you can manually deploy this code without using any tool or command, the following pre-install tool
may provide a simpler method.

The code pre-installer handles:
- Installation of application code
- Installation of shared libraries
- Site-specific customizations
- Python package dependencies

The tool provides two main commands:
- `prepare`: Package application code for installation
- `install`: Install packaged code to target sites

:ref:`pre_installer`


New documentations
------------------
Along with the new features, we add a lot documentations related to the features, in addition, we added the following
new documentations:

    - :ref:`flare_system_architecture`
    - :ref:`flare_security_overview`
    - :ref:`cellnet_architecture`





