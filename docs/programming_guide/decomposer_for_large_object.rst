.. _decomposer_for_large_object:

############################
Decomposer for Large Objects
############################

The payload of a message can be any simple or complex object. However, before the message can be sent, the object must be serialized to bytes. Similarly, on the receiving side, the received bytes are deserialized back to the correct object type for the application to process.

These are done with the FOBS (Flare OBject Serialization) system.

FOBS Introduction
=================

The object to be sent could be arbitrarily complex - composed of nested sub-objects of any type. FOBS is implemented with the open-source msgpack, which traverses the object tree recursively to turn each object into primitive types that it knows how to handle (e.g., bytes, str, numbers, list, dict, etc.). The process of turning an object into primitive types is called decomposition. The reverse process of turning primitive types into the original object type is called recomposition.

The object that does decomposition and recomposition for a target object type is called a decomposer. Flare provides decomposer implementations for many commonly used object types (all built-in Flare structures like DXO, Tensor, Numpy Array, Shareable, etc.).

Applications can build additional decomposers for any app-specific object types.

.. note::
    you can refer to :ref:`serialization` for more details on FOBS.

Serialization
=============

The general serialization process is like this:
- **Decomposition**: the object is traversed to turn each sub-object into primitive types. Appropriate decomposers are invoked to process objects unknown to msgpack.
- **Post Processing**: during the decomposition process, some decomposers may register post-process callbacks. Such callbacks, if any, are invoked one by one. A post-processing callback can register additional post-processing callbacks! The post-processing is finished after all such callbacks are called. A callback could add additional sections (called Datums) to the final message.
- **Message Assembly**: put all data sections (datums) together to form one message.

Externalization and Datums
==========================

The msgpack has a limitation of 4G for its serialization size. This can easily be exceeded if the object contains a large model. FOBS introduces a mechanism called externalization that moves large data (bytes or text) out of the msgpack serialization.

Firstly, the large data is moved into an object called datum, which has a unique ID and keeps the large data.

Secondly, the large data is replaced with a reference to the datum. Hence only the datum ref is included in the msgpack generated message body. Since externalization removes large data from msgpack processing (which generates the main body of the message), it assures that msgpack’s 4G limit won’t be exceeded.

Note: the final assembled message size can be arbitrarily large, even though the main body cannot exceed the 4G limit!

FOBS performs the externalization after invoking a decomposer.

Post Processing and DOT
=======================

If post-process callbacks are registered during the decomposition process (by decomposers), such callbacks are called. New datums could be added by the callbacks. Such datums usually need to be processed with special logic during deserialization. To facilitate this, such special datums need to be tagged with a Datum Object Type (DOT), and decomposers must be available to process each DOT.

DOT values must be globally unique.

Deserialization
===============

On the receiving side, received bytes are deserialized back to original object types:
- **Message dissection**: the bytes are parsed to extract the main body and datums (if any).
- **Pre-processing**: special datums that have non-zero DOT values are processed by registered decomposers. If no decomposer can be found for the datum, a RuntimeError exception will be raised.
- **Recomposition**: the main body is processed by the msgpack, which calls the “recompose” methods of appropriate decomposers. This will result in the final object of the original type.

Internalization
===============

If a sub-object’s recomposition results in a “datum ref”, it will be internalized by looking for the datum based on the unique datum ref ID. The large data held by the datum will replace the datum ref as the new value of the recomposed sub-object.

Issues with Large Objects
=========================

Though FOBS can handle objects of any size, a large object will be serialized to a message of large size. For very large objects, this could cause issues to the overall application process:

- The biggest concern is CPU memory usage. Large objects (e.g., large language models) already take a lot of memory, serializing them into bytes will take even more memory space.
- Network bandwidth could be saturated when trying to send such large messages.

The problem is multiplied when the large object needs to be sent to multiple recipients in the same period of time. This is the case in a typical FedAvg training, where multiple clients retrieve the same model at almost the same time. When memory space is saturated, the application won’t be able to perform normally.

File Based Decomposer
=====================

To address the issues with large objects, we developed a mechanism to reduce the message size by leveraging the FileDownloader. The general idea is that instead of serializing the large objects into a large number of bytes in memory, we write them into a file, and then make recipients download the file and reconstruct the objects.

Here are the details of this idea:
- Objects of target object types (e.g., Tensor, NP Array) contained in one message are collected into one single dict, and written to a file using the target object type specific file creation like savetensors.save_file, or np.save.
- Objects in the payload are replaced with simple references to the objects written to file. Hence the serialized message size is hugely reduced.
- The message is sent to recipients. Note that the message only contains references to the large objects, not the objects themselves.
- Each recipient downloads the referenced file from the sender, using FileDownloader. The recipient then loads the downloaded file back to original objects, using target object type specific file loading function.
- Once objects are recovered, the payload is recomposed and the object refs are replaced with the recovered objects.

With this approach, since the file is downloaded with small chunks, the memory space needed is small (no need to hold and transfer the huge objects via memory).

Message Root
============

When an object needs to be sent to multiple recipients, the object will go through the serialization process multiple times. Even if the resultant message is smaller, the serialization process itself could be time-consuming, since it has to generate a large file. The idea of message root is introduced to make the additional serialization more efficient.

Since all messages are for the same target object, this object is called the message root, and is assigned a unique UUID called message root ID.

When this object needs to be sent to multiple recipients, only the very first message (called the primary message) needs to go through the heavy process of file generation. All other messages are called secondary messages.

While the primary message goes through the serialization process, it generates file(s) and saves object reference information in a cache called the FOBSCache under the key of message root ID. All secondary messages will wait until the primary message is done.

When a secondary message goes through the serialization process, it no longer needs to create any files. Instead, it simply looks up the FOBSCache with the message root ID to find references for the objects.

Managing Generated Files
========================

As described above, files are generated during the serialization process. It’s not desirable to keep these large files on the file system for a long time.

Message root object is needed only for a limited amount of time. Typically after the object has been sent to all recipients, or when there is no need to wait for other recipients, the message root object is no longer needed. In this case, the message root is deleted, which will then cause all the temporary files associated with the message root ID to be deleted.

The following places in Flare system have been updated to use the message root mechanism:
- Task-based interactions (wf_comm_server, wf_comm_client, task_controller)
- Reliable Message, which may resend the message multiple times
- Task Exchanger, which sends message to client API
- Pipe Handler, which may resend the message multiple times

Another protection is that the generated files will be deleted after the download transaction is timed out (see FileDownloader for detail), regardless of whether the message root ID is deleted or not.

Finally, when a Flare Job is finished, the FileDownloader’s shutdown() method is always called, causing all files associated with all pending transactions to be deleted.

The three approaches described above assure that temporary files generated by the serialization process will eventually be deleted from the file system.

Developing a File Based Decomposer
==================================

If you need to send an object type that can potentially be very large, you should develop a file-based decomposer for this object type. You do this by extending ViaFileDecomposer.

.. code-block:: python

   class ViaFileDecomposer(fobs.Decomposer, ABC):
      @abstractmethod
      def dump_to_file(self, items: dict, path: str) -> Optional[str]:
         """Dump the items to the file with the specified path

         Args:
             items: a dict of items of target object type to be dumped to file
             path: the path to the file.

         Returns: if a new file name is used, return it; otherwise returns None.

         The "path" is a temporary file name. You should create the file with the specified name.
         However, some frameworks (e.g., numpy) may add a special suffix to the name. In this case, you must return the
         modified name.

         The "items" is a dict of target objects. The dict contains all objects of the target type in one payload.
         The dict could be very big. You must create a file to contain all the objects.
         """
         pass

      @abstractmethod
      def load_from_file(self, path: str) -> dict:
         """Load target object items from the specified file

         Args:
             path: the absolute path to the file to be loaded.

         Returns: a dict of target objects.

         You must not delete the file after loading. Management of the file is done by the ViaFile class.
         """
         pass

      @abstractmethod
      def get_file_dot(self) -> int:
         """Get the Datum Object Type to be used for file ref datum

         Returns: the DOT for file ref datum
         """
         pass

      @abstractmethod
      def get_bytes_dot(self) -> int:
         """Get the Datum Object Type to be used for bytes datum

         Returns: the DOT for bytes datum
         """
         pass

All you need to do is to provide the four methods required by this base class. The methods are self-explanatory. The only thing is that the DOT values are in the range of 1 to 127 and must be globally unique. If your decomposer is part of Flare’s core, it should register its DOT values in `nvflare.fuel.utils.fobs.dots.py`; otherwise, make sure its DOT values do not conflict with values defined there. Currently, only 4 DOT values are defined:
- NUMPY_BYTES = 1
- NUMPY_FILE = 2
- TENSOR_BYTES = 3
- TENSOR_FILE = 4
```
