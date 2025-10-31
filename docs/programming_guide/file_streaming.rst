.. _file_streaming:

####################
FLARE File Streaming
####################

File streaming is a function that allows a file to be shared with one or more receivers. The file owner could be the FL Server or any FL Client. File streaming can be an effective alternative to sending large amounts of data with messages.

There are two main issues with sending large messages:
- A large memory space is required to serialize the message into bytes before sending it. Once memory is saturated, everything becomes very slow.
- A large byte array sent as a single message could saturate the network, slowing down overall processing.

File streaming, on the other hand, sends the big file with many small messages, each containing a chunk of file data. The big file is never loaded into memory completely. Since only small messages are sent over the network, it is less likely to bog down the network.

Push vs. Pull
=============

There are two ways to get the file sent from one place to another: push and pull.

- **Push**: The file owner sends the file to recipient(s). The push process is somewhat strict; if the file is sent to multiple recipients, all must process the same chunks simultaneously. If any one of them fails, the whole sending process fails. Hence, it is most useful when sending the file to a single recipient. The “push” method is implemented with the `FileStreamer` class.

- **Pull**: The file owner first prepares the file and gets the Reference ID (RID) for the file. It then sends the RID to all recipients in whatever way it wants (e.g., broadcast). Once the RID is received, each recipient pulls the file chunk by chunk until the whole file is received. Pulling is more relaxed as recipients are not synchronized. Each recipient can pull the file at its own pace, which is useful when sharing a file with multiple recipients. The “pull” method is implemented with the `FileDownloader` class.

File Life Cycle Management
==========================

Though file download (pull) is more robust for multiple recipients, there is the issue of file management. Ultimately, it’s the file owner’s responsibility to remove the file (if necessary) when it is no longer needed.

Since each recipient can download the file at its own pace, there is no definitive time that the file is no longer needed at the file owner's side. One effective way is activity timeout: if there has been no downloading activity from any recipient for a specified period, we can assume the file is no longer needed.

FileStreamer
============

Since `FileStreamer` (push) sends a file to recipients unannounced, the recipients must be set up in advance to process the received file. This is done by calling `FileStreamer.register_stream_processing`.

.. code-block:: python

   class FileStreamer(StreamerBase):
      @staticmethod
      def register_stream_processing(
          fl_ctx: FLContext,
          channel: str,
          topic: str,
          dest_dir: str = None,
          stream_done_cb=None,
          chunk_consumed_cb=None,
          **cb_kwargs,
      ):
          """Register for stream processing on the receiving side.

          Args:
              fl_ctx: the FLContext object
              channel: the app channel
              topic: the app topic
              dest_dir: the destination dir for received file. If not specified, system temp dir is used
              stream_done_cb: if specified, the callback to be called when the file is completely received
              chunk_consumed_cb: if specified, the callback to be called when a chunk is processed
              **cb_kwargs: the kwargs for the stream_done_cb

          Returns: None

          Notes: the stream_done_cb must follow stream_done_cb_signature as defined in apis.streaming.
          """

A channel and topic must be arranged between the sender and all receivers for them to share files. All recipients must call this method once for each channel and topic expected to receive files. Typically, this call is made at the beginning of the application in an event handler that handles the START_RUN event.

The `stream_done_cb` is called to notify the application when the file is completely received. It must follow the following signature:

.. code-block:: python

   def stream_done_cb_signature(stream_ctx: StreamContext, fl_ctx: FLContext, **kwargs):
      """This is the signature of stream_done_cb.

      Args:
          stream_ctx: context of the stream
          fl_ctx: FLContext object
          **kwargs: the kwargs specified when registering the stream_done_cb.

      Returns: None
      """

The `stream_ctx` contains information about the stream, including the information from the file owner when it calls `stream_file` to send the file.

The received data is saved in a temporary file. Use the following methods to get file information from the `stream_ctx`:

.. code-block:: python

   @staticmethod
   def get_file_name(stream_ctx: StreamContext):
      """Get the file base name property from stream context.
      This method is intended to be used by the stream_done_cb() function of the receiving side.

      Args:
          stream_ctx: the stream context

      Returns: file base name
      """

.. code-block:: python

   @staticmethod
   def get_file_location(stream_ctx: StreamContext):
      """Get the file location property from stream context.
      This method is intended to be used by the stream_done_cb() function of the receiving side.

      Args:
          stream_ctx: the stream context

      Returns: location (full file path) of the received file
      """

.. code-block:: python

   @staticmethod
   def get_file_size(stream_ctx: StreamContext):
      """Get the file size property from stream context.
      This method is intended to be used by the stream_done_cb() function of the receiving side.

      Args:
          stream_ctx: the stream context

      Returns: size (in bytes) of the received file
      """

Note that it’s your responsibility to decide what to do with the received file and whether/when to delete the file.

Sending File
============

The file owner sends a file to one or more recipients by calling the `stream_file` function, as defined in the `FileStreamer` module.

.. code-block:: python

   def stream_file(
      channel: str,
      topic: str,
      stream_ctx: StreamContext,
      targets: List[str],
      file_name: str,
      fl_ctx: FLContext,
      chunk_size=None,
      chunk_timeout=None,
      optional=False,
      secure=False,
   ) -> (str, bool):
      """Stream a file to one or more targets.

      Args:
          channel: the app channel
          topic: the app topic
          stream_ctx: context data of the stream
          targets: targets that the file will be sent to
          file_name: full path to the file to be streamed
          fl_ctx: a FLContext object
          chunk_size: size of each chunk to be streamed. If not specified, default to 1M bytes.
          chunk_timeout: timeout for each chunk of data sent to targets.
          optional: whether the file is optional
          secure: whether P2P security is required

      Returns: a tuple of (RC, Result):
          - RC is ReturnCode.OK or ReturnCode.ERROR;
          - Result is whether the streaming completed successfully

      Notes: this is a blocking call - only returns after the streaming is done.
      """

The arguments are self-explanatory. Note that you can send any additional information through the `stream_ctx`, which is a dict. The information will be available to the recipient’s registered `stream_done_cb`.

FileDownloader
==============

The file downloading process requires three steps:

1. The data owner prepares the file(s) to be shared with recipients and obtains one reference id (RID) for each file.
2. The data owner sends the RID(s) to all recipients. This is usually done with a broadcast message.
3. Recipients download the files one by one with received RIDs.

Download Preparation
--------------------

The data owner first prepares files to be shared with other recipients using the `FileDownloader`’s `new_transaction` and `add_file` methods, defined as follows:

.. code-block:: python

   class FileDownloader:

      @classmethod
      def new_transaction(
          cls,
          cell: Cell,
          timeout: float,
          timeout_cb,
          **cb_kwargs,
      ):
          """Create a new file download transaction.

          Args:
              cell: the cell for communication with recipients
              timeout: timeout for the transaction
              timeout_cb: CB to be called when the transaction is timed out
              **cb_kwargs: args to be passed to the CB

          Returns: transaction id

          The timeout_cb must follow this signature:

              cb(tx_id, file_names: List[str], **cb_args)
          """

.. code-block:: python

      @classmethod
      def add_file(
          cls,
          transaction_id: str,
          file_name: str,
          file_downloaded_cb=None,
          **cb_kwargs,
      ) -> str:
          """Add a file to be downloaded to the specified transaction.

          Args:
              transaction_id: ID of the transaction
              file_name: name of the file to be downloaded
              file_downloaded_cb: CB to be called when the file is done downloading
              **cb_kwargs: args to be passed to the CB

          Returns: reference id for the file.

          The file_downloaded_cb must follow this signature:

              cb(ref_id: str, to_site: str, status: str, file_name: str, **cb_kwargs)
          """

First, you call the `new_transaction` method to get a transaction id. A transaction can include one or more files to be downloaded. The arguments are self-explanatory. The cell is for messaging with the recipients. You can get it from a `FLContext` object as follows:

.. code-block:: python

   engine = fl_ctx.get_engine()
   cell = engine.get_cell()

The timeout specifies when the transaction should time out: it is the maximum time within which no downloading activity is received from any recipient for any file in the transaction! Due to the distributed nature of recipients, they can download the file(s) at their own pace - some are downloading one file while others are downloading another file. The transaction is considered timed out only if no recipient is downloading any file of the transaction for the specified amount of time. The registered `timeout_cb` will be called with all the file names of the transaction. You can then decide what to do with these files.

You call the `add_file` method for each file to be downloaded. You receive a file reference id (RID) for each file added. You then send the RIDs to all recipients with a message.

Download File
-------------

Once the recipient receives RID(s), it calls the function to download the referenced file from the data owner.

.. code-block:: python

   def download_file(
      from_fqcn: str,
      ref_id: str,
      per_request_timeout: float,
      cell: Cell,
      location: str = None,
      secure=False,
      optional=False,
      abort_signal=None,
   ) -> (str, Optional[str]):
      """Download the referenced file from the file owner.

      Args:
          from_fqcn: FQCN of the file owner.
          ref_id: reference ID of the file to be downloaded.
          per_request_timeout: timeout for requests sent to the file owner.
          cell: cell to be used for communicating to the file owner.
          location: dir for keeping the received file. If not specified, will use temp dir.
          secure: P2P private mode for communication
          optional: suppress log messages of communication
          abort_signal: signal for aborting download.

      Returns: tuple of (error message if any, full path of the downloaded file).
      """

The arguments are self-explanatory. If the downloading is successful, you will get the full path to the downloaded file. It’s up to you what to do with the file.


Large Object Serialization with File Streaming or Download
----------------------------------------------------------

please refer to :ref:`decomposer_for_large_object`

