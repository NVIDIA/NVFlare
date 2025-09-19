.. _communication_configuration:

###########################
Communication Configuration
###########################

FLARE's communication system is based on the CellNet technology.
CellNet supports logical communication. Each site in the system is called a communication cell, or simply a cell.
All cells form a communication network called CellNet and each cell has a unique ID called Fully Qualified Cell Name (FQCN).
Any cell can communicate with any other cells via their FQCNs, regardless how the messages are routed. 

FLARE is a multi-job system in that multiple jobs can be executed at the same time.
When a FLARE system is started, the CellNet consists of the server and one client cell for each site.
All client cells are connected to the server cell. This topology is the backbone of the communication system and cells are called Parent Cells.

When a job is deployed, the job is done by new cells dedicated to the execution of the job, one cell at each site (server and clients).
These cells are called Job Cells which are started when the job is deployed, and stopped when the job is finished.

This communication system provides many powerful features (multiple choices of communication drivers, large message streaming, ad-hoc direct connections, etc.).
However, for these features to work well, they need to be configured properly.

This document describes all aspects that can be configured and how to configure them properly.

The following aspects of the communication system can be configured:

- Parameters of communication drivers
- Selection of gRPC driver implementation (asyncio vs. non-asyncio)
- Configuration of ad-hoc connections
- Configuration of internal connections
- Messaging parameters

General Configuration
=====================

The communication system is configured with the comm_config.json file. This file is to be maintained by Operation Staff of each FL site (servers and FL clients).
This file must be placed in the site's "local" folder:

``<site_workspace>/local/comm_config.json``

Some aspects of the communication system are configured with simple variables (e.g. max_message_size).
Variables can be defined in comm_config.json or via OS system environment variables.

To define a variable in comm_config.json, simply set it as the first-level element:

.. code-block:: json

  {
    "max_message_size": 2000000000
  }

You can also define the variable using an OS environment variable. The name of the env var the var name converted into uppercase and prefixed with ``NVFLARE_``.
For example, the env var name for max_message_size is: ``NVFLARE_MAX_MESSAGE_SIZE``.

If you define the same variable both in the file and as an environment variable, the value defined in the file takes precedence.

The following is an example of the comm_config.json:

.. code-block:: json

  {
    "allow_adhoc_conns": false,
    "backbone_conn_gen": 2,
    "max_message_size": 2000000000,
    "internal": {
      "scheme": "tcp",
      "resources": {
        "host": "localhost"
      }
    },
    "adhoc": {
      "scheme": "tcp",
      "resources": {
        "host": "localhost",
        "secure": false
      }
    },
    "grpc": {
      "options": [
        [
          "grpc.max_send_message_length", 1073741824
        ],
        [
          "grpc.max_receive_message_length", 1073741824
        ],
        [
          "grpc.keepalive_time_ms", 120000
        ],
        [
          "grpc.http2.max_pings_without_data", 0
        ]
      ]
    }
  }


Configuration of Communication Drivers
======================================

A communication driver is identified by its scheme (tcp, http, grpc, etc.).
The details of the driver can be configured with a section named with the scheme in the config file. In the example above, the "grpc" section defines the gRPC driver's options.

Note that different drivers have different configuration parameters.

GRPC Configuration
------------------

The GRPC driver's details are defined in the "options" section within the "grpc" section. Please see GRPC documentation for details of available options.

Note that since FLARE has built general messaging management for all drivers, you shouldn't need to configure GRPC options in most cases.

GRPC Driver Selection
---------------------

GRPC is the default scheme for communication between FL clients and the server.
FLARE provides two implementations of GRPC drivers, one uses GRPC's asyncio version (AIO), another uses GRPC's non-asyncio version (non-AIO).
The default driver is the non-AIO version.

According to GRPC documentation, the AIO GRPC is slightly more efficient.
But the main advantage is that it can handle many more simultaneous connections on the server side, and there is no need to configure the "num_workers" parameter.

Unfortunately the AIO GRPC client-side library is not stable under difficult network conditions where disconnects happen frequently.
The non-AIO GRPC library seems very stable.

If your network is stable and you have many clients and/or many concurrent jobs, you should consider using the AIO version of the GRPC driver.
This is done by setting use_aio_grpc to true:

``"use_aio_grpc": true``

On the server side if you use the non-AIO gRPC driver, the default maximum number of workers is 100, meaning that there can be at most 100 concurrent connections to the server.
If this is not enough, you will need to use the AIO gRPC driver.

Ad-hoc Connections
==================

By default, all sites only connect to the server. When a site needs to talk to another site, messages will be relayed through the server.
To improve communication speed, it could be configured to allow the two sites to communicate directly, if network policies of the sites permit.
A direct connection between two sites (cells) is called an ad-hoc connection.

First of all, the ad-hoc connection must be enabled. This is done by setting the allow_adhoc_conns variable to true (default value is false).

``"allow_adhoc_conns": true``

Secondly, in the "adhoc" section, you can further specify what scheme to use for ad-hoc connections, as well as resources for establishing the connections.

.. code-block:: json

  "adhoc": {
    "scheme": "tcp",
    "resources": {
      "host": "localhost",
      "secure": false,
      "ports": [8008, 9008]
    }
  }

In this example, we use tcp for ad-hoc connections, and we will listen on port number 8008 or 9008.
Note that the ad-hoc connection's port number is dynamically determined based on the port information in the config.

Config Properties
-----------------

Scheme
^^^^^^

You specify the communication driver with the "scheme" property. Available schemes are grpc, http and tcp.

If not specified, the default scheme is "tcp".

Host
^^^^

You specify the host of the connection with the "host" property. This value is part of the URL for the connector to connect to.

Secure
^^^^^^

The "secure" property to specifies whether the ad-hoc connections will use SSL.

Note that if secure is set to true for a site, then the site must have a "server certificate", even if the site is a FL Client.
The site's "server certificate" is generated during the provision process, if you configure the "listening_host" property for the site in project.yml.

In secure communication mode, this host name must match the Common Name of the site's "server certificate", which is the same as the "listening_host" property for the site in project.yml.

The default value of "secure" is false.

Port Numbers
^^^^^^^^^^^^

You can specify port numbers to be used for connecting to the host. If not specified, an available port number will be dynamically assigned at the time the ad-hoc listener is created.

To specify a single port number using the "port" property:

``"port": 8008``
	
To specify a list of port numbers using the "ports" property:

``"ports": [8008, 8009, 8010]``

To specify a list of port number ranges using the "ports" property. The following example specifies two ranges of port numbers, one from 8008 to 9008, another from 18000 to 19000.

``"ports": [8008-9008, 18000-19000]``


Internal Connections
====================

As described earlier, job cells are started when a job is deployed. There is one job cell at each site (server and FL clients).
Job cells at one site are connected to the Parent cell of the same site. Such job-cell/parent-cell connections are called internal connections, since they are internal within the same site.

By default, internal connections use tcp drivers on dynamically determined port numbers.
Since internal connections are used between processes running on the same host, they don't require SSL.

If this default setup does not work for you, you can configure it to your liking in the "internal" section. For example:

.. code-block:: json

  "internal": {
    "scheme": "grpc",
    "resources": {
      "host": "localhost",
      "secure": false,
      "ports": [8008, 9008]
    }
  }

In this example, we changed to use "grpc" as the communication scheme.

The syntax and meanings of the properties are exactly the same as the "adhoc" configurations.

Messaging Parameters
====================

FLARE's messaging functions should work well with default configuration settings. However you may find it necessary to tune some parameters under some circumstances.
This section describes all parameters that you can configure.
                                                                   
The messaging parameters can be specified in <site_workspace>/local/comm_config.json file as first-level elements, or by using environment variables as described in the beginning of this document.

This is an example of comm_config.json file with default values for all the parameters,

.. code-block:: json

  {
    "comm_driver_path": "",
    "heartbeat_interval": 60,
    "streaming_chunk_size": 1048576,
    "streaming_read_timeout": 60,
    "streaming_max_out_seq_chunks": 16,
    "streaming_window_size": 16777216,
    "streaming_ack_interval": 4194304,
    "streaming_ack_wait": 10
  }

When large amount of data are exchanged on busy hosts like in LLM training, following parameters are recommended in <site_workspace>/local/comm_config.json on both servers and clients,

.. code-block:: json

  {
    "streaming_read_timeout": 3000,
    "streaming_ack_wait": 6000
  }

The communication_timeout parameter should be adjusted as following on clients in <site_workspace>/local/resources.json,

.. code-block:: json

  {
    "format_version": 2,
    "client": {
      "communication_timeout": 6000
    },
  }

Here are the detailed description of each messaging parameter,

comm_driver_path
----------------

FLARE supports custom communication drivers. The paths to search for the drivers need to be configured using parameter "comm_driver_path".
The parameter should be a list separated by colon. For example,

``"comm_driver_path": "/opt/drivers:/home/nvflare/drivers"``

heartbeat_interval
------------------

To keep the connection alive, FLARE exchanges a short message (PING/PONG) for each connection if no traffic is detected for a period of time.
This is controlled through the parameter "heartbeat_interval". The unit is seconds and the default value is 60.

``"heartbeat_interval": 30``

This parameter needs to be changed if the network closes idle connection too aggressively.

FLARE supports streaming of large messages. With streaming, the message is sliced into chunks and each chunk is sent as an individual message.
On the receiving end, the chunks are combined into the original large message. The following parameters control the general streaming behavior,

streaming_chunk_size
--------------------

The chunk size in bytes. The default value is 1M. When deciding chunk size the following factors must be considered:
- Each chunk is sent with headers so there is some overhead (around 50 bytes) so try to avoid small chunks (< 1K).
- The relaying server has to buffer the whole chunk so the memory usage will be higher with bigger chunks.

streaming_read_timeout
----------------------

The receiver of streaming times out after this value while waiting for the next chunk. The unit is seconds and the default is 60. 

This timeout is used to detect dead senders. On a very slow network or extremely busy host, this value may need to be increased.

streaming_max_out_seq_chunks
----------------------------

The chunks may arrive on the receiving end out of sequence. 
The receiver keeps out-of-sequence chunks in a reassembly buffer while waiting for the expected chunk to arrive.
The streaming terminates with error if the number of chunks in the reassembly buffer is larger than this value. The default is 16. 

The streaming implements a sliding-window protocol for flow-control. The receiver sends ACKs after the chunks are retrieved by the reader.
The window is all the chunks sent but not being acknowledged by the receiver. Once the window reaches a certain size, the sender pauses and waits for more ACKs.
Following parameters are used to control the flow-control behavior.

streaming_window_size
---------------------

The sliding window size in bytes. The default is 16M. 

The larger the window size, the smoother the flow of data  but the memory usage will be higher.

streaming_ack_interval
----------------------

This parameter controls how often the receiver sends ACKs to the sender.
he unit is bytes and the default value is 4M (1/4 of the window size).

The smaller the value, the smoother the sliding window moves, however it generates more messages.

streaming_ack_wait
------------------

The number of seconds that the sender waits for the next ACK.
The default value is 10 seconds. 

This timeout is used to detect dead receivers. On a very slow network, this value may need to be increased.
