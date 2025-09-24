.. _hierarchical_communication:

#########################################
Hierarchical Communication and Clients
#########################################

In a basic FLARE deployment, clients are connected to the server directly. There is one dedicated connection between the server and each client. This works fine when the number of clients is relatively small (e.g. < 100). However, when the number of clients increases, the number of concurrent connections to the server increases accordingly, which makes the communication less efficient.

CellNet, the communication technology of FLARE, supports hierarchical communication topology. Cells are organized hierarchically to allow efficient connection management.

Communication Hierarchy
=======================

FLARE 2.7 makes use of this feature to manage FL clients efficiently, especially when the number of clients is very large (say > 1000).

Relay
-----

In a communication hierarchy, relays are nodes that are connected to the Server or to parent relays, as illustrated in this diagram:

The tree can be any level deep and a parent can have any number of child nodes. But since the purpose of this arrangement is for efficient connection management, the number of child nodes should be less than 100.

FQCN
----

In the communication hierarchy, each node is called a cell, and each cell has a unique name called a fully qualified cell name (FQCN). FQCN is simply its path from the server to it.

- The Server’s FQCN is simply “server”.
- The FQCNs of the server’s direct children are just their base names. In this example, the FQCNs of R1 and R2 are simply R1 and R2.
- The FQCN of R11 is R1.R11.  Similarly the R12’s FQCN is R1.R12.
- The FQCN of R21 is R2.R21.  Similarly the R22’s FQCN is R2.R22.

.. note::
   For simplicity, the root name of the hierarchy (i.e. server) is omitted from the FQCN. Otherwise every cell’s FQCN would have been started with “server”.

Cellnet guarantees that any cell can communicate with any other cells in the hierarchy.

Connect Clients to the Hierarchy
--------------------------------

Clients can be connected to any node in the hierarchy. They can still directly connect to the Server, or connect to intermediary or leaf relay nodes. You may prefer to connect clients only to the leaf relay nodes for simplicity, but this is not a requirement.

The following diagram shows a simple arrangement that 8 clients (C1 to C8) are connected to leaf nodes only.

As far as Cellnet is concerned, clients are nothing but cells, and each has its own FQCN. For example, C6’s FQCN is R2.R21.C6.

Here is another example where some clients are connected to intermediary nodes or the server.

No matter how clients are connected, they can still communicate with the Server or with other clients. This is completely transparent to application code.

Client Hierarchy
================

Even though clients may connect to the server or different relays in the communication hierarchy, they are equal in that they are all independent of each other.

FLARE 2.7 allows clients to also be organized hierarchically! This means that clients do not have to be independent of each other: some clients could be children of others. This would allow more efficient implementation of certain algorithms (e.g. hierarchical aggregation).

.. note::
   Client hierarchy and communication hierarchy are totally different. Communication hierarchy is for efficient connection management; whereas client hierarchy is for hierarchical algorithm implementation.

You can think of client hierarchy as a logical arrangement of client relationship, regardless of how they are connected in the communication hierarchy. In fact, child clients usually are not connected to their parent client.

The following diagram shows a client hierarchy.

FQSN
----

Each client in the client hierarchy has a unique name called a fully qualified site name (FQSN). FQSN specifies the path of the client from the server.

- In the above example, the FQSNs of C1 and C2 are C1 and C2 respectively. The FQSN of client C11 is C1.C11, and so on.

This client hierarchy can be implemented on any communication hierarchy.

This diagram shows how it is implemented with a relay hierarchy:

Or they can simply connect to the server directly:

Job Hierarchy
=============

When a job is deployed, job processes are created for each client and the server. These processes are called CJs (client job) and SJ (server job). There is one CJ for each client.

The relationship of job processes (CJs and SJ) follow the relationship between their corresponding clients. For example, since C11 is a child of C1, CJ on client C11 is also a child of the CJ on client C1.

Job hierarchy is important for the implementation of hierarchical algorithms, where results computed by child CJs are sent to the parent CJ for aggregation.

.. note::
   Even though client hierarchy and communication hierarchy are independent of each other, they share the same goal of optimizing overall system performance by reducing the pressure of central processing, which usually comes from communication and computation.

Without the client hierarchy, each client still sends its results to the server. Even though the communication hierarchy could reduce the number of connections to the server, the number of messages and amount of data to be processed by the server remain the same.

This is where client hierarchy can help. Since only the top-tier clients report to the server, client hierarchy reduces the amount of processing that the server has to do.

Hence the best way to implement client hierarchy is to follow the defined communication hierarchy, as illustrated in the first example. It minimizes the number of communication hops and amount of processing.

Provision
=========

The communication hierarchy and client hierarchy are created through the Provision process.

Relay
-----

A relay node connects to its parent or to the server. At the same time, it has to accept connections from other nodes. Hence a relay node must be both a listener (as a communication server) and a connector (as a communication client). As a result, the Provision process will create both a server credential (cert and private key) and a client credential (cert/key) for the relay node, and include them in the relay’s startup kit.

These are specified by the following properties:

listening_host
--------------

This property specifies where the relay will be running and the port number that it will open and listen to.

This property can have up to 5 elements.

- **scheme**: the communication protocol (http, grpc, or tcp). If not specified, use the overall scheme of the project.
- **host_names**: additional host names or IP addresses that this host will be known as. All the specified names will be included in the “Subject Alternative Names” field of the server certificate. This element is optional.
- **default_host**: the default host name to be used for connecting to the host. Must be specified.
- **port**: the port number to listen. Must be specified.
- **connection_security**: the connection security for incoming connections (tls, mtls, or clear). If not specified, use the project’s default connection security. If the project’s connection security is not explicitly specified, the default value is “mtls” (mutual TLS).

connect_to
----------

This property specifies the information necessary for the relay to make a connection.

This property can have up to 4 elements.

- **name**: the base name of the node in the hierarchy. Note that each node has a unique base name. If this is specified, then the relay will connect to the specified node at the default_host of the node.
- **host**: the host name or IP address to connect to. This should be accessible from the intended node (either its default host or in its host_names), unless BYOConn is used.
- **port**: the port number to connect to. This element is usually not needed, unless BYOConn is used.
- **connection_security**: the connection security for outgoing connections (tls, mtls, or clear). It usually does not need to be specified explicitly, unless BYOConn is used.

.. note::
   The name or host element must be specified, but not both.

A Note about BYOConn
---------------------

FLARE supports a feature called Bring Your Own Connectivity (BYOConn). With BYOConn, a listening endpoint could be protected by an ingress proxy. To connect to the endpoint, the “connect_to” property must be pointing to the ingress proxy.

Client Hierarchy
----------------

Clients are connected to the Server or a relay node. To connect to the Server, no extra configuration is required. To connect to a relay, use the “connect_to” property as described above.

Another aspect of the client hierarchy is the client’s position in the hierarchy. This is specified with the “parent” property. The value of this property is the base name of the parent client.

Example
-------

The following project.yml shows how to use these properties to specify the communication hierarchy and client hierarchy of the example in the FQSN discussion.

.. code-block:: yaml

   api_version: 3
   name: mobile
   description: NVIDIA FLARE sample project yaml file
   connection_security: clear
   allow_error_sending: false

   participants:
    - name: server
      type: server
      org: nvidia
      fed_learn_port: 8002
      host_names: [localhost, 127.0.0.1]
      default_host: localhost
    - name: R1
      type: relay
      org: nvidia
      listening_host:
        default_host: localhost
        port: 18004
    - name: R2
      type: relay
      org: nvidia
      listening_host:
        port: 28004
        default_host: localhost
    - name: C1
      type: client
      org: nvidia
      connect_to:
        name: R1
    - name: C11
      type: client
      org: nvidia
      parent: C1
      connect_to:
        name: R1
    - name: C12
      type: client
      org: nvidia
      parent: C1
      connect_to:
        name: R1
    - name: C2
      type: client
      org: nvidia
      connect_to:
        name: R2
    - name: C21
      type: client
      org: nvidia
      parent: C2
      connect_to:
        name: R2
    - name: C22
      type: client
      org: nvidia
      parent: C2
      connect_to:
        name: R2
    - name: admin@nvidia.com
      type: admin
      org: nvidia
      role: project_admin
      connect_to: 127.0.0.1

   builders:
    - path: nvflare.lighter.impl.workspace.WorkspaceBuilder
    - path: nvflare.lighter.impl.static_file.StaticFileBuilder
      args:
        config_folder: config

        # scheme for communication driver (grpc, tcp, http).
        scheme: grpc

    - path: nvflare.lighter.impl.cert.CertBuilder
    - path: nvflare.lighter.impl.signature.SignatureBuilder


