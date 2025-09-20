.. _flare_hierarchical_architecture

Hierarchical FLARE
==================

As discussed in this document :ref:`hierarchical_communication`, FLARE can scale up to support a large number of FL clients with communication hierarchy and client hierarchy. When properly configured, FLARE can achieve very efficient communication and hierarchical aggregation computation.

This diagram shows a hierarchical FLARE system that uses two levels of relays (the R nodes) for communication. A client hierarchy (CP nodes) is also defined and connected to the relays.

Note that the client hierarchy follows the relay hierarchy closely for the best performance. For example, CP1_1 and CP1_2 are children of CP1; the LCPs connected to R1_1 are children of CP1_1; the LCPs connected to R1_2 are children of CP1_2; and so on.

This client hierarchy is used to implement hierarchical aggregation algorithms for device training.

Leaf Client Process (LCP)
=========================

LCPs are the terminal nodes in the client hierarchy. They serve a special role in supporting edge applications.

Interface for Edge Communication
--------------------------------

First, LCPs serve as the entry point for messages coming from edge devices. However, these messages are not sent to LCPs directly. Instead, they pass through an intermediary component called a web node.

What are web nodes?
-------------------

Web nodes are routing components that receive messages from edge devices and forward them to the appropriate LCPs. They help manage load distribution and ensure that device messages are consistently routed.

Multiple web nodes could be deployed in different regions based on the estimated number of edge devices and their geological distribution. A web node can connect to all or a subset of LCPs, based on the routing configuration.

Web nodes typically aim to:

1. Route messages from the same device to the same LCP based on its device ID
2. Distribute devices evenly across all connected LCPs.

.. note::
   For privacy protection, the “device ID” only needs to be a globally unique number and does not need to be the real ID of the device. Once generated, this ID must stay constant at least for the duration of a training session.

The following diagram shows this architecture.

Routing to Client Job Processes (CJs)
-------------------------------------

Second, LCPs route received edge messages to the right client job processes (CJs), which implement application processing logic.

Note that unlike CJs that come and go, LCPs are permanent. There could be multiple jobs running at the same time and there is one CJ attached to the LCP for each running job, the LCP finds the right CJ for the received edge message.

The following diagram shows the system with a job deployed.

CJ Hierarchy
============

Once a job is deployed, there will be one SJ (Server Job) process and one dedicated CJ process for the job on each CP. The CJ hierarchy follows the hierarchy of their CPs. Device messages are received and processed by the CJs associated with LCPs.

The following diagram shows the CJ hierarchy corresponding to the example above.

Leaf CJs are associated with LCPs. They interact with the edge devices indirectly following the Edge Device Interaction Protocol (EDIP). They also serve as the first line aggregator that aggregates training results from their devices, and reports the aggregation result to their parent CJs. All intermediary CJs aggregate results from their children and report aggregation results to their parents, all the way to the SJ, which generates the final aggregation result.

Routing Proxy (Web Node Implementation)
=======================================

In the current implementation, the web node is realized as a component called the Routing Proxy.

Routing Logic
-------------

The Routing Proxy uses a hash-based routing strategy based on each device's unique identifier:

- The device ID is passed in the request.
- A consistent hash function maps the device ID to a specific LCP.
- This ensures that all messages from the same device are routed to the same LCP, which is important for session consistency.
- It also ensures even distribution of devices across available LCPs.

Provision
=========

The communication hierarchy and the client hierarchy discussed must be created properly with the Provision tool. This could be done with the listening_host, connect_to and FQSN properties, as discussed in this document, but it could be very tedious and error-prone to do it correctly, especially when the number of nodes is large.

Flare 2.7 offers a CLI tool called tree_prov to make it easier. With this tool, you only need to specify the shape of the communication hierarchy, and the tool will do the rest for you (i.e. creating client hierarchy following the topology of the communication hierarchy).

.. note::
   However, please note that this tool is only for simple prototyping on a single machine: all nodes are assumed to be on the local host. Tools for the production environment will come in future versions of Flare.

To run tree_prov:

.. code-block:: bash

   python -m nvflare.lighter.tree_prov options

Here is the list of options:

- `--root_dir, -r`: the directory for the provision result. Required.
- `--project_name, -p`: project name. Required.
- `--depth, -d`: depth of the relay tree that is the number of relay tiers. Required.
- `--width, -w`: width of the tree that is the number of child relay nodes for each parent relay. Note that this only applies to relay nodes. If not specified, default to 2.
- `--clients, -c`: number of clients (LCPs) for each leaf relay node. This only applies to the leaf relay nodes.
- `--max_sites, -m`: the max number of sites, including relays and FL clients. Note that the number of sites goes up exponentially when the depth goes up, this limit prevents the tool from generating too many sites when the user accidentally enters a large depth value. The default value of max_sites is 100.
- `--lcp_only, -l`: only generate provision results for LCPs. This is occasionally useful when new LCPs are added after the project is already provisioned.
- `--analyze, -a`: if specified, only perform analysis of the topology and do not generate provision results. The analysis shows the number of relay and client nodes in the hierarchy.
- `--rp`: the port number of the Routing Proxy, which implements the web nodes.

Here is an example of topology analysis:

.. code-block:: bash

   python -m nvflare.lighter.tree_prov -d 2 -w 2 -a -c 3 -r . -p x

The result is:

- Relays:  leaf=4; non-leaf=2; total=6
- Clients: leaf=12; non-leaf=6; total=18
- Total Sites: 25

There are 6 relay nodes in total: 2 non-leaf nodes, and 4 leaf nodes since each non-leaf node has 2 leaf nodes for width value of 2.

There are 18 client nodes in total. In the client hierarchy, there are 6 non-leaf clients, one for each relay node. There are 12 leaf clients, 3 for each leaf relay node.

The total number of sites is the sum of the total number of relays (6) and the total number of clients (18), plus 1 (the server). This gives 25.

In addition to the provision results, the tree_prov tool generates additional files for deployment of web nodes and some convenience scripts. These files are placed into the “scripts” folder of the provisioned result. Among these files, the following are important:

- `lcp_map.json`: this file contains port numbers that will be used by the web nodes to connect to LCPs.
- `start_rp.sh`: this shell script is used to start a web node (routing proxy)
- `rootCA.pem`: this file contains the root cert of the project. It is used by the web node to make secure connections to LCPs.

