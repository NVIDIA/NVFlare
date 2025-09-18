#################################
Reliable Federated XGBoost Design
#################################


*************************
Flare as XGBoost Launcher
*************************

NVFLARE serves as a launchpad to start the XGBoost system.
Once started, the XGBoost system runs independently of FLARE,
as illustrated in the following figure.

.. figure:: ../../resources/loose_xgb.png
    :height: 500px

There are a few potential problems with this approach:

 - As we know, MPI requires a perfect communication network,
   whereas the simple gRPC over the internet could be unstable.

 - For each job, the XGBoost Server must open a port for clients to connect to.
   This adds burden to request IT for the additional port in the real-world situation.
   Even if a fixed port is allowed to open, and we reuse that port,
   multiple XGBoost jobs cannot be run simultaneously;
   since each XGBoost job requires a different port number.


*****************************
Flare as XGBoost Communicator
*****************************

FLARE provides a highly flexible, scalable, and reliable communication mechanism.
We enhance the reliability of federated XGBoost by using FLARE as the communicator of XGBoost,
as shown here:

.. figure:: ../../resources/tight_xgb.png
    :height: 500px

Detailed Design
===============

The open-source Federated XGBoost (c++) uses gRPC as the communication protocol.
To use FLARE  as the communicator, we simply route XGBoost's gRPC messages through FLARE.
To do so, we change the server endpoint of each XGBoost client to a local gRPC server
(LGS) within the FLARE client.

.. figure:: ../../resources/fed_xgb_detail.png
    :height: 500px

As shown in this diagram, there is a local GRPC server (LGS) for each site
that serves as the server endpoint for the XGBoost client on the site.
Similarly, there is a local GRPC Client (LGC) on the FL Server that
interacts with the XGBoost Server. The message path between the XGBoost Client and
the XGBoost Server is as follows:

  1. The XGBoost client generates a gRPC message and sends it to the LGS in the FLARE client.
  2. The FLARE client forwards the message to the FLARE server. This is a reliable FLARE message.
  3. The FLARE server uses the LGC to send the message to the XGBoost server.
  4. The XGBoost server sends the response back to the LGC in the FLARE server.
  5. The FLARE server sends the response back to the FLARE client.
  6. The FLARE client sends the response back to the XGBoost client via the LGS.

Please note that the XGBoost Client (c++) component could be running as a separate process
or within the same process of FLARE Client.
