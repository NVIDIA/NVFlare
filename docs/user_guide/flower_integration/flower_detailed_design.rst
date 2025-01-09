***************
Detailed Design
***************

Flower uses gRPC as the communication protocol. To use FLARE  as the communicator, we route Flower's gRPC
messages through FLARE. To do so, we change the server-endpoint of each Flower client to a local gRPC
server (LGS) within the FLARE client.

.. image:: ../../resources/FLARE_as_flower_communicator.png

As shown in this diagram, there is a Local GRPC server (LGS) for each site that serves as the
server-endpoint for the Flower client on the site. Similarly, there is a Local GRPC Client (LGC) on the
FLARE Server that interacts with the Flower Server. The message path between the Flower Client and the Flower
Server is as follows:

   - The Flower client generates a gRPC message and sends it to the LGS in the FLARE Client
   - FLARE Client forwards the message to the FLARE Server. This is a reliable FLARE message.
   - FLARE Server uses the LGC to send the message to the Flower Server.
   - Flower Server sends the response back to the LGC in the FLARE Server.
   - FLARE Server sends the response back to the FLARE Client.
   - FLARE Client sends the response back to the Flower Client via the LGS.

Please note that the Flower Client could be running as a separate process or within the same process as the FLARE Client.

This will enable users to directly deploy Flower ServerApps and ClientsApps developed within the
NVFlare Runtime Environment. No code changes are necessary!
