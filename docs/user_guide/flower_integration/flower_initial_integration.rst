*******************
Initial Integration
*******************

Architecturally, Flower uses client/server communication. Clients communicate with the server
via gRPC. FLARE uses the same architecture with the enhancement that multiple jobs can run at
the same time (each job requires one set of clients/server) without requiring multiple ports to
be open on the server host.

Since both frameworks follow the same communication architecture, it is fairly easy to make a
Flower application a FLARE job by using FLARE as the communicator for the Flower app, as shown below.

.. image:: ../../resources/FLARE_as_flower_communicator.png

In this approach, Flower Clients no longer directly interact with the Flower Server, instead all
communications are through FLARE.

The integration with FLARE-based communication has some unique benefits:

   - Provisioning of startup kits, including certificates
   - Deployment of custom code (apps)
   - User authentication and authorization
   - :class:`ReliableMessage<nvflare.apis.utils.reliable_message.ReliableMessage>` mechanism to counter connection stability issues
   - Multiple communication schemes (gRPC, HTTP, TCP, Redis, etc.) are available
   - P2P communication: anyone can talk to anyone else without needing topology changes
   - Support of P2P communication encryption (on top of SSL)
   - Multi-job system that allows multiple Flower apps to run at the same time without needing extra ports on the server host
   - Use additional NVFlare features like experiment tracking
