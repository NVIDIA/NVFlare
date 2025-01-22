.. _communication_security:

Communication Security
======================

All data communications are through secure channels established with mutually-authenticated TLS connections. The
communication protocol between the FL Server and clients is gRPC. The protocol between FLARE Console instances and the
FL Server is TCP.
 
NVIDIA FLARE uses client-server communication architecture.  The FL Server accepts connection requests from clients.
Clients never need to accept connection requests from anywhere.
 
The IT infrastructure of the FL Server site must allow two ports to be opened: one for the FL Server to communicate with
FL Clients, and one for the FL Server to communicate with FLARE Console instances. Both ports should be unprivileged.
Specifically, we suggest against the use of port 443, the typical port number for HTTPS. This is because gRPC does
not exactly implement HTTPS to the letter, and the firewall of some sites may decide to block it.

The IT infrastructure of FL Client sites must allow the FL application to connect to the address (domain and port)
opened by the FL Server.
