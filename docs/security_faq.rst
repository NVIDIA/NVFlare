.. _security_faq:

############################
Security FAQ
############################

Where is the data being stored?
================================

The data is stored in local institutions' storage, either on-premises or in private cloud accounts. FLARE doesn't move the data.

What results are being shared with the central federated site?
===============================================================

The "central federated site" is known as the "FL server" in NVFlare. The location of the final results depends on the aggregator location, which varies by algorithm.

- **For FedAvg-type workflows** -- The aggregator is in the FL Server, and the final result is stored in the Job Store, which is a local disk or volume accessible to the FL server
- **For Swarm learning algorithm** -- The aggregator is randomly selected at each round among all participating sites, and the final result is stored locally at that site, not shared with the FL Server

In both cases, the aggregator can be a locked-down machine using current security and privacy best practices. Confidential Computing is the latest technology that adds an extra layer of security.

What's the IRB number?
========================

The IRB process is outside the scope of NVIDIA FLARE; consortium participants will need to handle this separately.

What do we need to install locally?
======================================

FLARE involves the following steps for local installations:

1. **Provision** -- A process to generate software packages and certificates for each participant (called startup kit)
2. **Distribution** -- Send the startup kit to each participant
3. **Start** -- Participants can pip install nvflare and start the startup kit

FLARE offers two ways to provision:

- **nvflare provision CLI command** -- The package is generated locally, the project administrator will then distribute the package manually (sftp, email etc.)
- **FLARE Dashboard** -- The web interface allows the project administrator to invite others to join the project and provide site-specific information themselves

For details, see :ref:`installation` and :ref:`deployment_overview`.

What's the infrastructure on each institution's side?
=======================================================

FLARE doesn't mandate a specific type of infrastructure, unless you want to leverage confidential computing with IP protection. You can run on CPU or GPU. The minimal requirement is an 8GB CPU with a Linux distribution (such as Ubuntu). For deep learning models, you will need a GPU for faster training.

Do we need to run Docker on our end?
=======================================

No, that's not required. You can use Docker if you like.

Does it run on Red Hat?
=========================

Yes.

Who maintains FLARE?
======================

NVIDIA FLARE is an open source project, contributed and maintained by NVIDIA and the NVFLARE community.

The software is distributed under the Apache 2.0 License, which is a permissive open-source license. Under this license, the software is provided "as is" without warranties or liabilities.

If formal support or indemnification is required, it can be obtained through a third-party service provider that offers commercial support for Apache 2.0-licensed software.

Who owns the data after it leaves the institution?
=====================================================

No raw data ever leaves any institution. Only the model weights are transmitted.

The model trained can be owned by different participants depending on the collaboration agreement. NVIDIA FLARE is not involved in these business decisions.

Is there a data use agreement?
================================

Since the data never leaves the institution, usually there is no specific data use agreement. However, collaborators usually need to decide what data to use to jointly train a model.

Is the code proxy-aware?
==========================

Yes, NVIDIA FLARE can operate through network topologies with proxies, such as reverse-proxies or Kubernetes ingress services.

FLARE supports two types of TLS protocols: mutual TLS and standard TLS.

- **For mutual TLS** -- The certificate termination point is at the server, not at the proxy. The proxy must be configured to enable TCP pass-through
- **For standard TLS** -- Users can simply use the pre-authorized certificate for TLS handshaking and the FLARE certificate for authentication

For details, see :ref:`Communication Security <communication_security>` and :ref:`Server Port Consolidation <server_port_consolidation>`.
