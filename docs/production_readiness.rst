.. _production_readiness:

##############################
Production Readiness Checklist
##############################

.. note::
   This checklist is coming soon. It will provide a comprehensive pre-go-live
   checklist for production NVIDIA FLARE deployments.

Use this checklist before deploying NVIDIA FLARE in a production environment.
Each section covers critical areas that should be reviewed and validated.

Infrastructure
==============

*Coming soon.* Will cover:

- Hardware requirements validated (CPU, memory, GPU, storage)
- Network connectivity between server and all client sites confirmed
- Firewall rules configured for FLARE communication ports
- DNS resolution working for all participating sites
- Load balancer configured (if applicable)

Security
========

*Coming soon.* Will cover:

- Provisioning completed with proper PKI certificates
- TLS/mTLS configured and verified
- Authorization policies reviewed and tested per site
- Data privacy filters configured (if required)
- Audit logging enabled
- Certificate expiration dates documented and rotation plan in place

For detailed security configuration, see the :ref:`Security Overview <security>`.

Deployment
==========

*Coming soon.* Will cover:

- Deployment mode selected (bare metal, Docker, Kubernetes)
- Server deployed and accessible from all client sites
- All client sites deployed and connected
- Preflight check passed on all sites
- Dashboard UI accessible (if using)

Run the :ref:`Preflight Check <preflight_check>` to validate connectivity:

.. code-block:: bash

   nvflare preflight_check -p startup_kit_dir

Monitoring & Operations
========================

*Coming soon.* Will cover:

- Monitoring stack deployed (StatsD, Prometheus, Grafana)
- Alerting rules configured for critical events
- Log rotation configured
- Backup procedures documented and tested
- Incident response plan in place
- Upgrade/rollback procedure documented

Job Validation
==============

*Coming soon.* Will cover:

- Job tested in FL Simulator
- Job tested in POC mode with representative data
- Expected training time estimated
- Resource requirements (memory, GPU) validated per site
- Timeout values configured appropriately

See :doc:`user_guide/timeout_troubleshooting` for timeout configuration guidance.
