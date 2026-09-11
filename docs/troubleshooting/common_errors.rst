:orphan:

.. _common_errors:

############################
Common Errors & Solutions
############################

.. note::
   This guide is coming soon. It will catalog the most frequently encountered
   errors in NVIDIA FLARE with root causes and solutions.

Overview
========

This page provides solutions for the most common errors encountered when developing,
testing, and deploying NVIDIA FLARE applications.

Installation & Setup Errors
============================

*Coming soon.* Will cover:

- Python version compatibility issues
- Package dependency conflicts
- GPU/CUDA setup issues

Connection & Communication Errors
===================================

*Coming soon.* Will cover:

- Client cannot connect to server
- TLS/SSL handshake failures
- Certificate errors
- Firewall and port issues
- Proxy configuration problems

Job Submission & Execution Errors
==================================

*Coming soon.* Will cover:

- Job submission failures
- Resource allocation errors
- Timeout errors (see also :doc:`/user_guide/timeout_troubleshooting`)
- Out-of-memory errors (see also :doc:`/programming_guide/memory_management`)
- Model serialization errors

Job fails with ``FAILED_TO_RUN`` and a job credential message
---------------------------------------------------------------

In secure mode every job runs on a :ref:`per-job certificate <per_job_certificates>`
and there is no fallback to the site certificates. The reason is recorded in
the job's ``job_deploy_detail``:

- ``server startup kit has no job CA (job_ca.crt / job_ca.key)``: the server was
  provisioned before this feature or with ``enable_job_ca: false``.
  Re-provision the project and redeploy the server startup kit.
- ``job CA expires at ...``: the job CA has less than one hour of validity
  left. Re-provision the project to renew it.
- ``deploy request carries no valid job credential`` (reported by a client):
  the server did not send a job credential. The server and client releases do
  not match; run the same release on all participants.
- ``has no job credential`` (reported by the Docker, Kubernetes, or Slurm
  launcher): same cause as above.

Training & Convergence Issues
==============================

*Coming soon.* Will cover:

- Model not converging in federated setting
- NaN/Inf values during training
- Data loading errors across sites
- Inconsistent results between simulator and production

Deployment & Operations Errors
================================

*Coming soon.* Will cover:

- Provisioning failures
- Dashboard UI issues
- Monitoring setup problems
- Log file analysis

Getting Help
============

If your error is not listed here:

1. Check the :doc:`/faq` for general questions
2. Run the :ref:`Preflight Check <preflight_check>` to diagnose connectivity issues
3. Search the `NVIDIA FLARE GitHub Issues <https://github.com/NVIDIA/NVFlare/issues>`_
4. Open a new issue with logs and error details
