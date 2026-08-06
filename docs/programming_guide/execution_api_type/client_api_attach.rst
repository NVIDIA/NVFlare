.. _3rd_party_integration:
.. _external_trainer_integration:
.. _client_api_attach:

######################
Client API Attach Mode
######################

Client API Attach connects an independently started and externally owned trainer
process to an NVFLARE Client Job (CJ). The trainer initiates the connection to a
dedicated listener owned by the CJ. NVFLARE exchanges tasks and results with the
trainer, but never starts, stops, signals, or waits for the trainer process.

An *independently managed trainer* is defined by process ownership, not by its
ML library or vendor: an external system starts, monitors, and stops it.
Library-specific integrations that NVFLARE launches and orchestrates, such as
Flower, are not Attach integrations.

.. code-block:: text

   External system  -- starts and owns -->  Trainer
   NVFLARE runtime  -- starts and owns -->  Client Job
   Trainer          ------ attaches ----->  Client Job Attach listener

Choose an Execution Mode
========================

Choose the Client API execution mode from trainer-process ownership, not from
which ML library the trainer uses.

.. list-table:: Client API execution modes
   :header-rows: 1
   :widths: 24 24 52

   * - Mode
     - Trainer process owner
     - Use when
   * - ``attach``
     - An external system
     - The trainer is started independently and must remain outside NVFLARE's
       process lifecycle.
   * - ``external_process``
     - NVFLARE
     - NVFLARE should launch and manage a separate trainer process.
   * - ``in_process``
     - NVFLARE
     - Training should run inside the CJ process.

Attach is not a general replacement for integrations that have their own
NVFLARE-managed execution model. For example, Flower uses ``FlowerController``
and ``FlowerExecutor`` on top of TIE (Technology for Integrating Everything),
and its application processes remain orchestrated by NVFLARE.

How Attach Works
================

The CJ owns a dedicated Attach listener and one Attach protocol session. The
trainer uses an Attach profile to discover or address that listener and then
uses the ordinary :mod:`nvflare.client` API. Task and result payloads travel
over Cell and use NVFLARE's existing serialization and large-object transfer
support.

The job and trainer share an ``attach_id``. It is a stable rendezvous and
routing name, not a password or authentication secret. Transport security is
provided by a protected shared filesystem or mutual TLS (mTLS).

Attach separates three configuration responsibilities:

* The **site administrator** configures the dedicated listener in the site's
  ``local/comm_config.json``.
* The **job author** configures
  :class:`ClientAPIExecutor<nvflare.app_common.executors.client_api_executor.ClientAPIExecutor>`
  with ``execution_mode="attach"`` and an ``attach_id``.
* The **trainer owner** supplies the matching Attach profile and starts the
  trainer independently.

Configure the Site Listener
===========================

The ``client_api_attach`` section is independent of the site's ``internal``
connection between the Client Parent (CP) and CJ. Changing
``local/comm_config.json`` requires restarting the site.

Shared-Filesystem Listener
--------------------------

Shared-file Attach is recommended when the CJ and trainer can access the same
filesystem at the same absolute path. It permits either the trainer or job to
start first, and the trainer needs no network access or TLS credentials.

Add the following to the client site's ``local/comm_config.json``:

.. code-block:: json

   {
     "client_api_attach": {
       "scheme": "shared-file",
       "resources": {
         "root_dir": "/absolute/shared/nvflare-client-api-attach",
         "connection_security": "clear"
       }
     }
   }

The CJ and trainer must see ``root_dir`` at the same absolute path. Use a
dedicated, non-world-writable directory whose group contains only the intended
site and trainer principals. The filesystem must support coherent atomic rename
and cross-node POSIX advisory locks.

For all listener properties and network configuration, see
:ref:`client_api_attach_configuration`.

Configure the Job
=================

The client job uses ``ClientAPIExecutor`` in Attach mode. This JSON example
configures a NumPy task exchange:

.. code-block:: json

   {
     "format_version": 2,
     "executors": [
       {
         "tasks": ["train"],
         "executor": {
           "path": "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor",
           "args": {
             "execution_mode": "attach",
             "attach_id": "trainer_a",
             "attach_timeout": 300.0,
             "heartbeat_interval": 5.0,
             "heartbeat_timeout": 30.0,
             "task_wait_timeout": 600.0,
             "params_exchange_format": "numpy",
             "server_expected_format": "numpy",
             "params_transfer_type": "FULL"
           }
         }
       }
     ],
     "task_result_filters": [],
     "task_data_filters": [],
     "components": []
   }

In a production deployment, the client site's component policy must allow the
exact ``ClientAPIExecutor`` class. Do not replace a site allow-list with a broad
package prefix.

Configure the Trainer
=====================

The trainer gets its connection information from a typed Attach profile. For
the shared-filesystem listener above, use:

.. code-block:: json

   {
     "schema_version": 1,
     "execution_mode": "attach",
     "attach_id": "trainer_a",
     "site_name": "site-1",
     "rendezvous_dir": "/absolute/shared/nvflare-client-api-attach",
     "job_wait_timeout": null
   }

``attach_id`` must match the job configuration, and ``rendezvous_dir`` must
match the site's ``root_dir``. ``job_wait_timeout`` bounds how long the trainer
waits for a matching job; ``null`` means it can wait without a deadline.

The trainer uses the same Client API loop as other execution modes:

.. literalinclude:: ../../resources/client_api_attach_trainer.py
   :language: python

``flare.init(config_file=...)`` recognizes the typed profile and selects the
Attach Client API engine. After initialization, the trainer can use
``flare.get_task_name()``, tracking APIs, and the normal ``FLModel`` receive/send
contract.

Start and Verify
================

With shared-file rendezvous, the trainer and job may start in either order:

.. code-block:: shell

   # Started by the trainer's external owner
   python trainer.py --config attach_profile_shared_file.json

   # Submitted independently through NVFLARE
   nvflare job submit -j <job-directory>

The CJ publishes its dynamic listener information while it holds the rendezvous
claim. A trainer that starts first waits for that claim; a job that starts first
waits for the trainer until ``attach_timeout`` expires.

Successful attachment creates a trainer Cell whose FQCN is a child of the CJ:

.. code-block:: text

   <site>.<job_id>.-client_api_<attach_id>

When the job ends, the CJ closes the Attach session and listener. It does not
terminate the trainer process. The trainer's ``is_running()``/``receive()`` loop
observes session shutdown and exits; its external owner remains responsible for
the process and any restart policy.

Network Attach
==============

Use network Attach when the trainer cannot share a filesystem with the CJ.
Production network Attach requires mTLS. The site needs a provisioned listener
certificate and key, and the trainer needs the site client certificate, client
key, and CA certificate.

A direct network profile contains the job-specific CJ FQCN and listener URL, so
the trainer normally starts after the job identity is known. Clear network
Attach, including loopback, is development-only and requires explicit insecure
opt-in on both the profile and job.

See the :github_nvflare_link:`Client API Attach example <examples/advanced/client-api-attach>`
for complete shared-file, mTLS network, and local POC instructions.

Security and Lifecycle Requirements
===================================

* Treat ``attach_id`` as public routing metadata, not a credential.
* Protect shared-file roots with filesystem ownership and group permissions.
* Use mTLS for production network Attach; one-way TLS is not supported.
* Do not use clear network Attach on an untrusted network.
* Keep the trainer alive until ``flare.send()`` returns. Large results may still
  be served from the trainer after their result envelope is accepted.
* Use a finite heartbeat or result-wait policy so a dead external trainer cannot
  leave a job waiting indefinitely.

Migration from Legacy External-Trainer Integration
==================================================

Attach replaces the former agent/exchanger patterns for independently managed
trainers.

.. list-table:: Legacy-to-Attach mapping
   :header-rows: 1
   :widths: 42 58

   * - Legacy external-trainer concept
     - Attach replacement
   * - ``agent_id`` or a pipe token
     - ``attach_id``
   * - ``TaskExchanger`` or ``IPCExchanger`` in the CJ
     - ``ClientAPIExecutor(execution_mode="attach")``
   * - ``FlareAgent`` or ``IPCAgent`` in the trainer
     - :mod:`nvflare.client` initialized with an Attach profile
   * - General ad-hoc connection or task pipe
     - Dedicated site-local ``client_api_attach`` listener
   * - Trainer workspace containing generated live job configuration
     - Independently supplied shared-file or direct network Attach profile

This migration applies only when the trainer is independently started and
externally owned. Trainers launched by NVFLARE should use ``external_process``;
trainers running in the CJ should use ``in_process``.

Troubleshooting
===============

* **The job never sees the trainer:** verify that ``attach_id`` and ``site_name``
  match and that ``attach_timeout`` has not expired.
* **A shared-file trainer never discovers the job:** verify identical absolute
  paths, cross-node advisory locking, and directory permissions.
* **A network trainer cannot connect:** verify the job-specific ``cj_fqcn``,
  listener URL, mTLS certificate paths, hostname, and firewall rules.
* **A second job is rejected:** one live job owns a given shared-file
  ``(site_name, attach_id)`` claim. Use a distinct ``attach_id`` for concurrent
  jobs.
* **Job completion waits after a result:** keep the trainer running while the
  result's lazy payloads reach terminal receiver confirmation.

For the protocol, retry, reconnect, security, and result-lifetime design, see
:github_nvflare_link:`Client API Attach Mode <docs/design/client_api_attach_mode.md>`.
