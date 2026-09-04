.. _per_job_certificates:

####################
Per-Job Certificates
####################

In secure mode (``secure_train=true``, the normal production setting where all
participants authenticate with TLS), every job process — the server job process
(SJ) and each client job process (CJ) — runs on its own short-lived X.509
certificate, issued for that job alone. Job processes run job-supplied code (custom Controllers and
Executors, third-party training code). Before this feature that code had access
to the site's long-lived certificate and private key: whatever the site could
do, the job could do, for as long as the site key stayed valid.

With per-job certificates:

- a job process never refers to the site's private key; its TLS and
  message-level credential is the job certificate;
- the Docker, Kubernetes, and Slurm job launchers do not give the job any site
  private key;
- a job certificate cannot register a site, log in as an administrator,
  impersonate the server, or act as another job's process, even if it leaks;
- there is no fallback: in secure mode a job either gets its certificate or it
  does not run.

Non-secure mode (``secure_train=false``) and the simulator use no certificates
and are not affected.

How it works
============

Provisioning creates a job-signing intermediate CA — ``job_ca.crt`` and
``job_ca.key`` — in the **server** startup kit only. It is signed by the project
root CA, so no participant needs a new trust anchor. When a job is deployed, the
server issues one certificate per participating site (``CN=<site name>``, with
an extension carrying the job ID), writes the server job's credential into the
job's run directory, and sends each client its own credential inside the deploy
message over the existing mutually authenticated channel. Every site verifies
these certificates against the ``rootCA.pem`` it already has.

Job certificates are valid for 30 days by default (see `Certificate validity`_)
and are not renewed: a job cannot run longer than its certificate.

What to expect after provisioning
=================================

The server startup kit gains two files::

   startup/
     job_ca.crt      job-signing CA certificate, signed by the project root
     job_ca.key      its private key, mode 0600 — protect it like server.key

Client and admin kits are unchanged. Keep ``job_ca.key`` with the server's other
secrets. Whoever holds it can issue job certificates, and nothing else:
site-level authentication (client registration, admin login, server identity)
rejects any certificate the job CA signed.

Each running job has its credential at::

   <workspace>/<job id>/job_cert/job.crt    certificate followed by job_ca.crt
   <workspace>/<job id>/job_cert/job.key    private key, mode 0600

Both files are deleted when the job process exits. The job workspace archived
on the server (``download_job``) never contains them.

Enabling and disabling
======================

The job CA is created by default. To turn it off, set ``enable_job_ca`` on
``CertBuilder`` in ``project.yml``:

.. code-block:: yaml

   builders:
     - path: nvflare.lighter.impl.cert.CertBuilder
       args:
         enable_job_ca: false

Only do this for non-secure deployments (``secure_train=false``, e.g. local
testing). A production server provisioned without the job CA cannot run jobs:
every job deploy fails with ``server startup kit has no job CA``, because jobs
are never started on the site's own certificate.

Upgrading an existing project
=============================

Server startup kits provisioned before this feature have no job CA. Once the
server runs a release with per-job certificates, every job deploy fails until
the project is re-provisioned:

#. Run ``nvflare provision`` again on the same provisioning workspace. The root
   CA and all participant certificates are reused from the workspace's
   ``state`` directory; only ``job_ca.crt`` / ``job_ca.key`` are added to the
   server kit.
#. Redeploy the server startup kit. Client and admin kits do not change.

Run the same NVFlare release on the server and on all clients. An older client
would run its job process on the site certificate, which a server with per-job
certificates does not allow; the client reports ``deploy request carries no
valid job credential`` and the job fails.

Certificate validity
====================

Job certificates are issued for ``job_cert_valid_days`` (default 30), clamped
to the job CA's own expiry. There is no renewal, so this value is the maximum
duration of a job; set it for the longest job the server runs. It is a server
setting, resolved in this order:

#. ``--set job_cert_valid_days=<days>`` on the server start command;
#. a top-level ``"job_cert_valid_days": <days>`` entry in the server's
   ``fed_server.json``;
#. the environment variable ``NVFLARE_JOB_CERT_VALID_DAYS``.

The job CA itself is valid for 360 days, bounded by the root CA. When it has
less than one hour of validity left, job deploys fail with ``job CA expires at
...``; re-provision the project to renew it (``nvflare provision`` regenerates
an expired job CA automatically).

Job launchers
=============

Job processes never need a site private key, so the launchers do not give them
one. In secure mode each launcher refuses to start a job that has no job
credential instead of falling back to site certificates.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Launcher
     - What the job process sees
   * - Process (in-process)
     - The site workspace as-is. The job process runs as the same user on the
       same host, so the site key is readable on disk; only the job's own
       configuration no longer refers to it.
   * - Docker
     - ``startup/`` bound into the container file by file, read-only, without
       ``*.key`` files. The job credential arrives with the job's read-write
       workspace bind.
   * - Kubernetes
     - A startup-kit Secret without ``*.key`` files. The job certificate and
       key are delivered through the per-pod credential Secret
       (``nvflare-cred-<pod>``) as the ``NVFLARE_JOB_CERT`` and
       ``NVFLARE_JOB_KEY`` variables, which the job process writes to its run
       directory and removes from its environment before anything else starts.
       Workspace bundles never contain the job credential.
   * - Slurm, ``sandbox: apptainer`` or ``pyxis``
     - A copy of the startup kit without ``*.key`` files, staged under the
       job's transient artifact directory and bound at ``<workspace>/startup``.
       The job credential is in the job's run directory.
   * - Slurm, ``sandbox: none``
     - The shared filesystem as-is, like the process launcher.

See :ref:`deploy_prepare_command`, :ref:`helm_chart`, and
:ref:`slurm_job_launcher`.

.. _per_job_certificates_external_ca:

Distributed provisioning and external PKI
=========================================

Kits assembled with the distributed ``nvflare cert`` / ``nvflare package``
workflow (:ref:`distributed_provisioning`) have no job CA, because the packaging
step never has the root CA key. A server running from such a kit cannot run
jobs in secure mode. Extending that workflow is addressed separately.

If you operate your own PKI you can issue the job CA yourself. The server only
requires that ``startup/job_ca.crt`` and ``startup/job_ca.key`` exist and that
the certificate chains to the ``rootCA.pem`` in the kits. The job CA must be an
intermediate CA certificate with:

- ``basicConstraints = critical, CA:TRUE, pathlen:0``;
- ``keyUsage = critical, digitalSignature, keyCertSign, cRLSign``;
- the job-CA marker: a **non-critical** extension with OID
  ``1.3.6.1.4.1.5703.300.2`` (any value). Sites use it to reject anything the
  job CA signed when a site identity is asserted, so it must be present.

The root CA certificate in ``rootCA.pem`` must itself carry a ``keyUsage``
extension that includes ``keyCertSign``. NVFlare validates certificate chains
strictly per RFC 5280 and rejects a root without it; roots created by
``nvflare provision`` or ``nvflare cert init`` always have it, but the default
OpenSSL ``v3_ca`` profile leaves ``keyUsage`` commented out.

An OpenSSL extension section that produces this:

.. code-block:: ini

   [ job_ca_ext ]
   basicConstraints = critical, CA:TRUE, pathlen:0
   keyUsage = critical, digitalSignature, keyCertSign, cRLSign
   subjectKeyIdentifier = hash
   authorityKeyIdentifier = keyid:always
   1.3.6.1.4.1.5703.300.2 = ASN1:UTF8String:job_ca

.. code-block:: bash

   openssl req -new -newkey rsa:2048 -nodes -keyout job_ca.key \
       -subj "/CN=job_ca.<project name>" -out job_ca.csr
   openssl x509 -req -in job_ca.csr -CA rootCA.pem -CAkey rootCA.key \
       -CAcreateserial -days 360 -extfile job_ca.cnf -extensions job_ca_ext \
       -out job_ca.crt
   chmod 600 job_ca.key

Copy ``job_ca.crt`` and ``job_ca.key`` into the server's ``startup/`` directory
before starting the server.

Troubleshooting
===============

Failures are recorded in the job's ``job_deploy_detail`` (shown by
``list_jobs`` and in the job metadata) and in the server or client log.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Message
     - Cause and fix
   * - ``server startup kit has no job CA (job_ca.crt / job_ca.key)``
     - The server kit was provisioned before this feature, with
       ``enable_job_ca: false``, or by ``nvflare package``. Re-provision the
       project and redeploy the server kit.
   * - ``job CA expires at ... (less than 1:00:00 left)``
     - Re-provision the project to renew the job CA.
   * - ``deploy request carries no valid job credential`` (client)
     - The server sent no credential: the server and client releases differ.
       Run the same release everywhere.
   * - ``has no job credential; secure jobs run only on per-job certificates``
       (Docker, Kubernetes, or Slurm launcher)
     - Same cause as above.
   * - ``authenticated with a certificate bound to job '...' but claimed
       endpoint ... is not part of that job``
     - A process presented another job's certificate. This does not happen in
       normal operation; investigate the site.
