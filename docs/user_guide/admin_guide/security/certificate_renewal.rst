.. _certificate_renewal:

Endpoint Certificate Renewal
============================

This opt-in feature reloads externally issued server/client certificates without
restarting parents or running jobs. The private key and root stay fixed until
restart. Enrollment and renewal scheduling remain deployment responsibilities;
jobs use separate :ref:`per_job_certificates`, not copies of the parent key.

Enable renewal
--------------

First follow :ref:`external_workload_certificates` for provisioning, credential
paths, listener identities, and secret isolation. Renew each endpoint chain with
its corresponding key. The :ref:`external_job_ca` is separate and is not renewed
by this feature. Enrollment must supply certificates valid for the transport's
TLS policy, including the appropriate keyUsage and client/server authentication EKUs.
Renewal is rejected at parent startup for signed HE/Confidential Computing kits,
whose integrity manifest binds the certificate's original bytes.

Add to each parent site's ``local/comm_config.json``:

.. code-block:: json

   {
     "certificate_renewal": true
   }

External server kits record ``auth_identity`` independently of the bind address;
older kits default to the service hostname, and clients to their participant name.
For aliases or wildcard server binds, set it explicitly in the existing
``fed_server.json`` server entry (clients use the ``client`` entry):

.. code-block:: json

   {
     "servers": [{
       "auth_identity": "flare-server.example.org"
     }]
   }

Use end-to-end mTLS. Supported transports are TCP, async TCP, HTTP/WebSocket,
gRPC, and async gRPC; custom drivers are untested. For ``listening_host``, explicitly
set ``connection_security: mtls`` in the project YAML, or set
``internal.resources.connection_security`` to ``mtls`` in the kit's
``local/comm_config.json``. A secure scheme alone does not change that default.
gRPC requires the corresponding credential pair for each role: ``server_cert`` /
``server_key`` for listeners and ``client_cert`` / ``client_key`` for outgoing
connections; it does not substitute the opposite role's pair.

File contract and validation
----------------------------

Write the new PEM chain to a temporary file in the same directory, then atomically
rename it over the certificate file; do not overwrite in place. Keep the same
private key; equivalent PEM re-encoding is allowed.

Credential consumers read files for new TLS handshakes and gRPC channels;
certificate exchange and application authentication read them when needed.
Every second, FLARE checks the chain, validity, original public key, configured
CN, and non-CA leaf constraint for diagnostics. Changes do not restart listeners
or disconnect established sessions.

These checks are not an activation gate: a consumer can read a replacement before
the next check or after it fails. FLARE adds no application-managed fallback
snapshot. Bad enrollment output can interrupt service and fail jobs; correct the
files to recover communication. Enrollment tooling owns certificate profiles, identity/permission
stability, and validation before publication. Stop the endpoint before replacing
its root or private key, then restart it. Replacing a key/certificate pair while
running can let new TLS handshakes succeed while encrypted messages fail: TLS
reads the new pair, but application encryption retains the startup key. The
diagnostic reports this mismatch; it does not block credential consumers.

Connection and failure behavior
-------------------------------

Python TLS listeners select a fresh context for each handshake and fail that
handshake if credential loading fails. gRPC listeners also attempt to reload for
each new connection, but the gRPC runtime retains its previous configuration if
the replacement cannot be loaded. Removing or corrupting a certificate file is
therefore not a way to disable new connections. Peer certificate validation still
applies to the retained credentials.

Existing sessions keep their original authenticated identity and may continue
past that certificate's expiry. A valid same-key renewal does not interrupt jobs
or restart parent/job processes.

Transport failures still use the existing reconnect behavior. Fresh certificate
authentication rejects expired credentials; TLS session resumption follows the
transport library's policy. Bad enrollment and unrelated outages can still fail
jobs. This feature adds neither outage recovery guarantees nor periodic reauthentication.

Certificate renewal is not access revocation. Refusing further issuance does not
evict an already-connected participant. Known compromise requires blocking access,
terminating affected sessions, and installing a new key on a trusted endpoint.
Reissuing a certificate for the same key does not repair a compromised key.
Root rollover, live key rotation, and admin/relay renewal are separate features.

Enrollment and key-age policy
-----------------------------

Configure the agent/issuer for the reference policy: 30-day certificates renewed
around half-life after reauthorization (shorter lifetimes for expiry tests).

For step-ca, submit fresh authorized signing requests over the existing key's CSR.
Derive identity and allowed SANs from authenticated workload identity, not the CSR;
disable possession-only renewal and other reauthorization bypasses. Neither
certificate duration nor identity templates enforce the reused key's age.

The issuer must track authoritative key history, reject issuance beyond one year
from key creation (or a shorter project limit), and warn before the required
restart. Certificate dates, file timestamps, and endpoint records do not prove
key age; FLARE reports it as unknown. Without this enforcement, renewal does not
bound key lifetime.

Agents can publish through a shared volume or host service; on Slurm, enrollment
belongs on the parent service node. cert-manager and SPIRE are compatibility
targets only: verify same-key issuance, certificate profiles, and atomic publication.

Operations and verification
---------------------------

``process_info <target>`` reports identity, SHA-256 fingerprint, effective expiry,
expired state, last observed renewal, errors, and unknown key age. These describe
the last successfully checked certificate, not every connection's loaded credential.
It exposes no keys or tokens and is not an issuer enrollment audit.

For key replacement or suspected compromise, stop the endpoint, generate a new
local key, enroll, install the pair, and restart. Plan for job interruption.

Run the unit, five-driver live TLS, and CPU-only federated-job tests:

.. code-block:: bash

   python -m pytest tests/unit_test/security/certificate_renewal_test.py
   python -m pytest tests/integration_test/slow/certificate_renewal_live_test.py
   python -m pytest tests/integration_test/slow/certificate_renewal_test.py

Tests generate temporary kits and jobs; no GPU or Kubernetes cluster is required.
