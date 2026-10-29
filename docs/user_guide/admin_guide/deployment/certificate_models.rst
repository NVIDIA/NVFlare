.. _certificate_models:

Installing NVIDIA FLARE with the right certificate model
========================================================

*A deployment guide for IT managers*

You can install a secure FLARE federation without connecting it to an enterprise certificate service. FLARE
can create its own trust root, issue the certificates needed by the participants, and give each running job
separate credentials. This self-contained model is the simplest to operate, but your team must protect the
provisioning workspace and distribute private keys securely.

Choose separately how people obtain admin credentials, who supplies server and site credentials, and who
supplies the server’s job-CA credential. These choices can be combined under one trusted root.
Public-trust-only provisioning requires external sources for admin, endpoint, and job-CA credentials. The
sections build the explanation from a self-contained example; you can adopt the choices independently, subject
to the dependencies in the final table. This guide covers the FLARE 2.10 deployment model.

1 Start with a self-contained federation
----------------------------------------

Assume you are installing one FLARE server and a long-running FLARE client at each participating site. The
server coordinates jobs, and each site’s client connects to it. This guide calls the server and these clients
endpoints. Participants is the broader term used in provisioning for the server, site clients, and admin
users.

Researchers use the FLARE administrator client to submit work. Each user receives the permissions assigned to
their role and studies. A study is a named collaboration with enrolled sites and authorized users.

What provisioning does
~~~~~~~~~~~~~~~~~~~~~~

Define the participants, addresses, and roles in ``project.yml``, then run ``nvflare provision``. In the
self-contained model, the :ref:`provisioning tool <provisioning>` creates a root certificate authority, or root CA.

Provisioning creates a startup kit for each participant: a package containing configuration, launch scripts,
public trust, and the credentials that participant needs.

A certificate binds an identity to a public key. The matching private key proves that identity; the CA signs
the certificate so other participants can verify it.

The root CA consists of a public certificate and a private signing key. Every startup kit receives the public
certificate as ``rootCA.pem``. The root private key stays in the provisioning workspace and its backups. It is
not installed on the running server or clients. The provisioning tool performs the root-CA role.

In the self-contained model, provisioning generates the following certificate material.

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Credential
     - What it is for
     - Where the private key is installed
   * - Root CA
     - Signs the federation certificates
     - Provisioning workspace only
   * - Server certificate
     - Identifies the long-running server
     - Server startup kit
   * - Client certificate for each site
     - Identifies the long-running site client
     - That site’s startup kit
   * - Administrator certificate for each user
     - Authenticates the user and signs submitted jobs
     - That user’s admin startup kit
   * - Server job CA certificate
     - Authorizes the server to sign job certificates (Section 2)
     - Server startup kit only
   * - Additional listener certificates
     - Authenticate a client that accepts inbound connections
     - The corresponding participant kit

The provisioning workspace retains generated certificates and private keys. Protect it as sensitive CA
material. Deliver each kit through an authenticated, protected channel and restrict access at its destination.

What you must operate
~~~~~~~~~~~~~~~~~~~~~

Install the server kit and each site’s client kit, configure the required network access, and start the
participants. Their certificates let them authenticate one another using mutual TLS. FLARE also checks the
expected identities and permissions; possession of a certificate from the trusted CA does not grant arbitrary
access.

No external CA service is needed during normal operation. Your team owns certificate expiry planning,
replacement, kit distribution, and protection of the provisioning workspace. Use this model when keeping those
responsibilities within the federation is acceptable.

Certificate ownership at a glance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: /resources/certificate_credential_sources.svg
   :alt: Three credential-source choices under one trusted root: admin credentials in purple,
         endpoint credentials in blue, and the server job CA in teal. Public-trust-only
         provisioning requires external sources for all three; the server always issues job credentials.
   :width: 100%
   :align: center
   :target: ../../../_images/certificate_credential_sources.svg

Choose credential sources independently under one trusted root. The colored lanes show the choices; the orange
band shows the dependencies of public-trust-only provisioning. The final table adds compatibility and
migration details.

2 Give each running job its own keys
------------------------------------

A site’s long-running client and the training code it launches have different responsibilities. The client
needs a reusable site identity. A training job should receive only the identity needed for that job.
:ref:`Per-job credentials <per_job_certificates>` provide this separation in the self-contained model as well
as in enterprise deployments.

What the job CA is
~~~~~~~~~~~~~~~~~~

During provisioning, the root CA signs a separate job-CA certificate and installs it with its private key in
the server kit. This is an intermediate CA: its authority comes from the root, but it can issue only
end-entity certificates, not another CA. FLARE recognizes the job CA’s purpose and rejects certificates it
issued when they are presented as site or administrator identities.

The server endpoint key and job-CA key are different. The first identifies the server; the second lets the
built-in job CA issue credentials for job processes. Neither activity needs the root private key.

What happens when a job starts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A user signs and submits the job. Once the job is accepted for deployment, the server generates a new private
key and certificate for the server-side job process and for each participating client-side job process. It
signs the certificates with its built-in job CA, installs the server-job credential locally, and sends each
site’s job credential through that site’s authenticated deployment message.

Each certificate is restricted to its site, job, and allowed communication identities. The job can communicate
as that job, not as the long-running site or a different job. No external CA request is made at job launch,
and the submitter does not generate or distribute these runtime keys.

Keep parent site keys and the job-CA key outside the job’s filesystem and mounts. Use an appropriately
isolated Docker, Kubernetes, or container-based Slurm launcher. A separate process running with unrestricted
access to the parent’s files is not a security boundary.

Job credentials are issued once and are not renewed. Their default maximum lifetime is 30 days, limited
further by the remaining validity of the job-CA certificate and the CA certificates above it. Plan
long-running jobs accordingly.

3 Replace manually distributed user credentials with SSO
--------------------------------------------------------

You can add organizational sign-in for people independently of how server, site, and job-CA credentials are
issued. With single sign-on, or SSO, you distribute an admin kit containing configuration and public trust.
Each user obtains their own short-lived credential after signing in through the organization’s identity
provider.

SSO replaces the manual distribution of long-lived user certificates. It does not remove certificates from
FLARE authentication or job signing.

What you install and configure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configure an :ref:`admin certificate provider <admin_certificate_provider_configuration>` in the kit. The
reference integration uses Smallstep’s step command-line tool on the user’s computer, its step-ca certificate
service, and the organization’s identity provider. The issuing CA’s certificate must chain to the federation’s trusted root (see Section 4).

When the user signs in, the provider creates a private key locally, obtains proof of organizational
authentication, and requests a short-lived certificate from the issuing service. CA and identity-provider
policies determine the permitted identity, organization, role, and any study memberships, called entitlements.
The private key stays on the user’s computer.

The admin client presents the certificate to FLARE and uses the corresponding key to sign submitted jobs.
FLARE continues to enforce role and study permissions. SSO authenticates the person; it does not automatically
authorize that person for every federation or study.

What changes for operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Your team now maintains provider configuration and the sign-in service rather than delivering each person’s
private key. Users need access to the identity provider and certificate service when obtaining a new
certificate. SSO can coexist with FLARE-managed server/site certificates and the FLARE-issued job CA, provided
the SSO issuer chains to the same trusted root.

Adding SSO does not by itself move the root key out of the provisioning workspace. The SSO certificate issuer
must chain to the same root as the federation’s other certificates, whether that root is federation-managed or
enterprise-managed.

Admin sessions expire with their certificates; already-running jobs continue with their separate job
credentials. A queued or resubmitted job may need a fresh signature because the submitter certificate must be
valid when the job is deployed.

For jobs with standing approval that are rerun regularly, IT can instead supply a non-ephemeral certificate to
a narrowly authorized machine account. File-based admins and SSO users can coexist.

4 Use enterprise services to issue the infrastructure certificates
------------------------------------------------------------------

You may also want your PKI team to control the server and site certificates. In that arrangement, an
enterprise CA issues the certificates and deployment tooling installs them where FLARE will use them. FLARE
continues authenticating the same participants; responsibility for obtaining and maintaining their credentials
moves to IT. Before choosing issuers, agree on the common trust root described under “Agree on the trust root
before changing issuers” below.

Replace endpoint credentials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For :ref:`externally managed server and client certificates <external_workload_certificates>`, set
``external_cert: true`` in ``project.yml`` for each affected participant. Provisioning omits those credentials from the generated kits. Your deployment tooling
generates the key at the endpoint, obtains an approved certificate and chain, and installs the matching files
before startup. Install ``server.crt`` and ``server.key`` in the server’s startup directory, or ``client.crt``
and ``client.key`` in the client’s startup directory.

A listener accepts incoming network connections. A client configured with ``listening_host`` in
``project.yml`` also needs a separate server certificate and key for those connections.

FLARE does not provide the enrollment agent or call the CA itself. Kubernetes deployments can use a controller
or sidecar whose workload identity the CA recognizes. Docker and bare-metal deployments need an IT-approved
bootstrap identity, such as a restricted enrollment token or managed host identity. The issuer must restrict
which participant that requester can obtain credentials for.

The same tooling schedules replacement. Stop the affected FLARE endpoint, install the new matching certificate
and key, then restart it, even when reusing the existing key. Plan for interrupted jobs and allow time for
peers to reconnect and authenticate after the restart.

Supply the server job CA separately
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The enterprise can also issue the server’s job-CA certificate. Set ``external_job_ca: true`` for the server in
``project.yml``, generate the job CA’s private key at the server, and install ``job_ca.crt`` and
``job_ca.key`` in the server startup directory. This is independent of who issued the server’s endpoint
certificate.

Ask the PKI team for a CA certificate with the FLARE job-CA profile, not an ordinary server certificate. It
must authorize job-certificate signing and satisfy the restrictions of the CAs above it. A path-length
constraint of zero (``pathlen:0``) permits a CA to sign end-entity certificates, but forbids another CA below
it. Such a CA cannot issue the server’s job-CA certificate.

One suitable enterprise hierarchy places the endpoint-issuing CA and the FLARE job CA beside each other
directly under the root. The enterprise issuer signs endpoint and user certificates; the FLARE job CA signs
job certificates. A deeper hierarchy is also possible when its constraints explicitly permit it. Do not relax
existing enterprise CA constraints to accommodate the deployment.

After installation, the built-in job CA still issues the individual job credentials locally. Moving its
certificate issuance to enterprise PKI does not introduce an enterprise CA call for every job. Job-CA renewal
and rotation are separate from routine endpoint replacement and are outside this design.

Agree on the trust root before changing issuers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All admin, endpoint, and job certificate chains must reach the same configured root. You cannot replace one
site’s certificate with a certificate under an unrelated enterprise root and expect the existing federation to
accept it.

For a new federation, choose that common root at deployment design time. Moving an existing federation from a
FLARE-generated root to an enterprise root requires a coordinated trust and credential migration. These
options do not provide automatic root rotation or a multiple-root transition.

5 Remove root private keys from FLARE provisioning
--------------------------------------------------

Externally issuing selected certificates is not enough to remove signing authority from provisioning. Legacy
provisioning may still generate other credentials, and the existing option to supply your own root requires
both the root certificate and its private key. That workflow changes where the root originated, but FLARE
provisioning still holds the signing key.

:ref:`Public-trust-only provisioning <public_root_only_provisioning>` removes that need. IT supplies only the
enterprise’s public root certificate. The provisioning tool generates configuration and launch scripts and
distributes the public trust certificate.
It neither receives a root private key nor generates or requests participant credentials.

What you provide
~~~~~~~~~~~~~~~~

Configure these credential sources in ``project.yml``:

- Public root: set ``CertBuilder.args.public_root_cert`` to a certificate-only PEM containing the single
  valid, self-signed enterprise root.

- Server and site clients: set ``external_cert: true`` for each participant.

- Server job CA: also set ``external_job_ca: true`` for the server.

- Admin users: configure an admin certificate provider, or set ``external_cert: true`` for certificate and key
  files supplied directly by IT.

For a file-based admin or machine account, use an account name in email-address form and install a certificate
whose common name matches it. The certificate supplies the organization and role.

You can generate configuration-only kits before enrollment. Install credentials at their destinations
separately, then start FLARE. Provisioning rejects a participant for which no credential source is configured.

What you no longer need and what remains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The provisioning host needs no CA connection, issuance credentials, CA adapter, or signing permission. It can
generate the kits offline from public trust and configuration. An enterprise root may be kept offline or use
an HSM or KMS under the PKI team’s controls; FLARE provisioning does not need access to that signing key.

The running components still need private keys. Each endpoint holds its own key, each user obtains or receives
an admin key, and the server holds the constrained job-CA key. Public-trust-only means provisioning no longer
handles those private keys; it does not mean FLARE runs without private keys or without a local job signer.

Distribute configuration and ``rootCA.pem`` over an authenticated channel, even though they are not secret.
Otherwise an attacker could replace the trust root.

To add a participant, regenerate kits into a new output directory and distribute only the new participant’s
kit. Existing installations need no redeployment unless their configuration must change.

These choices cover standard mutual TLS deployments. Signed Confidential Computing and homomorphic-encryption
startup kits need a separately supported content-signing path. Root rotation and multiple roots are outside
this guide’s scope.

Choose the model that fits your operating policy
------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 27 30 43

   * - IT requirement
     - Deployment choice
     - Combinations and dependencies
   * - No external certificate service
     - FLARE issues admin, endpoint and job-CA credentials.
     - Local per-job issuance is included. Adding SSO introduces an external sign-in and certificate service.
   * - Organizational sign-in for people
     - SSO-issued admin credentials.
     - Can retain FLARE-issued endpoint and job-CA certificates. The SSO issuer must chain to the same root.
       File-based admins can coexist.
   * - Enterprise ownership of server and site credentials
     - IT supplies endpoint files and manages replacement and restarts.
     - Compatible with SSO or file-based admins, and a FLARE-issued or externally supplied job CA, under the
       common root.
   * - Enterprise issuance of the job CA
     - IT supplies the job-CA certificate and key.
     - Independent of endpoint issuance and SSO. The server still issues job certificates. The issuing CA must
       permit a subordinate CA; pathlen:0 cannot.
   * - No root or participant private keys in provisioning
     - Public-trust-only kits.
     - Requires external endpoint and job-CA credentials. Admins use SSO or externally supplied files.
       Provisioning cannot issue credentials in this mode.
   * - Different roots for different credentials
     - Unsupported in this design.
     - Admin, endpoint and job chains must all reach the configured root, even when different issuers are
       used.
   * - Move an existing federation to an enterprise root
     - Coordinated trust and credential migration.
     - Avoid piecemeal replacement under an unrelated root. Automatic root rotation and multiple-root
       transition are outside this design.

You do not need to progress through every option. If your policy allows a federation-managed CA, the
self-contained model already provides per-job credentials. If policy prohibits CA private keys in application
provisioning, begin with public-trust-only provisioning and assign the credential owners before rollout. SSO
remains an independent choice in either design, subject to the common-root requirement.
