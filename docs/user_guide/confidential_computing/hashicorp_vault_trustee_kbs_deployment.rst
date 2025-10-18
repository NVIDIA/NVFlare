.. _hashicorp_vault_trustee_deployment:

#############################################################
HashiCorp Vault and Trustee KBS Joint Deployment Guide
#############################################################

Overview
========

This guide provides complete instructions for deploying HashiCorp Vault and Trustee KBS (Key Broker Service) as an integrated secret management system for Confidential Computing environments.

**Architecture:**

- **HashiCorp Vault**: Secure backend for storing secrets
- **Trustee KBS**: Frontend proxy for verifying client identities and brokering keys
- **Deployment Order**: Vault must be deployed first, then KBS

**What You'll Learn:**

- Understanding the deployment architecture and requirements
- Setting up HashiCorp Vault with proper TLS configuration
- Compiling and configuring Trustee KBS
- Testing the complete system with client operations
- Troubleshooting common issues

.. note::

   **TEE Environment Deployment Requirements**

   Before starting deployment, please understand the hardware environment requirements for each component to properly plan your deployment architecture.

Understanding the Architecture
===============================

Hardware Requirements
---------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 20 45

   * - Component
     - TEE Hardware
     - Deployment Location
     - Description
   * - HashiCorp Vault
     - ❌ Not Required
     - Regular Server
     - Secure storage of secret data, protected by software layers (encryption, access control, auditing)
   * - Trustee KBS
     - ❌ Not Required
     - Regular Server
     - Verifies client TEE evidence, acts as proxy between Vault and clients
   * - TEE Client
     - ✅ Required
     - TEE-enabled Device
     - Runs in trusted execution environment, generates hardware-based attestation evidence

Deployment Architecture
-----------------------

::

   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
   │   TEE Client    │───▶│   Trustee KBS   │───▶│ HashiCorp Vault │
   │ (TEE Hardware)  │    │ (Regular Server) │    │ (Regular Server) │
   └─────────────────┘    └─────────────────┘    └─────────────────┘
   │                 │    │                 │    │                 │
   │ Hardware:       │    │ Functions:      │    │ Functions:      │
   │ • Intel TDX     │    │ • Attestation   │    │ • Secret        │
   │ • AMD SEV       │    │   Verification  │    │   Storage       │
   │ • ARM TrustZone │    │ • Policy Engine │    │ • Access        │
   │ • TPM 2.0       │    │ • Key Broker    │    │   Control       │
   │                 │    │ • JWT Auth      │    │ • Audit Logs    │
   │                 │    │                 │    │ • Encrypted     │
   │                 │    │                 │    │   Transport     │
   └─────────────────┘    └─────────────────┘    └─────────────────┘

Environment Types
-----------------

**Test Environment** (covered in this guide):

- Vault and KBS deployed on regular servers
- Client uses "sample attester" to simulate TEE evidence
- Suitable for: functionality verification, development debugging, system integration testing

**Production Environment**:

- Vault and KBS still deployed on regular servers (data center)
- Clients **must** run on real TEE hardware
- Clients generate real hardware-based attestation evidence

Design Rationale
----------------

**Why not require TEE hardware for Vault and KBS?**

- **Security Separation**: Each component focuses on its specific responsibilities, reducing overall attack surface
- **Cost Optimization**: Use expensive TEE hardware only where needed (clients)
- **Flexible Deployment**: Vault and KBS can leverage mature data center management tools
- **Easy Maintenance**: Regular servers are easier to scale, monitor, and maintain

Deployment Phases
=================

This deployment consists of four phases:

1. **Environment Preparation** - Install required tools and dependencies
2. **Deploy HashiCorp Vault** - Set up the secure backend storage
3. **Deploy Trustee KBS** - Set up the attestation and key broker service
4. **Client Operations** - Test and verify the complete system

Phase 1: Environment Preparation
=================================

System Requirements
-------------------

**Operating System:**

- Ubuntu 22.04 or 24.04 (recommended)
- Debian-based distributions

**Required Tools:**

- Git
- Curl
- OpenSSL
- Build tools (gcc, clang)
- Protobuf compiler
- Rust (for KBS compilation)

Installation Steps
------------------

**1.1 Update System**

.. code-block:: bash

   sudo apt-get update
   sudo apt-get upgrade -y

**1.2 Install Basic Tools**

.. code-block:: bash

   sudo apt-get install -y git curl build-essential clang libtss2-dev openssl pkg-config protobuf-compiler

**1.3 Install Rust**

Rust is required for compiling Trustee KBS:

.. code-block:: bash

   curl https://sh.rustup.rs -sSf | sh
   source "$HOME/.cargo/env"

During installation, choose the default option (1).

Phase 2: Deploy HashiCorp Vault
================================

Vault serves as the secure backend for storing secrets. We'll configure it with TLS encryption and proper access controls.

2.1 Install Vault
-----------------

.. code-block:: bash

   wget -O - https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
   sudo apt update && sudo apt install vault

2.2 Create Directories
----------------------

.. code-block:: bash

   sudo mkdir -p /opt/vault/tls
   sudo mkdir -p /opt/vault/data

2.3 Generate TLS Certificates
------------------------------

**Option A: Self-Signed Certificate (Testing Only)**

For quick testing:

.. code-block:: bash

   sudo openssl req -x509 -newkey rsa:4096 -keyout /opt/vault/tls/vaultlocal.key \
     -out /opt/vault/tls/vaultlocal.crt -sha256 -days 365 -nodes \
     -subj "/CN=localhost"

.. warning::

   Self-signed certificates are for testing only. Production environments should use certificates from a trusted Certificate Authority.

**Option B: CA-Signed Certificate (Recommended)**

For proper TLS validation:

**Step 1: Generate Local CA**

.. code-block:: bash

   sudo openssl genrsa -out /opt/vault/tls/ca.key 4096
   sudo openssl req -x509 -new -key /opt/vault/tls/ca.key -sha256 -days 3650 \
     -subj "/CN=Local Test CA" \
     -addext "basicConstraints=critical,CA:true,pathlen:0" \
     -addext "keyUsage=critical,keyCertSign,cRLSign" \
     -out /opt/vault/tls/ca.crt

**Step 2: Generate Server Private Key**

.. code-block:: bash

   sudo openssl genrsa -out /opt/vault/tls/vault.key 2048

**Step 3: Create Certificate Signing Request (CSR)**

.. code-block:: bash

   sudo openssl req -new -key /opt/vault/tls/vault.key \
     -subj "/CN=localhost" -out /opt/vault/tls/vault.csr

**Step 4: Create SAN Configuration**

Replace the IP addresses with your actual server addresses:

.. code-block:: bash

   sudo tee /opt/vault/tls/san.cnf >/dev/null <<'EOF'
   basicConstraints=CA:false
   keyUsage=critical,digitalSignature,keyEncipherment
   extendedKeyUsage=serverAuth
   subjectAltName=DNS:localhost,IP:127.0.0.1,IP:10.176.193.230
   EOF

**Step 5: Sign Certificate**

.. code-block:: bash

   sudo openssl x509 -req -in /opt/vault/tls/vault.csr \
     -CA /opt/vault/tls/ca.crt -CAkey /opt/vault/tls/ca.key -CAcreateserial \
     -out /opt/vault/tls/vault.crt -days 825 -sha256 -extfile /opt/vault/tls/san.cnf

**Step 6: Verify Certificate**

.. code-block:: bash

   sudo openssl x509 -in /opt/vault/tls/vault.crt -noout -text | \
     sed -n '/Subject:/p;/Subject Alternative Name/,+1p;/Extended Key Usage/,+1p;/Basic Constraints/,+1p'

You should see:
- ``CA:FALSE``
- ``Extended Key Usage: TLS Web Server Authentication``
- ``Subject Alternative Name`` with your DNS/IP entries

**Step 7: Set Permissions**

.. code-block:: bash

   sudo chown -R vault:vault /opt/vault/tls
   sudo chmod 750 /opt/vault/tls
   sudo chmod 640 /opt/vault/tls/vault.key
   sudo chmod 644 /opt/vault/tls/vault.crt /opt/vault/tls/ca.crt
   sudo chmod 755 /opt /opt/vault

2.4 Configure Vault
-------------------

Edit the configuration file:

.. code-block:: bash

   sudo nano /etc/vault.d/vault.hcl

Replace with the following content (update the IP/hostname):

.. code-block:: json

   {
     "ui": true,
     "api_addr": "https://<your-server-IP-or-hostname>:8200", // Example URL
     "storage": {
       "file": {
         "path": "/opt/vault/data"
       }
     },
     "listener": {
       "tcp": {
         "address": "<your-server-IP-or-hostname>:8200", // Example address
         "tls_cert_file": "/opt/vault/tls/vaultlocal.crt",
         "tls_key_file": "/opt/vault/tls/vaultlocal.key"
       }
     }
   }

.. note::

   Replace ``<your-server-IP-or-hostname>`` with your actual server address.

2.5 Start Vault Service
------------------------

.. code-block:: bash

   sudo systemctl restart vault
   sudo systemctl enable vault

2.6 Verify Vault Installation
------------------------------

**Method 1: Check Service Status**

.. code-block:: bash

   sudo systemctl status vault

Look for green "active (running)" text.

**Method 2: Check Network Port**

.. code-block:: bash

   sudo netstat -tuln | grep 8200

You should see the system listening on port 8200.

**Method 3: Test HTTPS Endpoint**

.. code-block:: bash

   curl --cacert /opt/vault/tls/ca.crt https://<your-server-IP-or-hostname>:8200/v1/sys/health

**Method 4: Access Web UI**

Open ``https://<your-server-IP>:8200`` in your browser. You should see the Vault initialization page.

2.7 Initialize Vault
--------------------

**Access the Web UI** and follow these steps:

**Step 1: Initialize**

On the initialization page, configure:

- **Key shares**: ``1`` (for testing; use higher values in production)
- **Key threshold**: ``1`` (for testing; use higher values in production)
- Leave "Store PGP keys" unchecked

Click "Initialize" and **save the Root Token and Recovery Key** securely!

**Step 2: Login**

Use the Root Token to log in to Vault.

**Step 3: Enable KV Engine**

1. Select "Secrets Engines" from the left menu
2. Click "Enable new engine +"
3. Select "KV"
4. Configure:
   - **Path**: ``kv`` (must match KBS configuration)
   - **Version**: Select ``1`` (KBS requires KV v1)
5. Click "Enable Engine"

.. note::

   This path must be mounted as KV v1 engine; KBS currently uses kv1 API

.. important::

   **If you previously enabled KV v2:**

   1. In "Secrets Engines", find the ``kv`` mount
   2. Click "⋯" menu and select "Disable"
   3. Re-enable with Version 1 as described above
   4. Verify the engine page shows "Version: 1"

✅ **Vault deployment complete!**

Phase 3: Deploy Trustee KBS
============================

KBS acts as the attestation proxy between clients and Vault. We'll compile it from source and configure it to connect to Vault.

3.1 Clone Repository
--------------------

.. code-block:: bash

   git clone https://github.com/confidential-containers/trustee.git
   cd trustee/kbs
   git checkout a2570329cc33daf9ca16370a1948b5379bb17fbe

3.2 Compile KBS
---------------

**Compile KBS Server** (with Vault support):

.. code-block:: bash

   sudo cargo install --path . --features="vault"

.. important::

   The ``--features="vault"`` flag is required for Vault integration.

**Compile KBS Client** (with sample attester for testing):

.. code-block:: bash

   make cli CLI_FEATURES=sample_only
   sudo make install-cli

.. note::

   The ``sample_only`` feature enables testing in non-TEE environments.

3.3 Troubleshoot Compilation Issues
------------------------------------

**Issue: Compilation error "error[E0277]: can't compare"**

This is a type mismatch in the verifier dependency.

**Solution:**

1. Open ``deps/verifier/src/az_snp_vtpm/mod.rs``
2. Find line ~225:

   .. code-block:: rust

      && get_oid_octets::<64>(&parsed_endorsement_key, HW_ID_OID)? != report.chip_id

3. Add dereference operator:

   .. code-block:: rust

      && get_oid_octets::<64>(&parsed_endorsement_key, HW_ID_OID)? != *report.chip_id

4. Recompile:

   .. code-block:: bash

      sudo cargo install --path . --features="vault"

**Issue: Runtime error "unknown variant 'Vault'"**

This means the system is running an old KBS binary.

**Solution:**

1. Find the correct KBS path:

   .. code-block:: bash

      which kbs

2. Use absolute path when starting KBS:

   .. code-block:: bash

      sudo /home/user/.cargo/bin/kbs --config-file ./kbs-config.toml

3. **Optional**: Create permanent link:

   .. code-block:: bash

      sudo ln -sf /home/user/.cargo/bin/kbs /usr/local/bin/kbs

3.4 Generate KBS Certificates and Keys
---------------------------------------

Create directories:

.. code-block:: bash

   mkdir -p keys wkdir admin

**Generate HTTPS Certificates**

**Step 1: Generate KBS CA**

.. code-block:: bash

   openssl genrsa -out keys/kbs-ca.key 4096
   openssl req -x509 -new -key keys/kbs-ca.key -sha256 -days 3650 \
     -subj "/CN=KBS Local CA" \
     -addext "basicConstraints=critical,CA:true,pathlen:0" \
     -addext "keyUsage=critical,keyCertSign,cRLSign" \
     -out keys/kbs-ca.crt

**Step 2: Generate Server Key and CSR**

.. code-block:: bash

   openssl genrsa -out keys/key.pem 2048
   openssl req -new -key keys/key.pem -subj "/CN=localhost" -out keys/kbs.csr

**Step 3: Create SAN Configuration**

.. code-block:: bash

   tee keys/kbs-san.cnf >/dev/null <<'EOF'
   basicConstraints=CA:false
   keyUsage=critical,digitalSignature,keyEncipherment
   extendedKeyUsage=serverAuth
   subjectAltName=DNS:localhost,IP:127.0.0.1
   EOF

**Step 4: Sign Server Certificate**

.. code-block:: bash

   openssl x509 -req -in keys/kbs.csr \
     -CA keys/kbs-ca.crt -CAkey keys/kbs-ca.key -CAcreateserial \
     -out keys/cert.pem -days 825 -sha256 -extfile keys/kbs-san.cnf

**Step 5: Verify Certificate**

.. code-block:: bash

   openssl x509 -in keys/cert.pem -noout -text | \
     sed -n '/Subject:/p;/Subject Alternative Name/,+1p;/Extended Key Usage/,+1p;/Basic Constraints/,+1p'

**Step 6: Client Trust Setup**

Clients must trust the KBS CA. Choose one method:

**Option A: Explicit CA File** (for kbs-client):

.. code-block:: bash

   kbs-client --cert-file ./keys/kbs-ca.crt ...

**Option B: System CA Store** (recommended for services):

.. code-block:: bash

   sudo cp ./keys/kbs-ca.crt /usr/local/share/ca-certificates/kbs-ca.crt
   sudo update-ca-certificates

**Option C: Container Mount**:

Mount the file and set ``SSL_CERT_FILE=/etc/ssl/certs/kbs-ca.crt``

**Generate Admin Authentication Keys**

.. code-block:: bash

   openssl genpkey -algorithm Ed25519 -out admin/admin.key
   openssl pkey -in admin/admin.key -pubout -out admin/admin.pub

.. important::

   KBS admin API requires Ed25519 keys. RSA keys will cause "Invalid public key" errors.

3.5 Configure KBS
-----------------

Create ``kbs-config.toml`` in the ``kbs`` directory:

.. code-block:: toml

   [http_server]
   sockets = ["0.0.0.0:8999"]
   insecure_http = false
   private_key = "./keys/key.pem"
   certificate = "./keys/cert.pem"

   [admin]
   auth_public_key = "./admin/admin.pub"

   [attestation_token]
   insecure_key = true

   [attestation_service]
   type = "coco_as_builtin"
   work_dir = "./wkdir/attestation-service"
   policy_engine = "opa"

   [attestation_service.attestation_token_broker]
   type = "Ear"
   duration_min = 5

   [attestation_service.rvps_config]
   type = "BuiltIn"

   [attestation_service.rvps_config.storage]
   type = "LocalJson"
   file_path = "./wkdir/attestation-service/reference_values.json"

   [policy_engine]
   policy_path = "./wkdir/policy.rego"

   [[plugins]]
   name = "resource"
   type = "Vault"
   # Replace with your Vault address
   vault_url = "https://<your-vault-host>:8200"
   # Replace with your Root Token from Vault initialization
   token = "hvs.xxxxnnnnxxxxnnnn"
   # Must match the path configured in Vault (kv)
   mount_path = "kv"
   # Set to false for self-signed certificates
   verify_ssl = false
   # If verify_ssl=true with self-signed certs, provide CA path:
   # ca_certs = ["./wkdir/local-ca.pem"]

.. note::

   **Important Configuration Notes:**

   - Replace ``vault_url`` with your actual Vault address
   - Replace ``token`` with the Root Token from Vault initialization
   - ``mount_path = "kv"`` must match the KV engine path in Vault
   - KBS requires KV v1 (not v2)
   - If using self-signed Vault certificates, set ``verify_ssl = false``

3.6 Configure Attestation Policy
---------------------------------

For **testing in non-TEE environments**, use a permissive policy:

.. code-block:: bash

   cp ./sample_policies/allow_all.rego ./wkdir/policy.rego

.. warning::

   In production, use strict attestation policies that verify real TEE evidence. The ``allow_all`` policy is only for testing.

3.7 Start KBS Service
---------------------

Start KBS using the absolute path:

.. code-block:: bash

   sudo /home/user/.cargo/bin/kbs --config-file ./kbs-config.toml

If successful, you should see output indicating KBS is listening on port 8999.

✅ **KBS deployment complete!**

Phase 4: Client Operations and Verification
============================================

Now test the complete system by storing and retrieving secrets.

4.1 Locate KBS Client
---------------------

The compiled ``kbs-client`` is located at ``trustee/target/release/kbs-client``.

If your project is in ``/home/user/trustee``, the full path is:

.. code-block:: text

   /home/user/trustee/target/release/kbs-client

4.2 Store a Secret
------------------

**Step 1: Create Test Data**

.. code-block:: bash

   echo "this is a test file." > test.txt

**Step 2: Store in Vault via KBS**

.. code-block:: bash

   /path/to/target/release/kbs-client --url https://<trustee-service-host>:8999 \
     --cert-file ./keys/kbs-ca.crt \
     config --auth-private-key ./admin/admin.key \
     set-resource --path mysecrets/database/password \
     --resource-file test.txt

4.3 Retrieve a Secret
---------------------

**Step 1: Generate TEE Private Key** (for client simulation):

.. code-block:: bash

   openssl ecparam -name prime256v1 -genkey -noout | \
     openssl pkcs8 -topk8 -nocrypt -out tee_ec.key

**Step 2: Retrieve Secret** (with automatic attestation):

.. code-block:: bash

   /path/to/target/release/kbs-client --url https://<trustee-server-host>:8999 \
     --cert-file ./keys/kbs-ca.crt \
     get-resource --path mysecrets/database/password \
     --tee-key-file ./tee_ec.key

**Expected Behavior:**

- In non-TEE environments, you'll see: "Sample Attester will be used" (this is normal)
- On success, the output will be base64 encoded
- Decode the output: ``echo "base64output" | base64 -d``

✅ **Congratulations!** You have successfully deployed and tested the secret management system.

Troubleshooting
===============

Common Issues and Solutions
---------------------------

**Issue 1: "illegal token format" Error**

**Symptoms:**

.. code-block:: text

   Error: read token
   Caused by: illegal token format

**Root Cause:** KBS client doesn't have ``sample_only`` feature enabled.

**Solution:**

Recompile kbs-client with the feature:

.. code-block:: bash

   make -C trustee/kbs cli CLI_FEATURES=sample_only
   /path/to/trustee/target/release/kbs-client [parameters...]

**Issue 2: "Access denied by policy" Error**

**Symptoms:**

.. code-block:: text

   Error: request unauthorized
   ErrorInformation { error_type: "PolicyDeny", detail: "Access denied by policy" }

**Root Cause:** KBS policy rejects sample evidence.

**Solution:**

Update to permissive policy:

.. code-block:: bash

   cp ./sample_policies/allow_all.rego ./wkdir/policy.rego

Or via admin API:

.. code-block:: bash

   kbs-client --url https://<trustee-service-host>:8999 \
     --cert-file ./keys/kbs-ca.crt \
     config --auth-private-key ./admin/admin.key \
     set-attestation-policy --policy-file ./sample_policies/allow_all.rego

**Issue 3: Vault TLS Certificate Error**

**Symptoms:** "CaUsedAsEndEntity" error or Vault connection fails.

**Root Cause:** Vault using non-compliant certificates (CA cert used as server cert).

**Solution:** Follow Phase 2, Section 2.3 Option B to generate proper server certificates.

**Issue 4: KV Engine Version Mismatch**

**Symptoms:**

.. code-block:: text

   Invalid path for a versioned K/V secrets engine

**Root Cause:** Vault has KV v2 engine, but KBS uses KV v1 API.

**Solution:**

1. In Vault UI, go to "Secrets Engines"
2. Find the ``kv`` mount, click "⋯" → "Disable"
3. Re-enable with Version 1 (see Phase 2, Section 2.7, Step 3)

**Issue 5: RVPS Storage Permission Error**

**Symptoms:**

.. code-block:: text

   Permission denied (os error 13)

**Root Cause:** LocalJson storage tries to write to system directories without permissions.

**Solution:**

Add writable path in ``kbs-config.toml``:

.. code-block:: toml

   [attestation_service.rvps_config.storage]
   type = "LocalJson"
   file_path = "./wkdir/attestation-service/reference_values.json"

Issue 6: Normal Warning Messages in Test Environment

Symptoms: When testing in non-TEE environments, client outputs the following warning messages:

.. code-block:: text

   [WARN] No TEE platform detected. Sample Attester will be used.
   [WARN] Authenticating with KBS failed. Perform a new RCAR handshake: TokenNotFound

Explanation: These are normal warning messages, not errors:

- "No TEE platform detected":

  - Expected behavior when testing on regular servers
  - System automatically switches to sample attester to simulate TEE evidence
  - This is exactly what we expect in test environments

These are **normal** in test environments:

- "No TEE platform detected": Expected on regular servers, system uses sample attester
- "TokenNotFound": Normal on first access, system performs new RCAR handshake

**Confirmation of Success:**

- Check if you received base64 encoded output
- Decode and verify: ``echo "base64content" | base64 -d``
- These warnings are expected and indicate proper test environment behavior

Appendix
========

Building KBS Docker Image
--------------------------

**Prerequisites**

The default Dockerfile at commit ``a2570329cc33daf9ca16370a1948b5379bb17fbe`` has issues. Apply this patch first:

**Patch File:**

.. code-block:: diff

   diff --git a/kbs/docker/Dockerfile b/kbs/docker/Dockerfile
   index e529716..45b9271 100644
   --- a/kbs/docker/Dockerfile
   +++ b/kbs/docker/Dockerfile
   @@ -39,17 +39,17 @@ RUN if [ "${ARCH}" = "x86_64" ]; then curl -fsSL https://download.01.org/intel-s
    WORKDIR /usr/src/trustee
    COPY . .
    
   -RUN cd kbs && make AS_FEATURE=coco-as-builtin ALIYUN=${ALIYUN} ARCH=${ARCH} && \
   +RUN cd kbs && make VAULT=true AS_FEATURE=coco-as-builtin ALIYUN=${ALIYUN} ARCH=${ARCH} background-check-kbs && \
       make ARCH=${ARCH} install-kbs
    
   -FROM ubuntu:22.04
   +FROM ubuntu:24.04
    ARG ARCH=x86_64
    
    WORKDIR /tmp
    
    RUN apt-get update && \
       apt-get install -y \
   -    curl \
   +    curl gpg \
       gnupg-agent && \
       if [ "${ARCH}" = "x86_64" ]; then curl -fsSL https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key | \
       gpg --dearmor --output /usr/share/keyrings/intel-sgx.gpg && \

**Build Command:**

Inside the ``trustee`` folder:

.. code-block:: bash

   docker build -f kbs/docker/Dockerfile -t kbs:latest .

**Run Docker Container:**

.. code-block:: bash

   docker run -p 8999:8999 kbs:latest

Summary
=======

You have successfully:

✅ Deployed HashiCorp Vault as a secure secret backend
✅ Compiled and configured Trustee KBS with Vault integration
✅ Set up proper TLS certificates for both services
✅ Tested the system with client operations
✅ Learned how to troubleshoot common issues

**Next Steps:**

- For production deployment, use real TEE hardware for clients
- Implement strict attestation policies
- Use certificates from a trusted CA
- Configure proper access controls and audit logging
- Review the :ref:`NVFlare CC Architecture <cc_architecture>` for integration with NVFlare
