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

Deployment Architecture
-----------------------

::

   ┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
   │   TEE Client    │───▶│   Trustee KBS    │───▶│ HashiCorp Vault  │
   │ (TEE Hardware)  │    │                  │    │                  │
   └─────────────────┘    └──────────────────┘    └──────────────────┘
   │                 │    │                  │    │                  │
   │ Hardware:       │    │ Functions:       │    │ Functions:       │
   │ • Intel TDX     │    │ • Attestation    │    │ • Secret         │
   │ • AMD SEV       │    │   Verification   │    │   Storage        │
   │ • ARM TrustZone │    │ • Policy Engine  │    │ • Access         │
   │ • TPM 2.0       │    │ • Key Broker     │    │   Control        │
   │                 │    │ • JWT Auth       │    │ • Audit Logs     │
   │                 │    │                  │    │ • Encrypted      │
   │                 │    │                  │    │   Transport      │
   └─────────────────┘    └──────────────────┘    └──────────────────┘

Environment Types
-----------------

**Test Environment** (covered in this guide):

- Vault and KBS deployed on regular servers
- Client uses "sample attester" to simulate TEE evidence
- Suitable for: functionality verification, development debugging, system integration testing

**Production Environment**:

- Vault and KBS still deployed on secure environment (data center)
- Clients **must** run on real TEE hardware
- Clients generate real hardware-based attestation evidence

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

Install Vault
-------------

.. code-block:: bash

   wget -O - https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
   sudo apt update && sudo apt install vault

Create Vault certificate and data directories
---------------------------------------------

.. code-block:: bash

   sudo mkdir -p /opt/vault/tls
   sudo mkdir -p /opt/vault/data

Generate self-signed TLS certificates (for testing)
---------------------------------------------------

Execute the following commands to generate the vaultlocal.key and vaultlocal.crt files required by Vault:

.. code-block:: bash

   sudo openssl req -x509 -newkey rsa:4096 -keyout /opt/vault/tls/vaultlocal.key -out /opt/vault/tls/vaultlocal.crt -sha256 -days 365 -nodes -subj "/CN=localhost"

.. note::
   Production environments should use certificates issued by a trusted CA. Self-signed certificates generated by this command are for testing purposes only.

Configure Vault (/etc/vault.d/vault.hcl)
----------------------------------------

Use `sudo nano /etc/vault.d/vault.hcl` to edit the configuration file and replace with the following content:

.. code-block::

    {
      "ui": true,
      "api_addr": "https://<your-server-IP-or-hostname>:8200",  // Example URL
      "storage": {
        "file": {
          "path": "/opt/vault/data"
        }
      },
      "listener": {
        "tcp": {
          "address": "<your-server-IP-or-hostname>:8200",  // Example address
          "tls_cert_file": "/opt/vault/tls/vaultlocal.crt",
          "tls_key_file": "/opt/vault/tls/vaultlocal.key"
        }
      }
    }

Use CA-signed server certificates (for strict validation, recommended)
----------------------------------------------------------------------

If you need to enable strict TLS validation on the client side (such as KBS), do not directly use CA certificates as server certificates. Follow these steps to generate a "server certificate" signed by a local CA (must include SAN, CA:FALSE, and EKU includes serverAuth), then use this server certificate in Vault:

Generate local CA (only needed once)
------------------------------------

.. code-block:: bash

   sudo openssl genrsa -out /opt/vault/tls/ca.key 4096
   sudo openssl req -x509 -new -key /opt/vault/tls/ca.key -sha256 -days 3650 \
     -subj "/CN=Local Test CA" \
     -addext "basicConstraints=critical,CA:true,pathlen:0" \
     -addext "keyUsage=critical,keyCertSign,cRLSign" \
     -out /opt/vault/tls/ca.crt

Generate server certificate (with SAN, CA:FALSE + serverAuth)
-------------------------------------------------------------

.. code-block:: bash

   # Server private key
   sudo openssl genrsa -out /opt/vault/tls/vault.key 2048

   # Server CSR (non-interactive)
   sudo openssl req -new -key /opt/vault/tls/vault.key -subj "/CN=localhost" -out /opt/vault/tls/vault.csr

   # Write SAN configuration (replace IP/DNS with your actual address)
   sudo tee /opt/vault/tls/san.cnf >/dev/null <<'EOF'

   basicConstraints=CA:false
   keyUsage=critical,digitalSignature,keyEncipherment
   extendedKeyUsage=serverAuth
   subjectAltName=DNS:localhost,IP:127.0.0.1,IP:10.176.193.230
   EOF

Sign server certificate with CA (note: use ca.crt/ca.key generated in previous step)
------------------------------------------------------------------------------------

.. code-block:: bash

   sudo openssl x509 -req -in /opt/vault/tls/vault.csr \
   -CA /opt/vault/tls/ca.crt -CAkey /opt/vault/tls/ca.key -CAcreateserial \
   -out /opt/vault/tls/vault.crt -days 825 -sha256 -extfile /opt/vault/tls/san.cnf

Quick verification of certificate key extensions (should see CA:FALSE, serverAuth, and SAN list)
------------------------------------------------------------------------------------------------

.. code-block:: bash

   sudo openssl x509 -in /opt/vault/tls/vault.crt -noout -text \
   | sed -n '/Subject:/p;/Subject Alternative Name/,+1p;/Extended Key Usage/,+1p;/Basic Constraints/,+1p'

Fix Vault certificate file permissions and ownership (Vault runs as vault user)
-------------------------------------------------------------------------------

.. code-block:: bash

   # Directory and file ownership
   sudo chown -R vault:vault /opt/vault/tls
   # Directory and file permissions (directory traversable; private key readable only by owner; certificates readable)
   sudo chmod 750 /opt/vault/tls
   sudo chmod 640 /opt/vault/tls/vault.key
   sudo chmod 644 /opt/vault/tls/vault.crt /opt/vault/tls/ca.crt
   # If needed, ensure parent directories are traversable
   sudo chmod 755 /opt /opt/vault

Update Vault configuration and restart
--------------------------------------

Point the certificate paths in /etc/vault.d/vault.hcl to the new server certificate:

.. code-block::

   tls_cert_file=/opt/vault/tls/vault.crt
   tls_key_file=/opt/vault/tls/vault.key

Then restart and check status:

.. code-block:: bash

   sudo systemctl restart vault
   sudo systemctl status vault | cat
   # Verify HTTPS:
   curl --cacert /opt/vault/tls/ca.crt https://<your-server-IP-or-hostname>:8200/v1/sys/health | cat

Start Vault service
-------------------

.. code-block:: bash

   sudo systemctl restart vault
   sudo systemctl enable vault # Set to start on boot

Verify Vault deployment success
-------------------------------

Before continuing, confirm that Vault service is running properly using the following methods:

Method 1: Check service status

.. code-block:: bash

   sudo systemctl status vault

If successful, you'll see green "active (running)" text.

Method 2: Check network port

.. code-block:: bash

   sudo netstat -tuln | grep 8200

If successful, you'll see the system listening on port 8200.

Method 3: Access Web UI (most intuitive)

Visit https://:8200 in your browser. If you can see Vault's initialization or login page, the deployment is completely successful.

Initialize and configure in Vault UI
------------------------------------

a. Initialize: When accessing the UI for the first time, you'll see the initialization interface. This is the core of Vault's security mechanism, used to generate the master key.

- **Key shares**: The total number of parts the master key is split into.
- **Key threshold**: The minimum number of key parts required to "unseal" Vault each time.

For the test environment in this guide, use the following simplest configuration:

- **Key shares**: 1
- **Key threshold**: 1
- **Store PGP keys**: Keep unchecked.

After clicking the "Initialize" button, the system will generate a Root Token and a Recovery Key. Please be sure to safely copy and save both values!

b. Login: On the page after initialization is complete, use the Root Token you just saved to log in.

c. Enable KV engine:

- Select "Secrets Engines" from the left menu.
- Click "Enable new engine +".
- Select "KV".
- On the configuration page:
  - **Path**: Enter kv (this must match the mount_path in subsequent KBS configuration).
  - **Version**: Select 1 (KBS currently only supports V1 version).
- Click "Enable Engine".

.. important::
   If you have previously enabled KV v2 in the UI, follow these steps to change to v1 (web operation):

   - Open the "Secrets Engines" list on the left, find the entry with mount path kv, click the "⋯" menu on the right and select "Disable" and confirm.
   - Click "Enable new engine +", select "KV", in the configuration page set: Path fill in kv, Version select 1, then click "Enable".
   - Enter the engine page, the upper right corner should show "Version: 1"; if it's still v2, repeat the above steps.

Phase 3: Deploy Trustee KBS (Key Broker Service)
================================================

After Vault is ready, we deploy KBS as the core proxy connecting clients and Vault.  Optionally, you can
build a docker image and run it directly.  To build docker images, please follow the Appendix.

Clone and checkout specific version of code
-------------------------------------------

.. code-block:: bash

   git clone https://github.com/confidential-containers/trustee.git
   cd trustee/kbs
   git checkout a2570329cc33daf9ca16370a1948b5379bb17fbe

Compile KBS (Important!)
------------------------

To ensure KBS can communicate with Vault, the vault feature must be enabled during compilation.

Compile and install KBS service

.. code-block:: bash

   sudo cargo install --path . --features="vault"

Compile KBS client tool (supports non-TEE environment testing)

.. note::
   In non-TEE environments, sample_only feature needs to be enabled to support sample attester

.. code-block:: bash

   make cli CLI_FEATURES=sample_only
   sudo make install-cli

Troubleshooting: Fix compilation and runtime errors
---------------------------------------------------

Issue 1: Compilation error "error[E0277]: can't compare"

This is caused by type mismatch in the internal code of kbs dependency library verifier. We need to manually modify this dependency library's source file to solve it.

a. Locate file: In the trustee directory, find and open this file: deps/verifier/src/az_snp_vtpm/mod.rs.

b. Modify code: Find the code around line 225, which looks like this:

.. code-block:: rust

   // Original code
   && get_oid_octets::<64>(&parsed_endorsement_key, HW_ID_OID)? != report.chip_id

According to the compiler's hint, add an asterisk * before report.chip_id for dereferencing, modified as follows:

.. code-block:: rust

   // Modified code
   && get_oid_octets::<64>(&parsed_endorsement_key, HW_ID_OID)? != *report.chip_id

c. Save file and recompile: After saving the file modification, return to trustee/kbs directory, re-execute the compilation command

.. code-block:: bash

   sudo cargo install --path . --features="vault"

Issue 2: After recompiling, starting KBS still reports error "unknown variant 'Vault'"

Cause: This usually means your system is running an old version of the kbs program, not the new version you just installed with cargo.

Diagnosis and solution:

a. Confirm the correct path of kbs under your current user:

.. code-block:: bash

   which kbs

This command will show the absolute path of the newly compiled kbs (e.g., /home/user/.cargo/bin/kbs).

b. Start using absolute path (recommended): Don't run sudo kbs ... directly, but use the absolute path obtained in the previous step to start the new program:

Replace the path below with the real path you got in the previous step

.. code-block:: bash

   sudo /home/user/.cargo/bin/kbs --config-file ./kbs-config.toml

c. Permanent fix (optional): If you want to be able to use sudo kbs ... directly in the future, you can create a soft link.

Replace the source path below with the real path you found in step a

.. code-block:: bash

   sudo ln -sf /home/user/.cargo/bin/kbs /usr/local/bin/kbs

Generate various key files required by KBS (New)
------------------------------------------------

Before starting KBS, we need to generate HTTPS certificates and administrator authentication keys for it. Please execute in the trustee/kbs directory:

Create directories for storing keys

.. code-block:: bash

   mkdir -p keys wkdir admin

1. Generate KBS HTTPS certificate architecture (recommended CA-signed mode)

1.1) Generate KBS local CA (for signing server certificates)

.. code-block:: bash

   openssl genrsa -out keys/kbs-ca.key 4096
   openssl req -x509 -new -key keys/kbs-ca.key -sha256 -days 3650 \
   -subj "/CN=KBS Local CA" \
   -addext "basicConstraints=critical,CA:true,pathlen:0" \
   -addext "keyUsage=critical,keyCertSign,cRLSign" \
   -out keys/kbs-ca.crt

1.2) Generate KBS server certificate request

.. code-block:: bash

   openssl genrsa -out keys/key.pem 2048
   openssl req -new -key keys/key.pem -subj "/CN=localhost" -out keys/kbs.csr

1.3) Create server certificate extension configuration

.. code-block:: bash

   tee keys/kbs-san.cnf >/dev/null <<'EOF'
   basicConstraints=CA:false
   keyUsage=critical,digitalSignature,keyEncipherment
   extendedKeyUsage=serverAuth
   subjectAltName=DNS:localhost,IP:127.0.0.1
   EOF

1.4) Sign server certificate with KBS CA

.. code-block:: bash

   openssl x509 -req -in keys/kbs.csr \
   -CA keys/kbs-ca.crt -CAkey keys/kbs-ca.key -CAcreateserial \
   -out keys/cert.pem -days 825 -sha256 -extfile keys/kbs-san.cnf

1.5) Verify generated certificate

.. code-block:: bash

   openssl x509 -in keys/cert.pem -noout -text | \
   sed -n '/Subject:/p;/Subject Alternative Name/,+1p;/Extended Key Usage/,+1p;/Basic Constraints/,+1p'

1.6) Client trust setup (very important)

kbs-ca.crt (from step 1.1) is the CA root that signs KBS server cert.
Clients MUST trust this CA to connect to KBS via HTTPS.

Option A: pass explicitly to kbs-client

.. code-block:: bash

   --cert-file ./keys/kbs-ca.crt

Option B (recommended for services): install into system CA store (Ubuntu/Debian)

.. code-block:: bash

   sudo cp ./keys/kbs-ca.crt /usr/local/share/ca-certificates/kbs-ca.crt
   sudo update-ca-certificates

Option C (containers): mount file and set env SSL_CERT_FILE=/etc/ssl/certs/kbs-ca.crt

2. Generate administrator authentication key pair (Ed25519)

.. note::
   KBS admin API only accepts Ed25519 public keys for verifying JWT signatures

.. code-block:: bash

   openssl genpkey -algorithm Ed25519 -out admin/admin.key
   openssl pkey -in admin/admin.key -pubout -out admin/admin.pub

.. note::
   Please use the Ed25519 algorithm key pair generated above; RSA public keys will cause KBS to report error "Invalid public key".

Prepare KBS configuration file (kbs-config.toml)
------------------------------------------------

Create a file named kbs-config.toml in the kbs directory and fill in the following content.

.. code-block::

   [http_server]
   sockets = ["0.0.0.0:8999"]
   insecure_http = false
   private_key = "./keys/key.pem"
   certificate = "./keys/cert.pem"

   [admin]
   auth_public_key = "./admin/admin.pub"
   ... (other attestation_service, policy_engine configurations remain unchanged) ...

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

   [policy_engine]
   policy_path = "./wkdir/policy.rego"

   [[plugins]]
   name = "resource"
   type = "Vault"
   Fill in your deployed Vault address
   vault_url = "https://:8200"
   Fill in the root token you obtained during Vault initialization
   token = "hvs.xxxxnnnnxxxxnnnn"
   Must match the path configured in Vault
   mount_path = "kv"

.. note::
   This path must be mounted as KV v1 engine; KBS currently uses kv1 API

   If Vault uses self-signed certificates, set this to false
   verify_ssl = false

   If verify_ssl is true and using self-signed certificates, uncomment and provide CA certificate path
   ca_certs = ["./wkdir/local-ca.pem"]

.. note::
   Please replace vault_url and token with your actual information.

   If encountering "Permission denied" error, add to [attestation_service.rvps_config.storage] section:

   file_path = "./wkdir/attestation-service/reference_values.json"

Start KBS service
-----------------

Recommend using absolute path to start, ensuring the correct version is running

.. code-block:: bash

   sudo /home/user/.cargo/bin/ls

If the terminal shows no errors and displays that the service is listening on port 8999, then KBS has started successfully.

Configure attestation policy (required for non-TEE environments)
----------------------------------------------------------------

When testing in non-TEE environments, you need to configure a permissive attestation policy to allow sample attester to pass verification.

Method 1: Directly replace policy file (recommended)

.. code-block:: bash

   cp ./sample_policies/allow_all.rego ./wkdir/policy.rego

Method 2: Set via admin API (optional)

.. code-block:: bash

   /path/to/target/release/kbs-client --url https://<trustee-service-host>:8999 \
   --cert-file ./keys/kbs-ca.crt \
   config --auth-private-key ./admin/admin.key \
   set-attestation-policy --policy-file ./sample_policies/allow_all.rego

.. note::
   In production environments, strict attestation policies should be used to verify real TEE evidence. Permissive policies are only suitable for testing and development environments.

Phase 4: Client Operations and Verification
===========================================

Now the entire system is ready, and you can use kbs-client to test secret storage and retrieval.

.. note::
   The compiled kbs-client is located at trustee/target/release/kbs-client. If your project is in /home/user/trustee directory, the full path would be /home/user/trustee/target/release/kbs-client.

Store a secret
--------------

First, create a test file, for example test.txt:

.. code-block:: bash

   echo "this is a test file." > test.txt

Execute the following command to store the file content in Vault (admin operation):

.. code-block:: bash

   /path/to/target/release/kbs-client --url https://<trustee-service-host>:8999 \
   --cert-file ./keys/kbs-ca.crt \
   config --auth-private-key ./admin/admin.key \
   set-resource --path mysecrets/database/password \
   --resource-file test.txt

Retrieve a secret (remote attestation operation)
------------------------------------------------

First generate TEE private key (for simulating client):

.. code-block:: bash

   openssl ecparam -name prime256v1 -genkey -noout | \
   openssl pkcs8 -topk8 -nocrypt -out tee_ec.key

Retrieve secret (client will automatically execute attestation process):

.. code-block:: bash

   /path/to/target/release/kbs-client --url https://<trustee-server-host>:8999 \
   --cert-file ./keys/kbs-ca.crt \
   get-resource --path mysecrets/database/password \
   --tee-key-file ./tee_ec.key

.. note::
   Use compiled kbs-client: /path/to/target/release/kbs-client (replace with actual path)

   In non-TEE environments, you'll see "Sample Attester will be used" warning, which is normal

   On success, the command will output base64 encoded content, decode with echo "result" | base64 -d

Congratulations! You have successfully deployed and tested the secret management system consisting of HashiCorp Vault and Trustee KBS.

Troubleshooting
===============

Issue 1: get-resource fails with error "illegal token format"

Symptoms: Client executing get-resource reports error:

.. code-block::

   Error: read token
   Caused by: illegal token format

Root cause: In non-TEE environments, kbs-client doesn't have sample_only feature enabled, cannot generate valid attestation token.

Solution:

Recompile kbs-client with sample_only feature enabled:

.. code-block:: bash

   make -C trustee/kbs cli CLI_FEATURES=sample_only

Use the newly compiled client:

.. code-block:: bash

   /path/to/trustee/target/release/kbs-client [other parameters...]

Issue 2: Attestation fails with error "Access denied by policy"

Symptoms: Client reports error:

.. code-block::

   Error: request unauthorized
   ...ErrorInformation { error_type: "PolicyDeny", detail: "Access denied by policy" }

Root cause: KBS's default policy rejects sample evidence, only accepts real TEE evidence.

Solution:

Update policy file to permissive policy:

.. code-block:: bash

   cp ./sample_policies/allow_all.rego ./wkdir/policy.rego

Or set via admin API:

.. code-block:: bash

   kbs-client --url https://<trustee-service-host>:8999 \
     --cert-file ./keys/kbs-ca.crt \
     config --auth-private-key ./admin/admin.key \
     set-attestation-policy --policy-file ./sample_policies/allow_all.rego

Issue 3: Vault TLS certificate error

Symptoms: KBS startup reports error "CaUsedAsEndEntity" or Vault connection fails.

Root cause: Vault is using non-compliant certificates (CA certificate used as server certificate).

Solution: Refer to Phase 2 Step 3 in the documentation to generate correct server certificates.

Issue 4: KV engine version mismatch

Symptoms: set-resource reports error "Invalid path for a versioned K/V secrets engine".

Root cause: Vault has mounted KV v2 engine, but KBS uses kv1 API.

Solution: In Vault UI, disable existing KV engine and re-enable as v1 version.

Issue 5: RVPS storage permission error

Symptoms: KBS startup reports error "Permission denied (os error 13)", usually involving /opt/confidential-containers/attestation-service/ path.

Root cause: Built-in RVPS uses LocalJson storage, defaults to writing to system directories where regular users don't have write permissions.

Solution: Add writable path to [attestation_service.rvps_config.storage] section in kbs-config.toml:

.. code-block:: toml

   [attestation_service.rvps_config.storage]
   type = "LocalJson"
   file_path = "./wkdir/attestation-service/reference_values.json"

Issue 6: Normal Warning Messages in Test Environment

Symptoms: When testing in non-TEE environments, client outputs the following warning messages:

.. code-block::

   [WARN] No TEE platform detected. Sample Attester will be used.
   [WARN] Authenticating with KBS failed. Perform a new RCAR handshake: TokenNotFound

Explanation: These are normal warning messages, not errors:

- "No TEE platform detected":

  - Expected behavior when testing on regular servers
  - System automatically switches to sample attester to simulate TEE evidence
  - This is exactly what we expect in test environments

- "TokenNotFound" / "Perform a new RCAR handshake":

  - Normal authentication flow on first access
  - Client doesn't have cached attestation token
  - System automatically performs new RCAR (Relying Party Attestation Capabilities and Resource) handshake

How to confirm successful operation:

- Check final output: if you see base64 encoded secret content, operation succeeded
- Use echo "base64content" | base64 -d to decode and verify content correctness
- In test environments, these warning messages are completely normal and expected

Appendix
========

Build KBS docker images
-----------------------


You can build docker images for kbs based on the Dockerfile in the kbs/docker folder.
However, that file in the current trustee repo at commit id a2570329cc33daf9ca16370a1948b5379bb17fbe
either fails to build or produces docker images with missing dependencies.
You can patch that file with the following diff.

.. code-block:: diff

   $ git diff
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

To build the kbs docker image, run the following inside trustee folder

.. code-block:: bash

   docker build -f kbs/docker/Dockerfile .

You can run KBS inside a Docker container with ports exposed using the -p option. For example:

.. code-block:: bash

   docker run -p 8080:8080 <image_name>