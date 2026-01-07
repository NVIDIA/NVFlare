.. _enrollment_usage:

#################################
FLARE Enrollment Usage Guide
#################################

This guide shows how to set up NVIDIA FLARE with certificate enrollment for
secure federated learning. Choose the workflow that matches your scale.

***************
Quick Reference
***************

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Task
     - Small Scale (< 10 sites)
     - Large Scale (10+ sites)
   * - **Workflow**
     - Manual
     - Auto-Scale
   * - **Certificate Authority**
     - Local (your machine)
     - Certificate Service
   * - **Key Commands**
     - ``nvflare cert``, ``nvflare package``
     - ``nvflare token``, ``nvflare package``, ``nvflare enrollment``
   * - **Best For**
     - Research, POC, security-restricted environments
     - Production, Kubernetes

**Command Overview:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Command
     - Description
   * - ``nvflare cert init``
     - Create a root Certificate Authority (CA) for signing certificates
   * - ``nvflare cert site``
     - Generate signed certificate for a server, client, relay, or admin
   * - ``nvflare cert api-key``
     - Generate an API key for authenticating with Certificate Service
   * - ``nvflare token generate``
     - Create a single enrollment token for a site to use during auto-enrollment
   * - ``nvflare token batch``
     - Create multiple enrollment tokens at once (for bulk deployments)
   * - ``nvflare token info``
     - Decode and display token contents (expiry, subject, etc.)
   * - ``nvflare package``
     - Generate a startup kit (config files, scripts) for a participant
   * - ``nvflare enrollment list``
     - View pending enrollment requests waiting for approval
   * - ``nvflare enrollment approve``
     - Approve one or more pending enrollment requests
   * - ``nvflare enrollment reject``
     - Reject a pending enrollment request with optional reason
   * - ``nvflare enrollment enrolled``
     - List all successfully enrolled participants

*************************
Workflow 1: Manual
*************************

Best for small deployments with 5-10 participants. Also suitable for
security-restricted environments where external Certificate Services are not permitted.

Step 1: Initialize Root CA (Project Admin)
==========================================

Create a root Certificate Authority:

.. code-block:: bash

    nvflare cert init -n "My Project" -o ./ca

This creates:

- ``./ca/rootCA.pem`` - Public certificate (distribute to everyone)
- ``./ca/rootCA.key`` - Private key (keep secure!)

Step 2: Generate Server Certificates (Project Admin)
=====================================================

.. code-block:: bash

    nvflare cert site -n server1 -t server -c ./ca \
        --host server.example.com

Step 3: Generate Client Certificates (Project Admin)
=====================================================

For each client site:

.. code-block:: bash

    nvflare cert site -n hospital-1 -t client -c ./ca

Step 4: Generate Startup Kits (Org Admin)
==============================================

Each org admin generates their own startup kit locally.
Project Admin sends them: server endpoint, and their certificates.

**Server:**

.. code-block:: bash

    nvflare package -n server1 -e grpc://0.0.0.0:8002:8003 -t server

**Clients:**

.. code-block:: bash

    nvflare package -n hospital-1 -e grpc://server.example.com:8002 -t client

Step 5: Distribute and Start
============================

**Project Admin:**

- Sends certificates (``rootCA.pem``, ``*.crt``, ``*.key``) to each site via secure channel

**Org Admin:**

1. Copy the received certificates to the ``startup/`` folder:

   - Server: ``rootCA.pem``, ``server.crt``, ``server.key``
   - Clients: ``rootCA.pem``, ``client.crt``, ``client.key``

2. Start:

   .. code-block:: bash

       cd <package-folder> && ./startup/start.sh

*************************
Workflow 2: Auto-Scale
*************************

Best for large deployments (10+ sites) or Kubernetes.

Step 1: Deploy Certificate Service (Project Admin)
===================================================

The Certificate Service handles certificate signing. Deploy it using Docker:

.. code-block:: bash

    docker run -d \
        -v /data/cert-service:/var/lib/cert_service \
        -p 8443:8443 \
        nvflare/cert-service:latest

Or via Kubernetes (see deployment docs).

Step 2: Generate API Key (Project Admin)
========================================

.. code-block:: bash

    nvflare cert api-key -o api_key.txt
    
    # Configure the Certificate Service with this key
    # (via config file or environment variable)

Step 3: Generate Tokens (Project Admin)
=======================================

Generate enrollment tokens for each site:

.. code-block:: bash

    # Single token
    nvflare token generate -n hospital-1 \
        --cert-service https://cert-service.example.com:8443 \
        --api-key $(cat api_key.txt)

    # Batch tokens (100 sites)
    nvflare token batch -n 100 --prefix site \
        --cert-service https://cert-service.example.com:8443 \
        --api-key $(cat api_key.txt) \
        -o tokens.csv

Project Admin sends to each site: server endpoint URL, Certificate Service URL,
and their unique enrollment token.

Step 4: Create Packages with Tokens (Org Admin)
====================================================

**Option A: Embed token in package (recommended)**

.. code-block:: bash

    nvflare package -n hospital-1 -e grpc://server:8002 -t client \
        --cert-service https://cert-service.example.com:8443 \
        --token "eyJhbGciOiJSUzI1NiIs..."

**Option B: Use environment variables**

.. code-block:: bash

    # Generate package without token
    nvflare package -n hospital-1 -e grpc://server:8002 -t client

    # Set environment at runtime
    export NVFLARE_CERT_SERVICE_URL=https://cert-service.example.com:8443
    export NVFLARE_ENROLLMENT_TOKEN="eyJhbGciOiJSUzI1NiIs..."

Step 5: Start Sites (Org Admin)
====================================

.. code-block:: bash

    cd hospital-1 && ./startup/start.sh

The client automatically:

1. Generates a private key locally
2. Sends a certificate request to the Certificate Service
3. Receives signed certificate
4. Connects to the FL server

*****************************************
Managing Enrollments (Project Admin)
*****************************************

View pending requests:

.. code-block:: bash

    nvflare enrollment list \
        --cert-service https://cert-service:8443 \
        --api-key $API_KEY

Approve requests:

.. code-block:: bash

    # Single approval
    nvflare enrollment approve hospital-1 --type client \
        --cert-service https://cert-service:8443 \
        --api-key $API_KEY

    # Bulk approval
    nvflare enrollment approve --pattern "hospital-*" --type client \
        --cert-service https://cert-service:8443 \
        --api-key $API_KEY

Reject requests:

.. code-block:: bash

    nvflare enrollment reject bad-site --type client \
        --reason "Not authorized" \
        --cert-service https://cert-service:8443 \
        --api-key $API_KEY

List enrolled entities:

.. code-block:: bash

    nvflare enrollment enrolled --type client \
        --cert-service https://cert-service:8443 \
        --api-key $API_KEY

*****************************************
Custom Code and Dependencies
*****************************************

This section shows how to package custom training code and dependencies alongside
the FLARE startup kit generated by ``nvflare package``.

Non-Containerized: pip + requirements.txt
=========================================

**Step 1: Generate startup kit**

.. code-block:: bash

    # Manual workflow
    nvflare package -n hospital-1 -e grpc://server:8002 -t client -w ./workspace
    
    # Auto-scale workflow (with token)
    nvflare package -n hospital-1 -e grpc://server:8002 -t client -w ./workspace \
        --cert-service https://cert-service:8443 \
        --token "$TOKEN"

**Step 2: Add custom code and dependencies**

.. code-block:: bash

    # Create local directory for custom code
    mkdir -p workspace/hospital-1/local/custom
    
    # Copy your custom trainer, model, etc.
    cp my_trainer.py workspace/hospital-1/local/custom/
    cp my_model.py workspace/hospital-1/local/custom/
    
    # Add dependencies file
    cp requirements.txt workspace/hospital-1/

Result structure:

.. code-block:: text

    workspace/hospital-1/
    ├── startup/                # From nvflare package
    │   ├── fed_client.json
    │   ├── start.sh
    │   ├── enrollment.json     # If --cert-service provided
    │   └── enrollment_token    # If --token provided
    ├── local/
    │   └── custom/
    │       ├── my_trainer.py   # Your custom code
    │       └── my_model.py
    └── requirements.txt        # Your dependencies

**Step 3: For Manual workflow, add certificates**

.. code-block:: bash

    # Copy certs from Project Admin
    cp rootCA.pem client.crt client.key workspace/hospital-1/startup/

**Step 4: Install and start**

.. code-block:: bash

    cd workspace/hospital-1
    pip install -r requirements.txt
    ./startup/start.sh

Containerized: Docker
=====================

**Step 1: Create Dockerfile with your dependencies**

.. code-block:: dockerfile

    FROM nvflare/nvflare:latest
    
    # Install your dependencies
    COPY requirements.txt /tmp/
    RUN pip install -r /tmp/requirements.txt
    
    # Copy custom code into image
    COPY custom/ /app/custom/

**Step 2: Build image**

.. code-block:: bash

    docker build -t myorg/flare-client:latest .
    docker push myorg/flare-client:latest   # If using registry

**Step 3: Generate startup kit**

.. code-block:: bash

    nvflare package -n hospital-1 -e grpc://server:8002 -t client -w ./workspace \
        --cert-service https://cert-service:8443 \
        --token "$TOKEN"

**Step 4: Run with Docker**

.. code-block:: bash

    docker run -v $(pwd)/workspace/hospital-1:/workspace \
        myorg/flare-client:latest \
        /bin/sh -c "cd /workspace && ./startup/start.sh"

Or use docker-compose:

.. code-block:: yaml

    # docker-compose.yml
    services:
      flare-client:
        image: myorg/flare-client:latest
        volumes:
          - ./workspace/hospital-1:/workspace
        working_dir: /workspace
        command: ./startup/start.sh

***************************
Kubernetes Deployment
***************************

For Kubernetes deployments with dynamic scaling:

Step 1: Build Custom Image (Optional)
=====================================

If you have custom code or dependencies, build a custom image:

.. code-block:: dockerfile

    FROM nvflare/nvflare:latest
    
    # Install custom dependencies
    COPY requirements.txt /tmp/
    RUN pip install -r /tmp/requirements.txt
    
    # Copy custom code (optional - can also mount via ConfigMap)
    COPY custom/ /app/custom/

.. code-block:: bash

    docker build -t myorg/flare-client:latest .
    docker push myorg/flare-client:latest

Step 2: Create Token Secret
===========================

.. code-block:: bash

    # Generate batch tokens
    nvflare token batch -n 100 --prefix site \
        --cert-service https://cert-service:8443 \
        --api-key $API_KEY \
        -o ./tokens/

    # Create Kubernetes secret
    kubectl create secret generic flare-tokens \
        --from-file=./tokens/ \
        -n flare

Step 3: Deploy StatefulSet
==========================

Use your custom image (or ``nvflare/nvflare:latest`` if no custom dependencies):

.. code-block:: yaml

    apiVersion: apps/v1
    kind: StatefulSet
    metadata:
      name: flare-client
    spec:
      replicas: 10
      template:
        spec:
          initContainers:
            - name: init
              # Use custom image with your dependencies
              image: myorg/flare-client:latest  # or nvflare/nvflare:latest
              command: ["/bin/sh", "-c"]
              args:
                - |
                  ORDINAL=${HOSTNAME##*-}
                  SITE_NAME=$(printf "site-%03d" $((ORDINAL + 1)))
                  nvflare package -n $SITE_NAME \
                      -e grpc://flare-server:8002 \
                      -t client \
                      -w /workspace
                  echo $SITE_NAME > /workspace/site_name.txt
              volumeMounts:
                - name: workspace
                  mountPath: /workspace
          containers:
            - name: flare
              # Use custom image with your dependencies
              image: myorg/flare-client:latest  # or nvflare/nvflare:latest
              command: ["/bin/sh", "-c"]
              args:
                - |
                  SITE_NAME=$(cat /workspace/site_name.txt)
                  export NVFLARE_ENROLLMENT_TOKEN=$(cat /tokens/${SITE_NAME}.token)
                  export NVFLARE_CERT_SERVICE_URL="https://cert-service:8443"
                  cd /workspace && ./startup/start.sh
              volumeMounts:
                - name: workspace
                  mountPath: /workspace
                - name: tokens
                  mountPath: /tokens
                  readOnly: true
          volumes:
            - name: workspace
              emptyDir: {}
            - name: tokens
              secret:
                secretName: flare-tokens

Step 4: Scale
=============

.. code-block:: bash

    kubectl scale statefulset flare-client --replicas=50 -n flare

*****************************
Environment Variables
*****************************

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Variable
     - Purpose
   * - ``NVFLARE_CERT_SERVICE_URL``
     - Certificate Service URL
   * - ``NVFLARE_ENROLLMENT_TOKEN``
     - JWT enrollment token
   * - ``NVFLARE_API_KEY``
     - Admin API key (for token/enrollment commands)
   * - ``NVFLARE_CA_PATH``
     - Path to CA directory (for local cert generation)

*****************************
Command Summary
*****************************

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Command
     - Purpose
   * - ``nvflare cert init``
     - Create root CA (Manual workflow)
   * - ``nvflare cert site``
     - Generate site certificates (Manual workflow)
   * - ``nvflare cert api-key``
     - Generate API key for Certificate Service
   * - ``nvflare token generate``
     - Generate enrollment token (Auto-Scale)
   * - ``nvflare token batch``
     - Generate multiple tokens (Auto-Scale)
   * - ``nvflare token info``
     - Inspect token contents
   * - ``nvflare package``
     - Generate startup kit
   * - ``nvflare enrollment list``
     - List pending requests
   * - ``nvflare enrollment approve``
     - Approve pending request(s)
   * - ``nvflare enrollment reject``
     - Reject pending request
   * - ``nvflare enrollment enrolled``
     - List enrolled entities

*****************************
Troubleshooting
*****************************

**"Certificate not found" at startup**

- Ensure ``rootCA.pem`` is in the ``startup/`` folder
- Check ``NVFLARE_CERT_SERVICE_URL`` is set correctly

**"Invalid token" error**

- Token may have expired (default: 7 days)
- Generate a new token with ``nvflare token generate``

**"Connection refused" to Certificate Service**

- Verify Certificate Service is running
- Check firewall/security group rules
- Verify URL includes correct port (e.g., ``:8443``)

**"API key required" error**

- Set ``NVFLARE_API_KEY`` environment variable
- Or use ``--api-key`` argument

**"Already enrolled" error**

- The site has already enrolled
- Each site can only enroll once per token

*****************************
Next Steps
*****************************

- :ref:`provisioning` - Traditional provisioning (for comparison)
- :ref:`enrollment_design_v2` - Technical design details
- :ref:`security` - Security best practices

