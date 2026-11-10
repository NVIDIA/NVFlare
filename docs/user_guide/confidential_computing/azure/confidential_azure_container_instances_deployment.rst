.. _confidential_azure_container_instances_deployment:

#######################################################
Confidential Azure Container Instances Deployment Guide
#######################################################

This workflow runs one NVFlare server in Azure Container Instances (ACI).
Clients run in confidential virtual machines or Confidential Containers. The
project administrator generates an enforcement policy for the fixed server
image, approves that policy by running provisioning, and directly launches ACI.

Attestation is signed evidence about the running protected environment. Its
``x-ms-sevsnpvm-hostdata`` field identifies the enforced policy using SHA-256,
a fingerprint calculated from the exact policy contents. Every ACI checker
requires that fingerprint to match the administrator-approved policy.

The server startup kit contains configuration and credentials. Deliver it
separately from the image: embedding the final kit and its fingerprint in the
image would change the image for which the policy was generated. A read-only
Azure secret volume supplies a signed archive, a single bundled file containing
the kit. The fixed image verifies the archive before using it.

.. important::

   This direct-launch workflow trusts authorized deployment operators and Azure's
   deployment service with the supplied credentials. Supply the startup archive
   through a secure parameter, an input whose value Azure excludes from retained
   deployment history. Keep the local parameters file private. Do not embed the
   archive in the template or give the secure parameter a secret default value.
   A digital signature detects changes; it does not encrypt the archive. Azure
   secret volumes are not
   attestation-controlled secret release. Use a separately designed encrypted
   delivery and key-release workflow if protection from the deployment service
   is required.

Prepare the fixed image and deployment key
==========================================

Use the files in ``examples/advanced/azure_cc``. Commands below run from that
example directory. Install Azure's ``az`` command-line tool, its ``confcom``
policy-generation extension, Docker, OpenSSL, and Python. Choose your resource
group, registry, region, participant names, and network settings. Azure resource
creation and deployment are administrator operations; provisioning does not
perform them.

A deployment-signing key pair consists of a private key used by provisioning
and a public key built into the server image. Create it before building the
image. These filenames match the example server configuration:

.. code-block:: bash

   umask 077
   openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:3072 \
     -out deployment-private-key.pem
   openssl pkey -in deployment-private-key.pem -pubout \
     -out deployment-public-key.pem

Place the exact NVFlare wheel containing this implementation in ``wheels/``.
A wheel is a Python installation package. Include any separately approved
optional dependency wheels required by your application. The supplied
``Dockerfile`` installs those files and copies only the public key into the
image. Its ``.dockerignore`` excludes the private key and provisioned files from
Docker's build context.

Build and publish the image, using your registry instead of the example value:

.. code-block:: bash

   registry=ccprepreg.azurecr.io
   server_image="$registry/cc_prep:0.0.1"
   az login
   az acr login --name ccprepreg
   docker build --platform linux/amd64 -t "$server_image" .
   docker push "$server_image"

Also publish the attestation helper image from
`Microsoft's confidential sidecar containers <https://github.com/microsoft/confidential-sidecar-containers>`_.
The sample template uses ``skr:2.7`` with its attestation listener on port 8284,
which matches NVFlare's ACI token generator. If using another helper version,
ensure its listener matches before generating the policy. Keep that image and
its command fixed as well.

The server image must contain its public verification key at
``/opt/nvflare-trust/deployment-public-key.pem``. The fixed launch command uses
Python's isolated mode, ``-I``, which excludes environment-provided import paths:

.. code-block:: text

   /usr/local/bin/python3 -I -m nvflare.app_opt.confidential_computing.aci_bootstrap

The bootstrap module is the image's fixed startup program. It reads only
``/run/nvflare-input/startup-kit.tar`` and
``/run/nvflare-input/startup-kit.sig``; the latter is a separate binary signature
of the archive's exact bytes. It verifies approval using the image's public
key, safely extracts into a private directory under writable ``/workspace``,
and starts the fixed NVFlare server command. The supplied volume must not
cover the public key, interpreter, installed Python packages, or startup
program. Do not add launch environment variables that replace executable code.

Generate the policy before provisioning
=======================================

Copy ``docs/resources/ccprep.json`` into the example directory as
``aci-template.json``. This is an Azure Resource Manager deployment document,
which describes the container group. Update its image references, location,
resource group, networking, resource requirements, and registry authentication.
Protect any registry credentials through secure inputs as well; do not put
credentials into the template, its defaults, or its outputs.
Use immutable image references for both containers. Review the permitted
commands, mounts, debugging and execution permissions in the generated policy.
Provisioning approves the supplied policy; it does not determine whether that
policy is sufficiently restrictive.

The template defines a volume named ``nvflare-startup`` mounted read-only at
``/run/nvflare-input``. Its ``startup-kit.tar`` and ``startup-kit.sig`` files
refer directly to the secure inputs ``startupKitArchive`` and
``startupKitSignature``. The ``secureString`` input type prevents Azure from
retaining their values in deployment history; neither input has a default.
Keep these names, the mount, image references, and launch commands unchanged
after policy preparation. Secret file contents are supplied separately at
launch time.

See `Azure's secure-input documentation
<https://learn.microsoft.com/en-us/azure/azure-resource-manager/templates/data-types#secure-strings-and-objects>`_
for the deployment-history guarantees and their limits.

Before provisioning, the policy generator needs values for these required
inputs. Create ``aci-policy.parameters.json`` with harmless placeholders. The
document's ``$schema`` selects Azure's deployment-parameter format,
``contentVersion`` identifies this document version, and ``parameters`` maps
each input name to its supplied ``value``. Both placeholder values below encode
the word ``placeholder``; they cannot start the server. The policy describes
the secret mount, rather than these file contents.

.. code-block:: bash

   python3 - <<'PY'
   import json
   from pathlib import Path
   parameters = {
       "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentParameters.json#",
       "contentVersion": "1.0.0.0",
       "parameters": {
           "startupKitArchive": {"value": "cGxhY2Vob2xkZXI="},
           "startupKitSignature": {"value": "cGxhY2Vob2xkZXI="},
       },
   }
   Path("aci-policy.parameters.json").write_text(json.dumps(parameters, indent=2))
   PY

The policy generator prints Base64, a text encoding of binary contents. Decode
it to the raw policy file named by ``cce_policy_file``; do not provide the
encoded text or the entire deployment document to provisioning:

.. code-block:: bash

   az extension add --name confcom
   az confcom acipolicygen -a aci-template.json \
     --parameters aci-policy.parameters.json --print-policy > cce-policy.b64
   python3 - <<'PY'
   import base64
   from pathlib import Path
   encoded = b"".join(Path("cce-policy.b64").read_bytes().split())
   Path("approved-cce-policy.txt").write_bytes(base64.b64decode(encoded, validate=True))
   PY

Run policy generation on a platform supported by Azure's extension. Pull the
same image architecture Azure will run. The decoder changes only the transport
encoding; do not edit or reformat the resulting policy afterward.
The `policy-generator command reference
<https://learn.microsoft.com/en-us/cli/azure/confcom#az-confcom-acipolicygen>`_
documents the separate ``--parameters`` input file.

Provision the project
=====================

In ``cc_server.yml``, the ``azure_cc`` block selects the container target and
names the prepared policy and private deployment-signing key. Relative paths
are resolved against that configuration file:

.. code-block:: yaml

   azure_cc:
     deployment_target: confidential_container
     cce_policy_file: ./approved-cce-policy.txt
     startup_kit_signing_key_file: ./deployment-private-key.pem

Edit the participant names in ``project.yml`` and the attestation service
address and timing in ``cc_project.yml``. The example client is a confidential
virtual machine; use the corresponding unified configuration when deploying
Confidential Containers. See :ref:`cc_deployment` for those configuration
formats and their deployment prerequisites.

Running provisioning approves the exact supplied policy without another prompt:

.. code-block:: bash

   nvflare provision -p project.yml -w ./workspace

Provisioning snapshots the decoded policy, calculates its fingerprint, and
places that expected value in every ACI checker, including the server's
component. It packages the same policy with the server startup kit. The common
kit signature is completed before the final archive is signed using the
separate deployment key. Neither the private deployment key nor its path is
included in delivered files or public delivery metadata.

Do not rebuild the image or regenerate the policy after this step. The
provisioning delivery manifest identifies the server output and policy
fingerprint. The production output's server directory contains
``startup-kit.tar``, ``startup-kit.sig``, and ``approved_cce_policy.txt``.
Keep the private provisioning state and private deployment key
under administrator control; distribute only the intended participant outputs.

Launch with the approved policy and signed kit
==============================================

Use the server delivery files emitted by provisioning: ``startup-kit.tar``,
``startup-kit.sig``, and the packaged approved policy. Create a private launch
copy of the prepared template. Encode the archive and binary signature once
for the Azure secret-volume transport, and encode the raw approved policy once
for the deployment's ``ccePolicy`` field. Do not calculate a new policy.

The following Python script takes the prepared template and the three generated
files as arguments. It writes ``aci-launch.json`` with the unchanged secure
input references and approved policy, plus ``aci-launch.parameters.json`` with
the archive and signature values. The latter uses the parameter-document format
described above and contains credentials: do not commit, print, or share it with
clients. Both outputs are created with owner-only permissions and refuse to
overwrite existing files. The startup input values go directly into the secret
volume without a transformation inside the template. Azure's secure-input
redaction does not necessarily cover transformed copies of a secure value.

.. code-block:: bash

   umask 077
   python3 - aci-template.json /path/to/startup-kit.tar \
     /path/to/startup-kit.sig /path/to/approved_cce_policy.txt <<'PY'
   import base64
   import json
   import os
   import sys
   from pathlib import Path

   template, archive, signature, policy = map(Path, sys.argv[1:])
   document = json.loads(template.read_text())
   properties = document["resources"][0]["properties"]
   volume = next(v for v in properties["volumes"] if v["name"] == "nvflare-startup")
   expected = {
       "startup-kit.tar": "[parameters('startupKitArchive')]",
       "startup-kit.sig": "[parameters('startupKitSignature')]",
   }
   if volume["secret"] != expected:
       raise ValueError("Template must reference the secure startup inputs directly")
   for name in ("startupKitArchive", "startupKitSignature"):
       definition = document["parameters"][name]
       if definition["type"].lower() != "securestring" or "defaultValue" in definition:
           raise ValueError("Startup inputs must be secure strings without defaults")
   properties["confidentialComputeProperties"]["ccePolicy"] = (
       base64.b64encode(policy.read_bytes()).decode("ascii")
   )
   parameters = {
       "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentParameters.json#",
       "contentVersion": "1.0.0.0",
       "parameters": {
           "startupKitArchive": {"value": base64.b64encode(archive.read_bytes()).decode("ascii")},
           "startupKitSignature": {"value": base64.b64encode(signature.read_bytes()).decode("ascii")},
       },
   }
   for name, contents in (("aci-launch.json", document), ("aci-launch.parameters.json", parameters)):
       descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
       with os.fdopen(descriptor, "w") as output:
           json.dump(contents, output, indent=2)
   PY

   az deployment group create --resource-group cc-prep-rsr-grp \
     --template-file aci-launch.json --parameters @aci-launch.parameters.json --output none

Here ``@aci-launch.parameters.json`` tells Azure's command-line tool to read the
private input values from that file. The command line contains its path, not
the archive contents. Never pass credential bytes as command arguments, enable
debug logging, or copy the values into template literals, defaults, or outputs.
After deployment, securely remove unneeded local launch-input files according
to your credential-handling policy. Secure parameters prevent this deployment-
history disclosure; they do not encrypt the input files or protect credentials
from authorized deployment operators or Azure's deployment service.

The archive is uncompressed tar, a file-bundling format. The bootstrap limits
both archive size and total extracted file contents to 16 MiB (16 times
1,048,576 bytes), and permits at most 4,096 entries. Azure deployment request
or secret-volume limits may be lower; do not bypass those limits or expose the
archive through logs if Azure rejects it. The bootstrap rejects invalid
signatures, unsafe paths, duplicate entries, links, special files, and oversized
inputs before starting the server.

The complete template below contains placeholders, not working credentials or
an approved policy:

.. literalinclude:: ../../../resources/ccprep.json
   :language: json

Verify and maintain the deployment
==================================

The administrator should run these checks against the actual Azure deployment:

* Start the server with its generated archive and policy. Confirm that startup
  signature verification succeeds and a correctly configured client accepts
  its evidence and can register.
* In a separate test deployment, generate a different enforcement policy for a
  changed workload and obtain fresh evidence. Keep the client's original
  approved fingerprint. Confirm that the client rejects the server's evidence
  rather than releasing protected work to it.
* Change an archive byte without signing again and confirm startup fails before
  NVFlare starts. Confirm that a missing or replaced signature also fails.

Keep tests that change policy or damage files separate from the running
federation. Use fresh evidence: expired evidence tests expiration, not policy
approval. Preserve deployment errors and ordinary server logs without dumping
archive contents, launch documents, or private keys.

A policy change requires reprovisioning and redistribution of affected kits,
then deployment with the newly packaged policy. A deployment-signing key change
also requires rebuilding the fixed image and regenerating its policy. Existing
ACI configurations without the expected policy fingerprint must be
reprovisioned; there is no signature-only fallback. Previously signed archives
remain valid: this workflow does not impose an archive expiration or version
counter. Each participant must still supply its configured checker labels, which
select the configured verification components. A label alone does not establish
the evidence type. The Azure confidential-virtual-machine checker,
``AZCVMAuthorizer``, currently validates the evidence signature and applicable
time limits but does not require a statement identifying an Azure compliant
confidential virtual machine. An enrolled client can therefore present
otherwise-accepted ACI evidence under its required Azure virtual-machine checker
label. The client's transport identity must still authenticate. This limitation
concerns the Azure virtual-machine checker; it does not establish a bypass of
Confidential-Containers verification.

Recovery after a credential compromise
======================================

Revocation means withdrawing previously granted trust. Repeating provisioning
does not revoke old archives, and the provisioner can reuse the existing
participant certificate and private key from its saved state. Do not treat an
ordinary provisioning run as credential rotation.

To withdraw trust in old signed kits, replace the deployment-signing key, build
a new fixed image containing its new public key, generate and approve the new
image's policy, and reprovision. Redistribute the new expected fingerprint to
every peer before resuming protected work, and stop and retire the old ACI
deployment. Old images continue trusting the old signing key; key rotation
does not remotely invalidate them. A deliberately changed policy fingerprint
can also exclude an old deployment even if the signing key stays unchanged.

If the server identity key was disclosed, also replace the compromised identity
credentials and update peer trust so the old credentials are no longer
accepted, using your project's credential-replacement procedure. Retire retained
deployment templates and local launch documents that exposed those credentials,
and audit who could retrieve them. Removing the history does not undo an
earlier disclosure. The archive signature and policy fingerprint do not, by
themselves, revoke a stolen identity key or a previously issued attestation
token.
