# Provision NVFlare confidential containers

This directory uses the unified confidential-computing interface. The project
selects one [`cc_project.yml`](cc_project.yml), every protected server or client
selects one participant `cc_config`, and the common `CCPackager` owns the
private/public handoff.

The shared project file contains the Trustee object once, a named container
registry, and the trusted CoCo build command. Participant files contain only
the workload source, TEE selection, release name, repository path, and approved
platform profile. To protect the server, uncomment its `cc_config` in
[`project.yaml`](project.yaml).

Before provisioning, prepare the admin and Trustee inputs described by the
parent CoCo runbooks, replace all placeholder paths and endpoints, and install
the authenticated Trustee AS signing key as `trustee-as-public.pem`. Then run
from this directory:

```bash
nvflare provision -p project.yaml -w ./workspace
```

The trusted build command receives a generated request. It stages the declared
Docker context and signed kit, builds and encrypts the image, publishes it by
digest, and returns a measured Pod. The public production directory contains
only the Pod handoff and `cc_manifests/<participant>.json`; source, credentials,
receipts, and plaintext kits remain below private state.

See `docs/user_guide/confidential_computing/deployment.rst` in the NVFlare source
tree for the complete schema, CPU/GPU mapping, shared Trustee configuration,
result format, and migration rules.
