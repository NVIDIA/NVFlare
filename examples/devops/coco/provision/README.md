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

For an ordinary server with a TDX CPU-only client and an SNP plus NVIDIA
confidential-GPU client, use the complete
[mixed TDX/SNP project](mixed-tdx-snp/README.md). Its shared Trustee policy
contains CPU and measurement pins; each participant's `gpu_tee` determines
whether the generated peer verifier requires a signed GPU appraisal.

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

## In-process executor workaround

For the [external-process executor limitation](../FL-DEPLOYMENT.md#executor-limitation-and-workaround),
use a reviewed executor that performs training in the NVFlare job process:

1. Install the executor as an importable Python package, together with its
   reviewed training code and dependencies, in the participant's application
   image. The executor must perform the work in process rather than launch an
   external training command.
2. Add its exact fully qualified class name to the top-level `class_allow_list`
   in that participant's `cc_site-*.yml`, preserving any other required entries:

   ```yaml
   class_allow_list:
     - my_training.executor.InProcessExecutor
   ```

3. Reference that same class in the job's client configuration:

   ```json
   {
     "executors": [
       {
         "tasks": ["*"],
         "executor": {
           "path": "my_training.executor.InProcessExecutor",
           "args": {}
         }
       }
     ]
   }
   ```

   `my_training.executor.InProcessExecutor` is a placeholder for your reviewed,
   image-installed `Executor` implementation, not a class shipped with NVFlare.
4. Reprovision and build, sign and encrypt a new application release with a new
   per-release key, then obtain renewed release approval and deploy it. Submit
   job configuration and data without job-supplied Python code. Further changes
   to the baked executor or its dependencies require another reviewed release.

See `docs/user_guide/confidential_computing/deployment.rst` in the NVFlare source
tree for the complete schema, CPU/GPU mapping, shared Trustee configuration,
result format, and migration rules.
