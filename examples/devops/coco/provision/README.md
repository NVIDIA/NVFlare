# 3. Build NVFlare images and generate Pod YAML

Use the supplied project files with the existing Trustee and registry. Prepare
application images with the same NVFlare release used for provisioning.

The build machine needs Docker, skopeo, Cosign, OpenSSL, jq, Python 3.11+ with
PyYAML, and compatible Kata genpolicy/rules/settings already available.
The supported profile uses Kata/genpolicy 3.29.0 with matching rules/settings.
The runtime is inspected in place; no Kata image or chart is downloaded.
The build operator needs the existing engine privileges used by Docker/skopeo.

From `examples/devops/coco`, switch to the provisioning directory. The commands
below remain in that directory:

~~~bash
cd provision
~~~

## Configure the image builder

Create the image-builder configuration:

~~~bash
cp ../admin/platform.env.example ../admin/platform.env
chmod 0600 ../admin/platform.env
editor ../admin/platform.env
~~~

Set the existing tool paths, selected runtime, and capture output in that file:

~~~bash
WORK_ROOT="$HOME/coco-private/build"
GENPOLICY=/path/to/genpolicy
RULES=/path/to/rules.rego
SETTINGS=/path/to/genpolicy-settings.json
RUNTIME_CLASS=kata-qemu-nvidia-gpu-snp
WORKLOAD_LAUNCH_PROFILE="$HOME/coco-private/new-capture/approved-workload-launch-profile.json"
~~~

Keep the compatible generator version and digest-pinned key-provider image
from the template. The builder fetches that key-provider image automatically
if it is absent from its Docker daemon. Set the target cluster's Kubernetes
service host/port values. Publisher
credentials and Trustee endpoints come from the project CC configuration,
not this runtime profile. Skopeo publication uses the configured registry
CA/bundle automatically; it requires no manual global Skopeo trust changes.
For registries with a private CA, the Docker daemon must already trust that CA
when pulling application base images from the private registry.

## Prepare the workload and project

Replace the placeholder base in each selected Dockerfile with the
application image. Include NVFlare, Bash/coreutils, required dependencies,
and the executor/controller code. Include NVIDIA userspace libraries only when
the application requires them. The packager adds that participant's signed kit.

Use an in-process executor for the current packaged runtime. Bake its
code into the image and allow its exact class through `class_allow_list`.
Application data access and outputs need their normal owner-authorized channels.

Edit the supplied configuration:

| File | Inputs |
| --- | --- |
| [project.yaml](project.yaml) | Participant identities, `cc_project_config`, supplied builder order, and `CCPackager`. |
| [cc_project.yml](cc_project.yml) | Existing Trustee endpoint/CA/AS public key, guest AA token URL, registry/CA/publisher credentials, and build command. |
| [cc_site-1.yml](cc_site-1.yml) | CPU/GPU selection, application context, release/repository, and admin platform configuration. |
| [cc_server.yml](cc_server.yml) | Optional protected server configuration. |

Use the CPU/GPU pair from [README](../README.md) and a matching capture profile.
Set a new release name for each participant/image/key/policy change. Paths in
each CC file resolve beside that file; its Dockerfile resolves from its context.

The AS signing public key differs from the KBS TLS certificate. Supply it as
`trustee-as-public.pem`. Keep publishing credentials outside image contexts.
The guest-local AA token API stays inside the guest; its raw response contains
private key material and must not become application output.

The default protects one client. To protect the server too, enable its
`cc_config` in `project.yaml` and provide its own application image/profile.
GPU profiles require one pGPU per Pod, including a protected GPU server.

## Provision

While still in `examples/devops/coco/provision`, run:

~~~bash
nvflare provision -p project.yaml -w ./workspace
~~~

The runner builds, encrypts, signs, and publishes each image noninteractively.
It publishes only encrypted layers by immutable digest and generates the guest
policy, InitData, and final Pod.

The CLI prints the production directory containing the Pod and
`cc_manifests/<participant>.json`. With the supplied project and release, the
first run writes `workspace/coco_project/prod_00/site-1/site-1-v1-pod.yaml`
relative to the current `provision/` directory. Later runs increment `prod_NN`;
use the actual directory printed by the CLI. Plaintext kits/build state remain
private under `state/cc-private/` and the admin work directory.

Each release also produces:

~~~text
$WORK_ROOT/releases/<release>/handoff/
  trusted-service/                 # private Trustee image-key/policy inputs
  coco-it/<release>-pod.yaml       # final cluster manifest
~~~

The builder's exact Pod path is
`$WORK_ROOT/releases/<release>/handoff/coco-it/<release>-pod.yaml`, using
`WORK_ROOT` from `../admin/platform.env` and `coco.release_name` from the
participant CC file. Give the sibling `trusted-service/` directory confidentially
to the Trustee administrator and run
[release registration](../TRUSTEE.md#after-provisioning-register-each-workload-release).
`nvflare provision` does not configure Trustee automatically. After registration,
give only the Pod YAML to the cluster operator and follow [launch](../LAUNCH.md).

The [unified deployment schema](https://nvflare.readthedocs.io/en/2.9/user_guide/confidential_computing/deployment.html)
and [security architecture](https://nvflare.readthedocs.io/en/2.9/user_guide/confidential_computing/coco_security_architecture.html)
provide the full API and peer-proof details.
