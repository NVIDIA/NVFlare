# Configure the public package before deployment

This package contains code and templates, not an existing deployment's state.
Use dedicated Ubuntu 26.04 x86_64 provisioning/trusted/CoCo hosts and a fresh
Ubuntu 24.04 x86_64 secure-services host. The trusted and CoCo hosts need AMD
SEV-SNP enabled and one supported NVIDIA GPU in production CC mode. A CC-ready
label is not cryptographic GPU attestation. Secure services must successfully
appraise both CPU and GPU evidence before authorizing key release.

Read scripts before running them. Installers change packages, services,
networking and GPU ownership; they are not safe for a shared production host.
No lease, DNS, firewall or SSH identity is created by these scripts.

## 1. Local coordinator: validate and distribute code

From `NVFlare/examples/devops/coco`:

```bash
python3 validate-package.py
sha256sum --check --strict PACKAGE-SHA256SUMS
```

Use Python 3.11+ with PyYAML for tests and a Bash-capable Linux system. The
root [README](README.md#local-validation) provides virtual-environment setup
commands if those dependencies are not available. The validator never deploys
software or contacts a node. Checksums establish byte
integrity only; authenticate the code source independently. Copy each role
directory from a clean package to its corresponding operator using a trusted
channel with verified SSH host keys. Never copy a used working directory or
the entire private operational workspace. Public documents can be shared;
generated files must follow the destination table below.

## 2. Configure admin, secure services, and CoCo

On each destination, from its role directory:

```bash
umask 077
test ! -e platform.env
cp platform.env.example platform.env
chmod 0600 platform.env
hostname -f
editor platform.env
```

Set `EXPECTED_HOSTNAME` to that intended machine's exact `hostname -f` result
after confirming its identity. Set all secure-services endpoints to your
authenticated real DNS name; `example.com` endpoints are rejected. These
files are trusted shell code: do not source configuration supplied by CoCo IT
on a trusted machine. Do not change the pinned software inputs without a new
review and measurement rehearsal.

| Role | Configuration requiring deployment-specific review |
|---|---|
| Admin | `REGISTRY_HOST`, `KBS_URL`, `WORK_ROOT`, target Kubernetes service IP/port; later pin `WORKLOAD_LAUNCH_PROFILE_SHA256` to the authenticated contract |
| Secure services | `SERVICE_FQDN`, `REGISTRY_PUBLISHER_CIDR` (admin's exact egress IPv4/32), dedicated state paths; stage 02 fills the five blank reference fields |
| CoCo | `SERVICE_FQDN`, private `COCO_CONFIG` and state paths; the runtime does not need approved measurements in its configuration |

Keep registry TLS on 5000 and KBS HTTPS on 8443. Allow access only from intended
origins; never expose backend ports. The publisher and CoCo must have distinct
egress IPs for the current registry challenge configuration. CoCo must never
receive a publisher credential to work around an HTTP 401.

No certificates are included. Receive the new public certificates directly
from the secure-services administrator and authenticate fingerprints before
installation. TLS private keys remain on secure services. A certificate
rotation requires regeneration and authorization of affected measured Pods.

## 3. Trusted platform system configuration

From the package root on the trusted platform system (or adjust the paths if
only that role directory was delivered):

```bash
umask 077
install -d -m 0700 "$HOME/private-platform-reference"
test ! -e "$HOME/private-platform-reference/platform-reference.env"
cp trusted_system/sec-sys-launch-profile.env.example \
  "$HOME/private-platform-reference/platform-reference.env"
CONFIG="$HOME/private-platform-reference/platform-reference.env"
editor "$CONFIG"
test ! -e trusted_system/bootstrap/config.env
cp trusted_system/bootstrap/config.env.example trusted_system/bootstrap/config.env
chmod 0600 "$CONFIG" trusted_system/bootstrap/config.env
editor trusted_system/bootstrap/config.env
```

Set `EXPECTED_HOSTNAME` in **both** files. Choose a new `PLATFORM_PROFILE`
for every collection; use dedicated private work/tools directories and place
the reviewed source Pod YAML at `REHEARSAL_WORKLOAD_YAML`. For the included
one-container/one-GPU example, start from
`trusted_system/workload-source.yaml.example`; it defines only the launch
shape and must not be deployed as an application. The rehearsal substitutes
its collector image. Review the exact workload's launch conditions, the CPU
baseline assumption, and the limitations in
[the trusted-system guide](trusted_system/SEC-SYS-LAUNCH-PROFILE.md).

## 4. Execute in dependency order

1. Trusted system: stages **01–10**, with fresh evidence and a repeat rehearsal.
   Stage 01 takes `"$CONFIG"`; stage 02 uses `bootstrap/config.env`.
2. Admin coordinates the two stage-10 outputs. Secure services receives only
   the five-value JSON; admin authenticates and pins its separate launch contract.
3. Secure services: stages **01–11**, then RVPS restart/readback. See
   [fresh installation](service/SERVICE-INSTALLATION.md).
4. Admin: receive public trust and its private publisher credential, then
   stages **00, 05, 10, 20, 25, 30, 40**. Build the static example binary
   with `bash example-workload/build-example.sh` before stage 10 when using
   the example Dockerfile. Edit `BUILD_CONTEXT` in the workload environment
   to the actual absolute build directory. Stage 31 is optional failure recovery.
5. CoCo: stages **00, 10, 20, 30, 35, 40, 60** using the public chart and pins.
   Stage 00 checks vendored files; it no longer clones NVFlare. See
   [cluster installation](coco/COCO-MACHINE-REDEPLOY.md).
6. Admin sends its six-file confidential workload handoff to secure services,
   which reviews it and runs **12**. Only after acceptance, CoCo IT receives
   the single Pod YAML, authenticates its hash, and runs **50**, then **70**.
   Secure services runs **13** to inspect appraisal/resource-release evidence.

| Destination | Separately delivered material |
|---|---|
| Secure services | Trusted system's `platform-reference-values.json`; later admin's six-file `trusted-service/` workload handoff including `image_key` |
| Admin | Trusted launch contract; `public/trustee.crt`, `public/registry-ca.crt`; confidential publisher `username`/`password` directory |
| CoCo | `public/registry-ca.crt`, public Kata chart, included public digest pins; finally one Pod YAML with its independently authenticated hash |

Stage 03 on the trusted system downloads the chart under
`$PLATFORM_WORK_ROOT/$PLATFORM_PROFILE/artifacts/kata-deploy-3.29.0.tgz`.
Admin may relay **that single public chart**, not the artifacts directory or
private evidence, to CoCo's `public/`. The runtime image is downloaded by the
installer using the included immutable amd64 digest. No signed-bundle stage
is required. CoCo's honesty is not relied on for key-release authorization.

Host-specific teardown scripts are intentionally not part of this package.
See [maintenance boundaries](trusted_system/TEARDOWN.md). Local validation is
not an end-to-end hardware attestation test; perform the runbooks' positive
and negative tests in your deployment before relying on the result.
