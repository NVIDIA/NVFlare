# CoCo cluster administrator kit

First complete [public-package configuration](../CONFIGURATION.md).
Bootstrap stages derived from NVFlare commit
`54452740e68bd776343c4d0b8883459e0ffa9cd7` are vendored in `bootstrap/`, with
local hardening and pinned-runtime adaptation. No NVFlare clone is required.

CoCo IT is intentionally not a security authority. It receives a public
registry CA once and, for each release, exactly one digest-authenticated Pod
YAML. It receives no image plaintext, publisher credential, decryption key,
signing key, KBS credential, Trustee administrator access, or policy fragment.

## Build a fresh cluster

Read every script before running it:

```bash
cd "$HOME/coco-it"
./00-fetch-pinned-workflow.sh
./10-run-host-preflight.sh
```

The first run of `10` creates `~/.config/coco-platform/config.env` and exits.
Review the detected node IP, TEE platform, runtime class, package pins, and
intended `EXPECTED_HOSTNAME`. Then run:

```bash
./10-run-host-preflight.sh
./20-install-kubernetes.sh
```

Before stage 30, receive `kata-deploy-3.29.0.tgz` from provisioning_node and place it
at `public/kata-deploy-3.29.0.tgz`. Use this kit's
`public/kata-platform.env` for the approved chart hash and runtime image digest.
No platform archive, detached signature or platform signing public key is
required for this public runtime handoff. Then run:

```bash
./30-install-coco-gpu.sh
./35-repin-kata-deployment.sh public/kata-deploy-3.29.0.tgz
```

Stage 35 checks the chart SHA-256 and makes the final Kata DaemonSet use the
exact amd64 image digest. These pins are reproducibility checks, not a trust
grant to CoCo IT. CoCo is the untrusted cluster operator: do not run the
trusted-system measurement/rehearsal workflow here or approve references from
this host. Trustee independently enforces the approved platform references,
CPU/GPU appraisal and workload release policy before releasing keys.

These wrappers invoke only the vendored bootstrap stages `00`, `10`, and `20`. Do not run
upstream `30-deploy-security-services.sh` or `run-all.sh`: they deploy a local
registry and Trustee inside the adversarial cluster, while this design uses the
independent `service` host.

Authenticate the service administrator's `registry-ca.crt` by an independent
channel, place it in `public/`, and run:

```bash
./40-configure-registry-trust.sh
./60-verify-platform.sh
```

The container runtime gets only TLS trust and anonymous pull/resolve
capabilities. No registry publisher credential is installed.

## Launch a workload-owner handoff

Follow [COCO-IT-RUNBOOK.md](COCO-IT-RUNBOOK.md). In short:

```bash
./50-launch-handoff.sh RELEASE-pod.yaml EXPECTED_SHA256
```

Obtain `EXPECTED_SHA256` from the workload owner through a separate
authenticated channel. The script compares it, validates the restrictive Pod
and embedded Kata policy, performs a server-side dry run, asks for explicit
confirmation, applies the unchanged file, and waits for readiness.

Kubernetes status proves only that the cluster observed the Pod as ready. The
workload owner should use application-level mTLS for proof of the expected
confidential workload's successful execution.

See [SECURITY.md](SECURITY.md) for what the hostile host can and cannot do.

## Maintenance

Destructive lab-specific teardown scripts are not included. See
[maintenance boundaries](../trusted_system/TEARDOWN.md).
