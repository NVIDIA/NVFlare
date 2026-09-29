# Service security boundary

The service administrator is trusted to enforce release policy and protect
secrets. The CoCo owner controls Kubernetes, manifests, scheduling, the host,
network availability, and ordinary cluster observability, and is adversarial.

The service releases an image key only after attestation evidence is accepted by
the Attestation Service/RVPS and the KBS resource policy matches the authorized
init-data, immutable encrypted-image digest, process arguments, and platform
claims. The encrypted image remains ciphertext in the registry, on CoCo storage,
and to host-level inspection; decryption occurs inside the confidential guest.

For SNP, CPU appraisal requires the exact launch measurement extracted from the
trusted rehearsal system's challenge-bound AMD-signed report,
debug and migration disabled, and each signed `reported_tcb_*` SVN at or above
its independently approved numeric floor. Missing or nonnumeric RVPS floor data
must not compare successfully. A signature-valid report from an older TCB must
not produce the exact platform-approved trust vector merely because its
certificate chain verifies.

For TDX, CPU appraisal instead requires one complete approved MRTD/RTMR/XFAM/
kernel-event tuple in `coco_tdx_profiles_v2`, verified quote/collateral and
measured-boot evidence, acceptable TCB, and non-debug configuration. Independent
per-field allowlists must not approve mixed profiles. Each workload's KBS rule
also pins the CPU TEE, expected CPU-only or CPU+GPU submodules, and exact
InitData (zero-padded MRCONFIGID for TDX). CPU-only approval cannot bypass a
GPU-required release policy. See [TDX approval](TDX-REFERENCE-VALUES.md).

CPU-only releases require the approved CPU appraisal and exactly the `cpu0`
submodule; GPU-required releases require both `cpu0` and `gpu0` to pass. Installing
the GPU verifier does not make a GPU mandatory for every workload. Neither the
cluster owner nor the received token can downgrade a GPU-required workload to
CPU-only: the service administrator installs the workload owner's reviewed
target-specific release authorization.

This architecture protects confidentiality and launch integrity, not
availability. The CoCo owner can refuse to schedule, kill the Pod, block the
network, or submit a modified manifest that fails attestation. A Pod YAML is not
itself a secret or a trusted enforcement point; its security-relevant values are
committed into signed init-data and enforced by the KBS/AS policies.

Do not place SSH credentials, an SSH daemon, debug shells, host mounts,
privileged mode, host namespaces, or Kubernetes API credentials in the workload.
The workload's generated policy must allow only its intended command, mounts,
environment, and I/O. Application-level mTLS is needed when the workload owner
must authenticate successful execution rather than merely observe Kubernetes
status.
