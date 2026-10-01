# Architecture: three machines, one trust decision

## NVFlare provisioning node

- Builds and tests the plaintext workload image.
- Signs the immutable OCI digest.
- Encrypts image layers with a fresh per-release key.
- Pushes only ciphertext to the TLS registry.
- Generates the Kata Agent policy, measured InitData, release authorization, and digest-pinned Pod YAML.
- Sends secret release material only to the independent trust administrator; sends only the Pod YAML and authenticated hash to CoCo IT.

## Secure services

- TLS private registry stores encrypted images.
- TLS Trustee endpoint fronts KBS, AS, and RVPS.
- Verifies platform and workload handoffs.
- Installs approved SNP measurements/TCB floors or complete TDX reference tuples.
- Stores image keys and exact-path, default-deny release rules.
- Releases resources only when the required CPU/GPU evidence and workload identity agree.

## Adversarial CoCo cluster

- Kubernetes, containerd, Calico, digest-pinned Kata SNP/TDX runtime, and NVIDIA GPU Operator for GPU targets.
- CoCo IT installs public digest-pinned runtime inputs without a signed bundle and launches unchanged Pod YAML.
- The host sees ciphertext and metadata; image decryption happens only inside the SNP/TDX Kata guest.
- CoCo retains availability control, but receives no image key, plaintext image, publisher credential, or KBS administration authority.

---

# Trusted reference meets live evidence at one equality gate

## Establish the trusted reference

1. The trusted reference system generates a fresh 64-byte nonce.
2. One selected Kata collector Pod obtains fresh SNP report or TDX quote evidence bound to that nonce.
3. The trusted host verifies signatures/collateral, nonce, security baseline and actual launch; TDX additionally requires verified measured-boot replay. A second fresh rehearsal must agree.
4. The platform authority approves the stable measurements and launch artifacts; stage 09 independently reverifies retained evidence and approvals.
5. Stage 10 exports SNP's five fields or TDX's versioned complete-profile JSON through the provisioning node; a separate v4 launch contract goes to the image owner.
6. The secure-services administrator authenticates the sender, approves the reference set, and installs RVPS references with its own AS policies. No signed bundle is required.

The offline `sev-snp-measure` result is retained as a diagnostic model; it never replaces the hardware-signed report value.

## Appraise every live workload

1. Kata starts the selected SNP/TDX guest and submits CPU evidence, bound InitData, and NVIDIA evidence when a GPU is required.
2. KBS obtains a signed EAR from AS/RVPS after CPU appraisal and, for GPU releases, NVIDIA GPU appraisal. SNP requires its approved measurement/TCB; TDX requires a complete approved measurement/boot-event tuple and verified security baseline.
3. KBS requires the approved CPU type, exactly `cpu0` for CPU-only or both `cpu0` and `gpu0` for GPU, and each exact target trust vector. Failure of any required attestation denies every resource.
4. KBS then matches the InitData hash, immutable image digest, command, and actual resource path. Only after all platform and workload gates pass does it JWE-encrypt the approved resources to the attested guest key.

---

# Attacks controlled by the CoCo owner

| Attack on the CoCo cluster | Mitigation | Result |
|---|---|---|
| Read registry blobs, containerd cache, or host files | OCI layers are encrypted before publication; the key is released only to an attested guest | Ciphertext only |
| Replace an image, tag, manifest, or layer | Immutable digest, Cosign verification, authenticated encryption, Agent policy, and KBS image binding must agree | No start or no key |
| Change command, UID/GID, environment, mounts, or OCI properties | Kata Agent policy constrains the exact `CreateContainerRequest`; KBS binds command and image | Guest request denied |
| Modify Pod InitData, Trustee endpoint, CA, or image policy | Exact InitData digest is bound in SNP HOST_DATA or zero-padded TDX MRCONFIGID and matched by release policy | Attestation/release denied |
| Replace measured guest kernel, firmware, rootfs, or launch configuration | Verified SNP measurement or TDX complete measurement/boot-event tuple differs from the approved AS/RVPS reference | CPU trust vector fails |
| Fake evidence, replay evidence, enable debug or downgrade TCB | Fresh challenge/session binding, hardware and EAR signatures, quote/report verification and approved security baseline | Appraisal fails |
| Request another KBS resource or another release's key | The actual path is evaluated for every request against an exact allow-list with default deny | Resource denied |
| Use `kubectl exec`, attach, copy, ephemeral debug, or SSH | Agent policy denies exec/stream operations; the Pod has no SSH server, credentials, host mounts, or service-account token | No guest shell |
| Inspect or alter guest memory from the host/VMM | SNP/TDX memory isolation protects the approved guest, subject to platform security assumptions | Plaintext remains in TEE |
| Kill, pause, starve, or disconnect the Pod | Availability is deliberately outside the confidentiality/integrity guarantee | Denial of service remains possible |
