---
aspectratio: 169
---

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
- Installs the approved SNP measurement and TCB floors.
- Stores image keys and exact-path, default-deny release rules.
- Releases resources only when live CPU/GPU evidence and workload identity agree.

## Adversarial CoCo cluster

- Kubernetes, containerd, Calico, NVIDIA GPU Operator, and digest-pinned Kata runtime.
- CoCo IT installs public digest-pinned runtime inputs without a signed bundle and launches unchanged Pod YAML.
- The host sees ciphertext and metadata; image decryption happens only inside the SEV-SNP Kata guest.
- CoCo retains availability control, but receives no image key, plaintext image, publisher credential, or KBS administration authority.

# Trusted reference meets live evidence at one equality gate

## Establish the trusted reference

1. The trusted reference system generates a fresh 64-byte nonce.
2. One Kata CoCo collector Pod runs `snpguest` and places that nonce in SNP `REPORT_DATA`.
3. The trusted host verifies the AMD ARK/ASK/VCEK chain, report signature, reported TCB, and exact nonce binding.
4. Stage 09 extracts `MEASUREMENT` directly from signed report bytes `0x90..0xbf` and reverifies retained hashes and TCB values.
5. Stage 10 exports a five-value JSON through the provisioning node; a separate launch contract goes to the image owner.
6. The secure-services administrator authenticates the sender, approves the five values, and installs RVPS references with its own AS policies. No signed bundle is required.

The offline `sev-snp-measure` result is retained as a diagnostic model; it never replaces the hardware-signed report value.

## Appraise every live workload

1. Kata starts a fresh SEV-SNP guest and submits CPU evidence, InitData/HOST_DATA, and NVIDIA GPU evidence to Trustee.
2. KBS obtains a signed EAR from AS/RVPS after appraisal of both AMD SEV-SNP CPU evidence and NVIDIA GPU evidence. CPU appraisal requires the live signed `MEASUREMENT`, TCB floors, and disabled debug/migration; GPU appraisal requires accepted NVIDIA evidence.
3. KBS requires exactly the `cpu0` and `gpu0` submodules and the exact approved eight-component trust vector for each. Failure of either attestation denies every resource.
4. KBS then matches the InitData hash, immutable image digest, command, and actual resource path. Only after all platform and workload gates pass does it JWE-encrypt the approved resources to the attested guest key.

# Attacks controlled by the CoCo owner

| Attack on the CoCo cluster | Mitigation | Result |
|---|---|---|
| Read registry blobs, containerd cache, or host files | OCI layers are encrypted before publication; the key is released only to an attested guest | Ciphertext only |
| Replace an image, tag, manifest, or layer | Immutable digest, Cosign verification, authenticated encryption, Agent policy, and KBS image binding must agree | No start or no key |
| Change command, UID/GID, environment, mounts, or OCI properties | Kata Agent policy constrains the exact `CreateContainerRequest`; KBS binds command and image | Guest request denied |
| Modify Pod InitData, Trustee endpoint, CA, or image policy | Exact InitData hash is carried in SNP `HOST_DATA` and matched by release policy | Attestation/release denied |
| Replace Kata kernel, firmware, rootfs, or runtime | Live AMD-signed SNP `MEASUREMENT` differs from the approved AS/RVPS reference | CPU trust vector fails |
| Fake evidence, replay evidence, enable debug, migrate, or downgrade TCB | Fresh challenge/session binding, AMD and EAR signatures, debug/migration checks, and minimum TCB floors | Appraisal fails |
| Request another KBS resource or another release's key | The actual path is evaluated for every request against an exact allow-list with default deny | Resource denied |
| Use `kubectl exec`, attach, copy, ephemeral debug, or SSH | Agent policy denies exec/stream operations; the Pod has no SSH server, credentials, host mounts, or service-account token | No guest shell |
| Inspect or alter guest memory from the host/VMM | SEV-SNP encrypts guest memory and protects its integrity from the hypervisor | Plaintext remains in TEE |
| Kill, pause, starve, or disconnect the Pod | Availability is deliberately outside the confidentiality/integrity guarantee | Denial of service remains possible |
