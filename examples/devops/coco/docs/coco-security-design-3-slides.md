<a id="architecture-three-machines-one-trust-decision"></a>

# Four roles, two security decisions

Presentation summary of the [CoCo + NVFlare security architecture](https://nvflare.readthedocs.io/en/2.9/user_guide/confidential_computing/coco_security_architecture.html),
which defines the full guarantees, assumptions and residual risks. The deployment
steps cover the four explicit SNP/TDX CPU-only/GPU profiles. TDX support remains
experimental pending successful hardware acceptance; see [runtime status](../RUNTIME-VARIANTS.md#support-status-and-hardware-validation).

## NVFlare provisioning node

- Builds and tests the plaintext workload image.
- Encrypts image layers with a fresh per-release key.
- Pushes only ciphertext to the TLS registry.
- Signs the published encrypted image's immutable OCI digest.
- Generates the Kata Agent policy, attestation-bound InitData, release authorization, and digest-pinned Pod YAML. InitData is hash-bound in hardware evidence, not a separately signed artifact.
- Sends secret release material only to the independent trust administrator; sends only the Pod YAML and authenticated hash to CoCo IT.

## Secure services

- TLS private registry stores encrypted images.
- TLS Trustee endpoint fronts KBS, AS, and RVPS.
- Verifies platform and workload handoffs.
- Installs approved SNP measurements/TCB floors or complete TDX reference tuples.
- Stores image keys and exact-path, default-deny release rules.
- Releases resources only when the required CPU/GPU evidence and attested workload-policy claims agree.

## Adversarial CoCo cluster

- Kubernetes, containerd, Calico, digest-pinned Kata SNP/TDX runtime, and NVIDIA GPU Operator for GPU targets.
- CoCo IT installs public digest-pinned runtime inputs without a signed bundle and launches unchanged Pod YAML.
- The host sees ciphertext and metadata; image decryption happens only inside the approved SNP/TDX Kata guest. Application output needs its own protection.
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
4. KBS then matches the InitData hash, policy-bound image digest and command, and actual resource path. The guest Agent enforces the approved process request; KBS does not inspect the live image pull, Kubernetes YAML or running process directly. Only after the release gates pass does it JWE-encrypt the approved resources to the attested guest key.

NVFlare separately verifies site/audience-bound proofs containing the signed EAR.
New peer proofs can contain cached EARs; periodic verification is not hardware
attestation for every job. A protected server participates just like a protected
client; an ordinary server can verify client proofs locally.

---

# Attacks controlled by the CoCo owner

| Attack on the CoCo cluster | Mitigation | Result |
|---|---|---|
| Read registry blobs, containerd cache, or host files | OCI layers are encrypted before publication; the key is released only to an authorized attested guest | No image plaintext; metadata remains visible |
| Replace an image, tag, manifest, or layer | Guest signature/integrity checks apply; KBS checks policy-derived image claims, not the actual pull. Exact-digest authorization remains unresolved for another accepted-key/same-repository signed image | Invalid signatures or ciphertext fail; universal image-substitution rejection is not established |
| Change command, UID/GID, environment, mounts, or OCI properties | Kata Agent policy checks the effective `CreateContainerRequest`; KBS binds attested command and image claims | Requests outside the enforced guest rules are denied; permitted changes can pass |
| Modify Pod InitData, Trustee endpoint, CA, or image policy | Exact InitData digest is bound in SNP HOST_DATA or zero-padded TDX MRCONFIGID and matched by release policy | Attestation/release denied |
| Replace measured guest boot inputs | The approved SNP measurement or complete TDX tuple covers measured inputs; an external disk/rootfs is not automatically covered and needs its own reviewed integrity mechanism | Changed covered inputs fail appraisal; do not infer protection for unmeasured storage |
| Fake or replay evidence within attestation, enable debug or downgrade TCB | Hardware signatures, challenge/session binding, verified report/quote and approved security baseline | Appraisal fails; this is not workload anti-rollback or exactly-once execution |
| Request another KBS resource or another release's key | The actual path is evaluated for every request against an exact allow-list with default deny | Resource denied |
| Use `kubectl exec`, attach, copy, ephemeral debug, or SSH | Guest policy denies exec/stream operations; the reviewed application must not expose SSH/debug services or credentials, and the packager does not scan custom images for daemons | Agent-mediated access denied; no SSH access only if the application review requirement is met |
| Inspect or alter guest memory from the host/VMM | SNP/TDX memory isolation protects the approved guest, subject to platform security assumptions | Plaintext remains in TEE |
| Read application output or replay a valid workload launch | Applications must protect their own outputs; duplicate execution and persistent-state rollback need separate owner-controlled protocols | No blanket log confidentiality, anti-cloning or rollback guarantee |
| Kill, pause, starve, or disconnect the Pod | Availability is deliberately outside the confidentiality/integrity guarantee | Denial of service remains possible |
