# CVM Builder — Design

**Status:** Owner decisions D3–D17, including exclusive attachment, binding-addressed KBS resources, OCI delivery, directional sidecars and measured multi-GPU support, are implemented. Test evidence and remaining production acceptance limits are recorded in [VALIDATION.md](VALIDATION.md); see also Revision history (§15).
**Date:** 2026-09-09 (latest review pass: 2026-09-21)
**Scope:** General-purpose bare-metal CVM and application-vault builder, currently located in `nvflare/lighter/cc/image_builder`. NVFlare is the main use case; application provisioning is outside the builder's scope.

---

## 1. Summary

The CVM Builder runs arbitrary compatible containerized applications inside a confidential VM. Its Docker container is generic: the application supplies the image and launch configuration, with no required NVFlare package, process, role or directory layout. NVFlare is the main use case, not a dependency of the builder or guest runtime. NVFlare startup-kit generation and `cc_params.yml` belong to NVFlare provisioning; that external layer translates its inputs into the generic builder contract (§9.3).

**Build one generic CVM per CPU platform (AMD SEV-SNP, Intel TDX) and reuse it across applications. Populate the vault content once per application release and site, and seal it into an independent copy for each platform it will run on.** (Throughout this document: *build* = Stage 1 generic-CVM construction; *populate* = Stage 2 content assembly; *seal* = Stage 2 per-platform authenticated-encryption copy.) The CVM is shared infrastructure, built once per platform for each profile version. Each vault contains the application and its site-specific configuration and secrets; its content is platform-neutral, but every sealed copy has its own key, identity and runtime state, is authorized for exactly one platform, and is never shared between platforms or between running CVMs. A vault image may be copied; each runtime file copy is dedicated to one CVM, with no concurrent attachment of that file (D8).

This design splits the builder into two independent stages and moves attestation into the booted OS:

1. **Generic CVM build — once per platform for each profile version.** Produces a platform-specific (SNP or TDX), application-agnostic, measured boot set: firmware, kernel, initramfs, dm-verity root image, and launch/shutdown script templates. It contains the OS, Docker, GPU driver, platform attestation tooling and the `/vault` unlock service, but no application, no site configuration and no secrets. The SNP and TDX bundles of one **profile version** are built from the same base image and package pins, so the *vault-facing contract* (paths, Python, Docker/containerd, glibc, directory contract) is identical across them; only firmware, CC kernel modules, the attester and the launcher differ. The two bundles are nonetheless **distinct measured root images with distinct `cvm_build_id`s** — they share base/pins but are not byte-identical, and a vault copy's manifest pins exactly one of them. All compatible applications and their releases reuse these bundles and their registered boot measurements.
2. **Vault build — content once per application release and deployment, sealed once per target platform.** Populates the application's Docker image archive, generic launch configuration and optional application files once, then seals that content into one authenticated, encrypted `vault.qcow2` per target platform (`amd_sev_snp`, `intel_tdx`), each with its own LUKS header, UUID, unlock secret, binding and launcher, plus three clear sidecar drives with directional roles: writable CVM output in `applog`, and read-only operator-supplied inputs in `user_config` and `user_data`. The KBS releases a 64-byte secret for `vault.qcow2` only. Application content is byte-identical across copies within one Stage 2 run (excluding each copy's identity manifest; see D3). Each sealed copy has independent cryptographic material, is authorized for exactly one platform bundle and follows the exclusive-attachment rule (D8). Each new application or application update produces new copies with fresh secrets and bindings. Deployment-specific configuration and secrets require separate vaults for each deployment. A deployment may represent a site, but the builder assigns no application-specific meaning to it. This stage requires no CVM rebuild, boot re-measurement, CC hardware or OS provisioning; runtime depends on vault capacity, content size and the number of platforms.

**KBS provisioning follows the Kata/CoCo separation of reference values, resources and policy (D9).** Stage 1 provisions generic boot reference values in Trustee/RVPS and reusable KBS rules once per CVM bundle. Stage 2 only uploads each vault's unlock secret at `keys/<cvm-bundle-id>/<vault-binding-id>`. The shared rule checks the approved bundle measurement and compares the requested binding with signed TEE evidence. There is **no separate per-vault measurement-registration file, resource-to-profile inventory or per-vault policy update**. Trustee still stores reference values, keys and policy; creating a key resource is the trusted authorization action for that vault. The binding-addressed path is this design's use of CoCo primitives, not a path convention required by Kata (§7.5).

At boot, the CVM verifies its root with dm-verity and stacks a tmpfs overlay. A systemd service detects the platform, checks the attached vault header against the TEE's instance-configuration field, attests the CPU with the matching attester, retrieves the vault secret, opens the authenticated storage stack and verifies it before starting the workload. The kernel command line is identical for all vaults sharing a generic bundle; it contains no per-vault identifiers or digests.

The key-binding mechanism and the TDX measurement path are taken from Confidential Containers (CoCo) / Kata: measured dm-verity root hash on the kernel command line, TDX evidence from RTMRs + CCEL event log via the guest `tdx_guest` interface, and the **initdata** mechanism (SNP `HOSTDATA` / TDX `MRCONFIGID`) to bind per-instance configuration into the attestation report.

---

## 2. Goals and non-goals

### Goals

- **G1. Build once, reuse across applications.** Build and measure one generic CVM per platform for each profile version. Populate and seal a separate vault for each application release and deployment. Application code, dependencies, configuration or secret changes rebuild the affected vault while reusing the same compatible CVM bundles and boot measurements, without CC hardware.
- **G1a. One vault build, one sealed copy per platform.** Vault content is platform-neutral and populated once; platform-specific tooling (attesters, TEE device names, launch parameters) lives only in the generic CVM and the launcher, never in the vault. A site that runs on both platforms receives two sealed copies with distinct keys, identities and runtime state. Each runtime file copy is used by one CVM at a time; copying a stopped image is allowed, while concurrent attachment of the same file is prohibited (D8). A copy sealed for one platform is refused on the other.
- **G2. Root integrity.** Read-only dm-verity root; root hash covered by the launch measurement; tmpfs upper layer so nothing persists in the clear.
- **G3. Vault confidentiality and integrity.** Application code, secret data, confidential configuration and scratch live on authenticated LUKS2 storage. The KBS holds the unlock secret; the trusted builder holds it transiently during construction, and the production guest holds it only in protected memory. Payload authentication is enforced on every read, including reads of services, scripts and Docker state.
- **G4. Key binding.** Release of every vault key requires an approved CPU appraisal, the registered measurement of the bundle named by the key path, and a path binding equal to the header digest in signed TEE evidence. The secret must exist at that exact path. The verified guest must check the digest against the header it actually uses to open the vault. GPU evidence is not a universal key gate: a CPU-only profile never requires it, while its bundle resource rule additionally requires GPU appraisal for a key that protects a confidential-GPU workload. GPU profiles require exactly `gpu_count` distinct NVIDIA GPU submods in the same signed EAR before the key is released; CPU-only rule bytes are unchanged (§6.2). An instance running vault A must not obtain vault B's key merely because both use the same root. This authorizes a root-and-vault combination; it does not identify a unique host or prevent another genuine instance from booting a complete copy of the same authorized vault (§6.2).
- **G5. Multi-platform.** AMD SEV-SNP (`kernel-hashes=on`) and Intel TDX (TDVF measured direct boot; RTMR/CCEL evidence) behind one build and boot flow, with one vault format, one population of the vault content, and one KBS resource per sealed copy.
- **G6. Generic operator surface.** Keep the partition/mount layout, `cvm_app.service` Docker contract, `applog`/`user_config`/`user_data` mounts, authenticated vault `nfs_mount` configuration (NFS `krb5p`) and firewall lockdown. `/applog` is a clear writable output disk so an operator can inspect logs without a KBS key. `user_config` and `user_data` are clear, host-readable input disks and are read-only at both the QEMU block layer and guest mount layer. The builder accepts source trees for both input disks and rejects private-key containers, filenames, PEM markers and symlinks. Expose application-neutral build inputs for direct use or external provisioning tools.
- **G7. Fail closed.** A startup verification failure (verity, attestation/appraisal, key release, vault identity or payload authentication) prevents workload startup and halts the guest. An integrity or attestation failure detected after startup stops the workload and powers off the guest.
- **G8. No KBS traffic from the initramfs.** The initramfs only opens the dm-verity root and stacks the overlay. Attestation, KBS calls and LUKS unlock run as a systemd service in the verified root (§5.1). The initramfs carries no network configuration, KBS endpoint, CA certificate or attestation binaries.
- **G9. No per-application measurement or policy registration.** Adding or rebuilding vaults leaves generic reference values and the resource policy unchanged. Keep versioned bundle artifacts, RVPS references and KBS resources; do not introduce a duplicate per-vault authorization ledger (§7.5).
- **G10. No workload-framework dependency.** Run any Docker application compatible with the selected guest architecture and profile. An image plus generic launch configuration is sufficient; application files are optional. The builder does not generate startup kits, interpret NVFlare roles or consume NVFlare provisioning schemas.

### Non-goals

- Application-specific provisioning, including NVFlare startup kits, `cc_params.yml`, project schemas and participant roles. Those are owned by the application provisioning layer (§9.3).
- Kubernetes deployments (CoCo is the recommended path there).
- Encrypting the generic root image or the three sidecar images. The root is public and reproducibly integrity-protected with dm-verity. `/applog` is intentionally clear CVM output; `user_config` and `user_data` are intentionally clear, read-only guest inputs. Confidential data and logs belong in `/vault`.
- Replacing Trustee as KBS. The design keeps Trustee (KBS + Attestation Service + RVPS) and its `kbs-client`; other KBSes could be plugged in behind the same service interface later.
- Making GPU evidence a gate for every key. CPU-only NVFlare and other CPU-only confidential workloads use CPU-CVM attestation without a GPU prerequisite. GPU profiles use composite CPU/GPU authorization before key release; physical conformance acceptance remains pending in [CONFORMANCE.md](CONFORMANCE.md).
- Preventing cloning of an authorized root-and-vault combination, or detecting replay of previously valid mutable disk sectors/snapshots. Attestation freshness and disk-state freshness are distinct (§6.3). Deployments requiring disk rollback protection need an external trusted state/version service.

---

## 3. Threat model

| Actor | Trusted? | Notes |
|---|---|---|
| CPU/TEE firmware (AMD SP, Intel TDX module, TDVF/OVMF) | Yes | Root of trust; vendor-signed. |
| KBS / Attestation Service operator | Yes | Holds vault keys and reference values. Must be run by the data owner or a mutually trusted party. |
| Builder host | Yes, at build time only | Sees plaintext vault content and the key while building. Must scrub scratch. |
| CVM host / hypervisor operator | **No** | Can read/replace any qcow2, edit QEMU command line, snapshot memory of non-CC VMs, spoof network. |
| Workload (Docker image) | Trusted by the party that built the vault | Runs inside the TEE. |

Attacks in scope: booting a modified root; obtaining keys from debug-enabled or unapproved CPU configurations; substituting a vault header or presenting a different binding in evidence; altering vault ciphertext, authentication tags or IVs; replaying stale attestation evidence; reading vault secrets from persistent storage; and preventing attestation services from responding. The host can read or replace clear sidecar contents. Applications therefore treat `user_config` and `user_data` as untrusted inputs and never put secrets in them; `/applog` contains only output intentionally disclosed to the host.

The application and admitted guest services remain part of the trusted computing
base. Guest services run as root unless configured otherwise; executable-path
validation is not a privilege sandbox. Containers can write only application
`runtime/` and `data/`, while service definitions, launch configuration, application
executables and Docker metadata are inaccessible or read-only. Trusted guest
services must not execute writable workload data as code. Workload compromise
can still disclose any secrets the workload is authorized to use.

Deployment requirements:

- Use a dedicated Trustee instance, resource storage and publisher state for each independently administered project or tenant and security profile. The KBS endpoint and trust roots are measured Stage 1 inputs inherited by every vault. Bundle-scoped resource roles restrict paths within that trusted administration domain; they do not isolate mutually untrusted tenants sharing a generic bundle. Policy administration replaces the instance-wide policy. Do not issue these administrative credentials to independent tenants on a shared instance. CoCo can manage the same Trustee when its workloads share this trust domain; independent domains need separate instances and generic profiles.
- Every participant trusts the selected Trustee operator with its vault keys. A federation whose participants do not share that trust cannot use one generic profile and one Trustee for all participants.
- Protection is asymmetric: the data owner running an opaque encrypted workload must trust the workload publisher with all data given to that workload. The CVM protects application/model secrets from the host; it does not prove that the workload preserves the data owner's confidentiality or prevent an authorized workload from exporting data. CIDR/port controls reduce destinations but are not a data-use policy. Review the workload, output policy and organizational agreement separately.

Out of scope: side channels, host DoS, physical attacks, bugs in the workload.

---

## 4. Architecture

```
   Stage 1 (once per platform, per profile version)         Stage 2 (once per app + deployment)
   cvmctl build auto-detects; optional -p <platform>
   ┌──────────────────────────┐ ┌──────────────────────────┐   ┌───────────────────────────────┐
   │ Generic CVM: amd_sev_snp │ │ Generic CVM: intel_tdx   │   │ Vault build                   │
   │ cvmctl build             │ │ cvmctl build             │   │ cvmctl vault                  │
   ├──────────────────────────┤ ├──────────────────────────┤   ├───────────────────────────────┤
   │ same base image + pins   │ │ same base image + pins   │   │ Docker archive (any app)      │
   │ OVMF.amdsev.fd, snp attes│ │ OVMF.inteltdx.fd, tdx att│   │ optional application files    │
   │ → verity_root.qcow2      │ │ → verity_root.qcow2      │   │ vault_build.yml               │
   │ → vmlinuz, initrd, OVMF  │ │ → vmlinuz, initrd, OVMF  │   │ profile set (both bundles)    │
   │ → CVM OCI artifact       │ │ → CVM OCI artifact       │   │ populate content ONCE, then   │
   └────────────┬─────────────┘ └────────────┬─────────────┘   │ seal one copy per platform:   │
                │  profile_set.json (build ids, measurements)   │ → amd_sev_snp/vault.qcow2     │
                └──────────────────┬───────────────────────────►│ → intel_tdx/vault.qcow2       │
                                   │ references + bundle rules  │   (own UUID, secret, bind,    │
                                   │                            │    launcher, sidecars each)   │
                                   │                            │ → delivery OCI per platform  │
                                   ▼                            └───────────────┬───────────────┘
                     ┌────────────────────────────────────────────────────────────┐ upload key only
                     │ Trustee: RVPS (measurements)  ·  KBS (keys, resource policy)│◄┘ per sealed copy
                     └────────────────────────────────────────────────────────────┘
```

Runtime (operator host; each host runs the copy sealed for its platform):

```
<platform>/launch_cvm.sh and shutdown_cvm.sh   (zero-argument runtime controls)
   ─► QEMU (SNP copy: sev-snp-guest,kernel-hashes=on,host-data=<b64(VAULT_BIND_snp)>
            TDX copy: tdx-guest,mrconfigid=<b64(VAULT_BIND_tdx‖0×16)>)
   -bios <platform OVMF>  -kernel vmlinuz  -initrd initrd.img   (from the matching generic bundle)
   -append "<exact cmdline recorded by that generic bundle>"
   disks: verity_root(ro) applog user_config(ro) user_data(ro) vault   (this platform's sealed copy)
```

### 4.1 Artifacts

**Generic CVM bundle workspace** `target/cvm_<profile_version>/<platform>/` (one per platform)

| File | Purpose |
|---|---|
| `OVMF.fd` | Platform firmware: `OVMF.amdsev.fd` (SNP, supports `kernel-hashes`) or `OVMF.inteltdx.fd` (TDVF) |
| `vmlinuz`, `initrd.img` | Kernel + initramfs (verity open, overlay). No KBS settings baked in. |
| `verity_root.qcow2` | ext4 data area + appended dm-verity hash tree |
| `launch_cvm.sh.tmpl`, `shutdown_cvm.sh.tmpl` | Integrity-covered runtime wrappers rendered into every vault delivery. The launcher uses the manifest for measured QEMU settings; the shutdown wrapper addresses the exact root-owned launcher state. |
| `cvm_manifest.json` | `platform`, `profile_version`, `build_id`, `root_hash`, `hash_offset`, exact `cmdline`, expected measurements (`snp.measurement` or `mr_td/rtmr_0/rtmr_1/rtmr_2`), launch shape, pinned tool versions, storage format, Trustee contract, appraisal policy id/digest |
| `attestation_policy.rego`, `reference_values.json` | Versioned AS CPU appraisal policy and approved boot/TCB/configuration references for KBS-admin installation |
| `resource_policy.rego` | Reusable rule for this bundle: exact bundle id and measurements, approved CPU appraisal, and requested binding equal to signed `init_data`. Contains no individual vault bindings. Installed with other active bundle rules (§7.5). |

**Profile set** `target/cvm_<profile_version>/profile_set.json` — generated by Stage 1, with one top-level `profile_version`, the shared vault-facing `contract`, and `bundles` keyed by `amd_sev_snp` or `intel_tdx`. Each entry contains only `build_id` and `manifest_sha256`; its directory is always the platform name next to `profile_set.json`, with no `path` field. The contract contains runtime, package, trust and vault-format settings, without a duplicate `profile_version`. Measurements and the cmdline digest remain in each bundle's manifest. Stage 2 verifies each manifest's top-level version, contract, build id and digest against the profile set.

**Generic CVM OCI deliverable** `target/cvm_<profile_version>/cvm_<profile_version>_<platform>.oci.tar` — a standard OCI image-layout tar with artifact type `application/vnd.nvidia.cvm.bundle.v1`. It contains one deterministic gzip-compressed CVM layer with the platform bundle and, after finalization, a single-platform `profile_set.json`. Deferred construction emits a pending artifact for transfer to its target TEE host; finalization and approval replace it atomically with the artifact for the new state. Only an approved artifact is a production deliverable. `./cvmctl pull --merge` combines finalized platform artifacts only when their profile version and complete shared contract match, verifies each profile entry against the included manifest, and updates the aggregate profile set under a lock. `oci_artifacts.json` records both the outer archive SHA-256 and the OCI manifest digest.

**Vault bundle** `target/vault_<deployment_id>/<platform>/` (one sealed copy per target platform; `deployment_id` is generated once per Stage 2 run)

| File | Purpose |
|---|---|
| `vault.qcow2` | Authenticated LUKS2 (`aes-xts-random` + `hmac-sha256`, dm-integrity journal), ext4 inside; format and validation requirements in §6.4. Application content identical across copies in one run, excluding each copy's identity manifest; header, UUID, unlock secret and ciphertext unique to this sealed copy. |
| `applog.qcow2` | Clear ext4 output sidecar, writable by the CVM so the operator can inspect logs without a KBS key. Its contents are public and untrusted. Do not write confidential logs here. |
| `user_config.qcow2`, `user_data.qcow2` | Clear ext4 input sidecars populated from optional operator-supplied directories. QEMU exposes both read-only, the guest mounts both `ro,noload,nosuid,nodev,noexec`, and the container receives read-only bind mounts. Their contents are host-readable and untrusted. |
| `cvm_bundle/` | Complete byte-for-byte copy of the approved generic platform bundle, including its root disk, kernel, initramfs, firmware, manifest, approval receipt and policy artifacts. Stage 2 copies these files and never rebuilds them. |
| `launch_cvm.sh` | Zero-argument launcher rendered for this copy. It verifies the embedded `cvm_bundle/`, checks bundle id and host TEE, auto-selects exactly `gpu_count` NVIDIA display devices for GPU profiles, binds every function in each isolated slot to VFIO, gives each GPU root port a 256 GiB prefetchable MMIO reserve, uses IOMMUFD for TDX GPU assignment, and restores host drivers at exit. `--cvm-bundle` and `--gpu` remain explicit overrides. Exclusive image locking refuses an attached vault (§4.1, D8). |
| `shutdown_cvm.sh` | Zero-argument shutdown wrapper. It validates the root-owned PID/start-time record for this delivery and signals that exact launcher, which requests an ACPI power-off through a root-only QMP socket, terminates QEMU only after a bounded grace period, and waits until QEMU has exited and released the vault. Both wrappers refuse a delivery directory that is not root-owned or is group- or world-writable. |
| `vault_manifest.json` | Public delivery metadata: `platform`, `profile_version`, the single `cvm_build_id` this copy is sealed for, the vault `luks_uuid`, `vault_bind`, header length, storage format, derived KBS resource path and required bundle-policy revision. Never an authority for guest key selection; this is a delivery artifact, not a registration ledger. |
| `README.txt` | Operator notes |

Each copy is packaged separately (`vault_<deployment_id>_<platform>.oci.tar`) as one self-contained OCI artifact with artifact type `application/vnd.nvidia.cvm.delivery.v1`. Its first deterministic layer contains the byte-for-byte generic platform bundle under `cvm_bundle/`; its second contains the sealed vault, sidecars, launchers, delivery manifest and launcher modules. Registry storage can therefore deduplicate the unchanged CVM layer across application releases. Stage 2 does not rebuild or reapprove those generic bytes. After verified materialization, `launch_cvm.sh` discovers and verifies `cvm_bundle/` without a path argument or another download. The same OCI layout tar works offline; `./cvmctl publish` copies it to a registry, and `./cvmctl pull` accepts either the tar or an immutable `registry/repository@sha256:...` reference. A copy is authorized for exactly one platform bundle: its key exists only under that bundle id, whose rule requires that bundle's measurements. A TDX guest cannot retrieve a key under the SNP bundle prefix; the corresponding key does not exist under the TDX prefix. The in-guest manifest also checks platform and build id. Operators who run a site on both host types receive separately sealed copies that share neither keys nor runtime state. The same generic bundles are reused for application A, application B and later releases of either application. An application update replaces vault copies and key resources while retaining the source generic CVM bundles, reference values and resource policy.

OCI descriptor digests authenticate content relative to the selected top-level digest; they do not identify its publisher. Production registry references use an immutable manifest digest and a verified release signature, while offline distribution authenticates the outer archive SHA-256 through a trusted release channel. Mutable tags are publication aliases only. Materialization verifies the layout, manifest, config and every layer digest, rejects links and path traversal, and atomically creates the familiar runtime directory. Registry transport uses HTTPS by default; `--plain-http` is an explicit test-only opt-in for an isolated lab registry. OCI is the transport and storage envelope; the measured-boot manifest, approval, vault binding and KBS policy remain the runtime security authorities.

**Exclusive attachment and copying (D8).** A vault image file must be attached to at most one CVM at any time. Copying a stopped, detached vault for deployment, transfer or backup is allowed; each runtime copy has its own vault and sidecar files and is assigned to one CVM. Separate file copies may be used by separate CVMs on the authorized platform; runtime writes affect only the copied vault and `applog`. A byte-for-byte copy retains the original UUID, unlock secret, binding and KBS resource path; it is not a new cryptographic identity. Use Stage 2 re-sealing when independent keys or revocation are required, and always for a different platform (D1).

The launcher requires exclusive QEMU image locking and fails if the vault is already attached or an exclusive lock cannot be obtained. It must not disable image locking or enable shared writable attachment. A restart or reassignment proceeds only after the previous QEMU process has exited and released the image, including after a slow shutdown or crash; there is no overlap window. Operators coordinate transfers between hosts and copy only detached images. This is an operational attachment rule: KBS still does not detect or prevent separately copied authorized vaults (§6.2). The guest needs no coordination between CVMs sharing a vault because that attachment mode is unsupported.

### 4.2 Partition / mount layout

| Disk serial | Image | Mount | Mode | Protection |
|---|---|---|---|---|
| `cvm-root` | `verity_root.qcow2` | `/` (overlay lower at `/rofs`) | ro + tmpfs upper | dm-verity, hash in measured cmdline |
| — | tmpfs | `/cow` (overlay upper/work), `/tmp` | rw, RAM | TEE memory encryption; `/cow` has the measured `root_overlay_max_mib` capacity limit |
| `cvm-applog` | `applog.qcow2` | `/applog` | rw | Clear ext4; intentional CVM-to-operator output, host-readable without a key |
| `cvm-user-config` | `user_config.qcow2` | `/user_config` | ro | Clear ext4; QEMU block node and guest mount are read-only; untrusted input |
| `cvm-user-data` | `user_data.qcow2` | `/user_data` (optional NFS is separate at `/nfs_data`) | ro | Clear ext4; QEMU block node and guest mount are read-only; untrusted input |
| `cvm-vault` | `vault.qcow2` | `/vault` | rw | Authenticated LUKS2 over dm-integrity, unlock secret from KBS |

The launcher assigns these stable SCSI serials. The initramfs and verified runtime
select `/dev/disk/by-id/scsi-0QEMU_QEMU_HARDDISK_<serial>` and wait for the required
device. Linux `/dev/sdX` letters depend on asynchronous probing and must never
identify a disk's role. Serial selection provides reliable routing; dm-verity,
local header binding and authenticated LUKS2 provide the security checks.

Inside `/vault`: `docker/` (image archive and runtime data-root, see §7.3), `config/` (authenticated generic application configuration), optional `application/` (opaque application files), optional `services/` and `scripts/`, and `vault_manifest.json`. The container's own filesystem and entrypoint are defined by its image; no host-side Python venv, framework package or framework-specific directory is required. An application may supply additional files under `application/` without assigning them meaning in the builder.

**Platform-neutrality rule for `/vault`.** The same populated content is sealed into every platform copy. Application launch configuration and optional guest-side services must not hard-code a TEE device, attester, firmware or platform. The per-copy identity manifest records the target platform and bundle without selecting platform tooling. `cvm_app.service` and any optional application services obtain required TEE settings from `EnvironmentFile=/run/cvm/platform.env`, which the generic root's `cvm_bootstrap.service` writes at boot (`TEE_PLATFORM`, `TEE_DEVICE=/dev/sev-guest|/dev/tdx_guest`, `TEE_DEVICE_ARGS=--device …`). TEE-device access is opt-in for applications that need it; ordinary containers require no attestation tooling. Platform report adapters and CPU appraisal live on the generic root in `cvm.host.platforms` and `cvm.runtime.attestation`; `cvm.runtime.gpu` sets CUDA readiness after composite appraisal at Trustee. The builder rejects hard-coded platform selection in configuration and services it installs. Application archives remain opaque workload content; the builder does not interpret framework files or claim that a text scan can prove arbitrary container code is platform-neutral. Compatibility with every selected bundle is part of workload validation.

### 4.3 Build and rebuild lifecycle

| Change | Generic CVM bundles (one per platform) | Application vault (content once, one sealed copy per platform) |
|---|---|---|
| First profile version | Build and measure once per platform; provision RVPS references and reusable bundle rules; publish the profile set | Populate content once per deployment site; seal one copy per target platform and upload each key |
| Add a platform to an existing profile version (e.g. TDX after SNP) | Build and measure the new bundle from the same pins; provision its references/rule and extend the profile set | Existing copies unchanged. For sites that need it, run Stage 2 with `platforms: [intel_tdx]` to seal a new copy and upload its key (content re-populated from pinned inputs per D3; byte-identity with earlier copies is not guaranteed) |
| Another application using the same profile version | Reuse | Populate and seal a new vault (content and copies) for the new application and each site |
| New application release, application dependencies, configuration or embedded secrets | Reuse | Re-populate and re-seal the affected vaults: new content and new copies with new UUIDs, secrets and bindings |
| Generic infrastructure changes: OS/kernel/firmware, Docker, GPU driver, attestation tools, RAM-backed root capacity or baked-in KBS settings | Build and measure a new profile version on every supported platform | Populate and seal new vault revisions bound to that profile version before deployments switch over |
| Move a deployment from an SNP host to a TDX host (or back) | Copy the other platform's generic bundle of the same profile version | Boot the site's copy sealed for the other platform. Copies do not share runtime state; any mutable state that must carry over is exported and re-imported by the application's own procedure |
| Run one site on two hosts of different platforms | Each host uses its platform's generic bundle | Each host uses its own sealed copy, never the same image |
| Copy a vault for another CVM on the same authorized platform | Reuse | Copy the stopped, detached image to a separate writable file with separate sidecars; dedicate each runtime file copy to one CVM (D8) |
| Restart an unchanged deployment or write normal application runtime data | Reuse | Reuse the existing writable copy; restart only after the previous QEMU process has exited and released it (D8) |

An application release runs Stage 2 only when a profile set already exists. “Build once” applies to each generic infrastructure version per platform; application changes alone do not trigger Stage 1, and adding a platform seals new copies without touching existing ones. Vault replacement follows the key-provisioning and rollout rules in §§7.2 and 12; preserving mutable application state across releases or across copies is part of the application's upgrade procedure.

---

### 4.4 Implementation packages

The Python implementation uses the standalone `cvm` namespace under
`nvflare/lighter/cc/image_builder`. `cvm.build` owns both build stages and the
standard-library construction provisioner; `cvm.runtime` executes inside the
completed guest; `cvm.host` owns QEMU, VFIO and physical-host checks;
`cvm.trustee` contains operator/client tooling for the existing CoCo Trustee;
`cvm.artifacts` handles bundle/approval validation and OCI transport;
`cvm.common` contains shared formats, pure policy generation and low-level
utilities. No package imports NVFlare.

Dependencies point toward shared contracts: `common` depends on no other CVM
package; `runtime` and `artifacts` depend only on `common`; `host` and `trustee`
may also use `artifacts`; `build` orchestrates host, artifact and Trustee client
operations. Host and guest platform access are separate from pure measurement
parsing. Build configuration is separate from shared application validation.
Image construction is separate from shared LUKS validation.

`cvm/build/payload.py` explicitly lists deployment modules. The provisioner
installs only runtime/shared modules under `/usr/lib/cvm/cvm`, and guest units
invoke `cvm.runtime.bootstrap` or `cvm.runtime.integrity`. Deliveries contain
host launch and bundle-verification modules without build/admin code. The
construction provisioner is sent separately and never installed in the final
image. Tests enforce package boundaries and import each staged payload in an
isolated interpreter, including checks for lazy-import dependencies.

The existing `runtime_source_sha256` contract field retains its conservative
scope: the complete `cvm` source tree (including the provisioner and payload
lists), systemd units, initramfs scripts and launcher templates. The package
migration therefore requires rebuilt generic bundles, fresh measurements and
new approval; it does not reuse earlier acceptance receipts.

## 5. Boot flow

```
QEMU launch
 │  firmware + kernel + initrd + cmdline measured (SNP launch digest / TDX MRTD,RTMR[1..2])
 │  vault_bind included in signed HOSTDATA (SNP) / MRCONFIGID (TDX)
 ▼
initramfs  (local-top/verity_root)
 │  parse roothash=, verity_hash_offset=, cvm.root_overlay_max_mib= from /proc/cmdline
 │  select root by serial cvm-root; open verity_root with panic-on-corruption
 ▼
initramfs  (local-bottom/overlay_root)
 │  tmpfs /cow ; overlay(lower=/rofs, upper=/cow/upper) → $rootmnt
 ▼
switch_root → systemd
 │  nftables.service loads measured /etc/nftables.conf before network-pre.target
 │  network-online.target → chrony.service
 ▼
cvm_bootstrap.service  (Type=notify; long-lived supervisor)
 │  emit reference frames before waiting for disks; verify nft table inet cvm
 │  require synchronized time, ≤0.5 s remaining correction and ≤1000 ppm skew
 │  1. select cvm-vault by serial; snapshot its fixed LUKS header into protected memory
 │     compute vault_bind; require a local TEE binding check to succeed (§6.2)
 │  2. detect platform (sev_guest vs tdx_guest);
 │     attest with the matching snp or tdx attester → Trustee
 │     fresh CPU report/quote (+ CCEL for TDX), KBS nonce and ephemeral TEE public key
 │     GPU profiles: NVAT evidence shares that nonce/key; CPU REPORT_DATA covers its bytes
 │     AS verifies NRAS JWTs and applies the immutable GPU policy and RVPS versions
 │  3. get-resource keys/<cvm_build_id from verified root>/<binding-id from checked header>
 │     KBS checks bundle measurements and path binding against signed evidence (§6.2)
 │     GPU profiles additionally require exact, distinct, affirming NVIDIA gpu0..gpuN-1
 │     (resource and secret are specific to this sealed copy and its platform)
 │     receive exactly 64 binary secret bytes in locked memory; never log them
 │  4. cryptsetup open the vault with its frozen header and protected key FD; erase the key
 │  5. start vault integrity monitoring; scan the decrypted vault for authentication errors
 │     mount /vault; validate bundle/platform/profile and vault UUID; then mount clear sidecars
 │     (`applog` rw; `user_config` and `user_data` ro,noload,nosuid,nodev,noexec)
 │     write /run/cvm/binding.json and /run/cvm/platform.env
 │  6. apply application firewall and hosts, mount optional NFS, install app units
 │  7. send READY=1, then synchronously start cvm_app.service and app_*.service
 ▼
docker.service → cvm_app.service (docker run … image from /vault)
 ▼
bootstrap supervisor: isolated periodic child every five minutes or on SIGUSR1
 │  CPU/composite GPU appraisal + key re-authorization; 300-second child deadline
```

Every gate must succeed before the application services can start. Before the first KBS contact, bootstrap narrows the measured firewall to the DNS resolvers the guest learned from DHCP; a missing, empty or invalid resolver list denies DNS. DNS checks precede conntrack and general egress allows. Only the measured discovery rules explicitly permit unrestricted DNS before the runtime resolver list is available. Measured NTS-only time sources default to `1.ntp.ubuntu.com`, `2.ntp.ubuntu.com` and `3.ntp.ubuntu.com`; profiles may supply another non-empty authenticated list. DHCP hooks and distribution source includes cannot add time sources. Both normal and initrd GPT automount discovery are disabled on the measured kernel command line. Clock synchronization is a measured-root gate before the first CPU or GPU attestation, and is checked again before periodic appraisal. On a bootstrap or monitor failure, PID 1 executes `FailureAction=poweroff-force`; bootstrap and monitor clean exits also execute `SuccessAction=poweroff-force`. A failed *periodic* check enters quarantine instead (D19): the supervisor stops the application units and the container daemons, unmounts `/vault` and closes its mapping so the key leaves the kernel, then retries a fresh appraisal and key retrieval once a minute for up to fifteen minutes. A successful `reopen` child re-checks the same vault identity, re-mounts and restarts the units; an exhausted window, or any failure inside the quarantine sequence, ends in the forced power-off. No Python failure-handler process is required. One CPU attest-plus-resource transaction has a 60-second monotonic budget shared by both network commands; a silent DROP cannot consume independent command timeouts. GPU profiles use one 240-second composite transaction budget, including at most 60 seconds for evidence collection and 180 seconds for backend NRAS work; the periodic child has a 300-second outer deadline. These conservative limits require timing acceptance on the exact hardware profile. The integrity monitor remains active throughout runtime (§6.4). The internal `/vault/vault_manifest.json` records `platform`, `profile_version`, the single `cvm_build_id` the copy was sealed for, all encrypted-disk LUKS UUIDs and storage format; it deliberately omits the header digest to avoid a build-time dependency cycle. The external delivery manifest is informational.

Audit output uses one daemon writer and a bounded eight-record queue. Host-controlled journal/console/applog I/O never runs on the supervisor's authorization or revocation path; congested or exiting processes may lose advisory records. PID 1 independently enforces a 360-second watchdog during steady-state waiting, appraisal and workload start. Before failure cleanup and quarantine revocation, the supervisor reduces that deadline to 180 seconds; an unresponsive stop, unmount or close therefore forces power-off. After workload/key removal, each reopen attempt and retry sleep has its own watchdog deadline within the fifteen-minute quarantine window, with ten seconds of cleanup allowance. Notifications have a one-second socket timeout. Fatal bootstrap errors request `WATCHDOG=trigger` before best-effort diagnostics.

Every disk role is mounted explicitly as ext4. Before mounting public output, the guest reformats `/applog` without an ext4 journal and eagerly initializes inode tables. Logs from previous boots must be copied off before restart; they are discarded at the next boot. This removes stale metadata/journal recovery at initial mount. It does not authenticate clear storage against concurrent malicious host mutation: the ext4 kernel parser remains part of the trusted computing base for clear sidecars, and host-controlled disk stalls remain a DoS risk. Keep secrets and confidential logs in the authenticated vault.

Generic units live in `/usr/lib/systemd/system`; validated application units go in `/run/systemd/system`. Bootstrap completes the vault, sidecar, firewall, hosts, NFS and unit-installation steps before sending `READY=1`. This completes its systemd start job, so the following synchronous `systemctl start cvm_app.service app_*.service` cannot wait on bootstrap's own unfinished start job. The application services require and bind to `cvm_bootstrap.service`. No workload target or asynchronous-start special case remains.

The integrity monitor has no dependency on bootstrap. `systemctl start cvm_integrity.service` therefore waits synchronously for the monitor's `READY` before any authenticated scan read. Bootstrap verifies the monitor is still active after the scan. Neither the initramfs nor `/etc/fstab` independently opens or mounts `/vault`.

Reference launches emit the unchanged `CVM_REFERENCE_V2` serial frames before the disk wait. The measured `console=ttyS0` command line and bootstrap's `StandardOutput=journal+console` carry the frames to the existing collector. A reference launch has no vault disk and subsequently fails closed.

### 5.1 Why attestation moves out of the initramfs (G8)

The initramfs must not make KBS calls. Attestation and vault unlock run as a systemd service in the verified root, providing:

- Full networking stack (DNS, proxies, NIC drivers) and a normal TLS store instead of `configure_networking` in busybox.
- The attestation client binaries, KBS URL and CA cert live on the **measured root**, not in the initramfs hooks, so changing KBS settings no longer needs `-i` initramfs rebuilds. The KBS URL and CA are part of the root image and therefore part of the measurement.
- The same code path serves SNP and TDX; the initramfs stays platform-neutral (verity + overlay only).
- Guest journals remain inside the volatile overlay. They are never forwarded wholesale to `/applog`, because application tracebacks may contain confidential values. Only allowlisted advisory attestation metadata and explicitly public application output belong on that clear disk.

The root itself needs no secret to open, so nothing security-relevant is lost by doing key release after `switch_root`. The overlay upper is tmpfs; nothing written before the vault is mounted leaks to disk. Its maximum writable capacity is `root_overlay_max_mib` MiB from the generic profile, defaulting to half of `memory_gib` when omitted. The builder validates that it is a positive integer no larger than guest RAM and places the resolved value on the measured kernel command line. The limit includes upper-layer file data and filesystem metadata; it does not apply to the separately mounted encrypted vault, clear sidecars or `/tmp`.

Consequences for the initramfs contents: drop the `cc_binaries` hook (`kbs-client`, `snpguest`, `kbs.cert`, `xxd`, `base64`) and the `network` prereq; keep `veritysetup`, `dm_verity` and `overlay`. The platform attestation drivers, `dm_crypt`, `dm_integrity` and vault cryptographic algorithms load from the normal root. `init_app.sh` is replaced by a minimal `initramfs/scripts/local-top/verity_root` script that has no network access. `initramfs/scripts/local-bottom/overlay_root` preserves only the overlay assembly and never mounts the vault.

---

## 6. Measurement and key binding

### 6.1 What is measured, per platform

| Component | AMD SEV-SNP | Intel TDX (direct boot with TDVF) |
|---|---|---|
| Firmware | launch digest (`snp.measurement`) | `mr_td` |
| Kernel, initrd, cmdline | launch digest via `kernel-hashes=on` | TDVF measured boot across `rtmr_1` and `rtmr_2`; record both for the pinned firmware |
| Boot variables / TD HOB / CFV | — | `rtmr_0` (pinned because the profile fixes memory size, CPU count and launch shape) |
| OS loader | — (none, direct boot) | `rtmr_1`, including the loaded kernel EFI image on the supported TDVF path |
| Per-instance binding (`vault_bind`) | `HOSTDATA` (32 B, `host-data=` on `sev-snp-guest`) | `MRCONFIGID` (48 B, `mrconfigid=` on `tdx-guest`) |
| Freshness / channel binding | `REPORT_DATA` binds the KBS nonce and ephemeral TEE public key through pinned RCAR | same, in TD quote `report_data` |

The dm-verity root hash is on the kernel command line, as in Kata's confidential guest rootfs. Changing that hash changes measured boot; changing root blocks while keeping the approved hash causes dm-verity to fail. Stage 1 records the exact command-line bytes and measurement inputs. Stage 2 copies that command line unchanged: no `vault_bind`, UUID, site id, KBS resource path or other per-site substitution. HOSTDATA/MRCONFIGID supply the independent instance binding.

### 6.2 Binding the vault to the CVM (G4)

The binding follows CoCo's initdata principle: hardware reports a host-supplied digest, trusted guest software checks the actual configuration against it, and KBS checks that the requested resource matches the attested value. This design uses a domain-separated header digest directly, rather than the serialized CoCo initdata document format. No list of individual header digests is installed in RVPS or the resource policy.

```
H = first header_bytes bytes of the guest-visible vault block device
vault_bind = SHA-256( UTF8("nvflare-vault-v2") ‖ 0x00 ‖ H )   # 32 B, unique per sealed copy
   SNP: -object sev-snp-guest,…,host-data=<base64(vault_bind)>
   TDX: -object tdx-guest,…,mrconfigid=<base64(vault_bind ‖ 16 zero bytes)>
   cmdline: exact generic-bundle cmdline of the platform bundle in use, unchanged for every vault
```

The domain-separation string is a fixed protocol constant, not an application identifier; every workload uses the same value.

Each sealed copy has its own header and therefore its own `vault_bind`. The copy sealed for SNP has its digest placed in `host-data` by its launcher; the copy sealed for TDX has its different digest zero-padded into `mrconfigid`. The builder stores each secret only under the intended bundle id and binding. That namespace restriction, together with the bundle's measurement check, authorizes the copy for exactly one platform bundle; the two sealed copies of one site share no cryptographic material.

**Deterministic resource identity.** `cvm-bundle-id` is the existing `cvm_build_id`, not a new identifier or registry entry. Stage 1 chooses an immutable unique value matching `[a-z0-9][a-z0-9_-]{0,63}`, writes it to `/etc/cvm_build_id` before measuring the root, and records it as `build_id` in the generic manifest. Never reuse an id for another bundle. The guest obtains it only from the verified root; the builder obtains it from the trusted profile set. `vault-binding-id` encodes the checked header digest as specified below. The LUKS UUID remains diagnostic and manifest metadata; it no longer selects the key.

```
SNP binding-id = lowercase_hex(vault_bind)                # 64 hex characters
TDX binding-id = lowercase_hex(vault_bind ‖ 16 zero bytes) # 96 hex characters
client resource = keys/<cvm_build_id>/<binding-id>
KBS resource URI = kbs:///keys/<cvm_build_id>/<binding-id>  # repository/type/tag
```

The platform-specific encodings preserve a simple, lossless conversion from each verifier's signed claim without treating binary digests as UTF-8 in Rego. They do not affect application content. Accept only canonical encodings and exactly these path components; no UUID aliases, extra segments or alternate namespaces.

**Decision (2026-09-11): per-platform copies are re-sealed (D1).** Each platform receives a separately sealed copy so its unlock secret, KBS resource and authorization lifecycle are independent of the other platform. A byte-for-byte file copy retains its existing platform authorization and cryptographic identity; copying alone does not authorize it on a different platform. Such file copies are permitted under D8, with each runtime file dedicated to one CVM. Re-sealing costs one extra integrity initialisation and content copy per platform at build time. Exclusive attachment applies to every runtime file regardless of how it was produced.

- The storage profile fixes `header_bytes = 16777216` and the payload offset at 16 MiB; formatting must explicitly produce and validate that layout, including both LUKS metadata copies and the complete keyslot region. Hash logical block bytes through NBD, never the qcow2 container's first bytes. Freeze the header after construction; unsupported offsets or layouts are rejected. **Builder and guest must read the same bytes by the same definition:** the builder hashes bytes `[0, header_bytes)` of the LUKS-formatted block device via NBD; the guest hashes bytes `[0, header_bytes)` of the stable `cvm-vault` device (the qcow2 is transparent, so this is the same LUKS block device). The acceptance matrix (§12.1) asserts byte-equality of the two digests for a sealed copy; a builder/guest byte-range mismatch is a build failure, not a tolerated difference.
- The guest reads `H` once into a sealed memory-backed file, computes the digest and checks it against its **local hardware report's** HOSTDATA or MRCONFIGID. For TDX compare all 48 bytes, including zero padding. Use the selected attester's `bind_init_data` operation where supported by the pinned guest-components version, or an equivalent platform adapter. Require an explicit successful verification result; a no-op or unsupported operation is failure. This check is an explicit responsibility of the new unlock helper, not an assumed side effect of the `kbs-client` CLI.
- Only after that check may the helper request a key. The RCAR evidence comes from the same local TEE and selected platform; those instance fields cannot change during its lifetime. Derive the resource path from `/etc/cvm_build_id` and the digest of the checked header, never from a sidecar, cmdline or unverified manifest. KBS independently reconstructs the permitted path from the bundle id fixed in its matching measurement rule and the signed `init_data` claim. Exact path equality is required before resource lookup/release.
- `cryptsetup open` must use the **same frozen header snapshot**, via `--header` and a memory-backed FD, with the stable `cvm-vault` device as the payload device. Re-reading an untrusted header after checking it would permit a check/use race. **All keyslot material used to unlock must be read from the frozen memfd header, not re-read from the stable `cvm-vault` device**; this is the property that makes the frozen-header check meaningful. Reading all LUKS metadata (keyslots included) from the `--header` file is documented cryptsetup behavior; verify it anyway against the pinned cryptsetup version. Validate the resulting cipher, integrity algorithm, offsets and mapping before mounting any filesystem.
- **Why vault A cannot obtain vault B's key while running A (G4).** HOSTDATA/MRCONFIGID is host-set, not derived by hardware from the attached disk. Three checks provide the link: (1) the measured guest derives its path from its own bundle id and the digest of the attached, frozen header; (2) the local check rejects attached-A with host-set `bind(B)` before any key request; (3) the shared KBS rule compares the requested binding with signed evidence, so an A-attested request for B's path fails even when both vaults use the same generic measurement. Passing the local check and requesting B's path requires actually using B's header and `bind(B)`; altered payload still fails authentication. A different bundle cannot retrieve B's key by changing the prefix because that prefix's rule requires B's approved bundle measurements. A made-up binding does not create a key: only the trusted builder/admin can provision resources (§7.5). The local check and dynamic path comparison are both required. Exercise the cross-vault request using an in-guest test harness, since the production helper never emits it.
- HOSTDATA/MRCONFIGID are host-set configuration fields, not proofs of disk attachment. The guest check provides that link. A copied header with altered payload passes the binding check but fails payload authentication (§6.4); it cannot authorize execution of arbitrary replacement content. A complete copy of a valid vault can boot on another approved instance. Preventing that requires additional owner/instance authorization and is outside G4.

The claim/path contract uses unmodified CoCo Trustee v0.22.0 (`512fed65642015b849f38fb13bfdec7806639987`), paired with CoCo v0.23.0. KBS, AS, RVPS and the guest client use upstream source without CVM patches. Production configuration and acceptance requirements remain in §7.5. These namespaces and encodings are intentionally distinct:

| Value | SNP | TDX |
|---|---|---|
| QEMU instance-field argument | base64 of 32-byte digest | base64 of digest plus 16 zero bytes |
| EAR `ev["init_data"]` | lowercase hex of HOSTDATA (64 characters) | lowercase hex of the complete 48-byte MRCONFIGID (96 characters) |
| EAR measurement claims | `snp.measurement`, lowercase hex (96 characters) | `tdx.quote.body.mr_td`, `rtmr_0`, `rtmr_1`, `rtmr_2`, lowercase hex |
| Resource binding id | lowercase hex of the 32-byte digest (64 characters) | same 96-character lowercase hex as EAR, including 16 zero bytes |
| Client resource path | `keys/<SNP_build_id>/<SNP_binding_id>` | `keys/<TDX_build_id>/<TDX_binding_id>` |
| Policy request path | `data.plugin == "resource"` and `data["resource-path"] == ["keys", "<SNP_build_id>", "<SNP_binding_id>"]` | corresponding TDX bundle and binding |

The KBS HTTP router supplies `plugin`, the resource-path **array** without the plugin, and query parameters as policy **data**; EAR claims are **input**. The verifier's top-level `init_data` is lifted alongside `snp`/`tdx` into annotated evidence without changing its encoding. Pin and test the KBS, AS, verifier and client versions as a set; pinning only the client does not establish this contract. Sources at the pinned commit: [policy engine](https://github.com/confidential-containers/trustee/blob/512fed65642015b849f38fb13bfdec7806639987/deps/policy-engine/src/policy/rego.rs), [HTTP router](https://github.com/confidential-containers/trustee/blob/512fed65642015b849f38fb13bfdec7806639987/kbs/src/api_server.rs), [SNP claims](https://github.com/confidential-containers/trustee/blob/512fed65642015b849f38fb13bfdec7806639987/deps/verifier/src/snp/mod.rs), [TDX claims](https://github.com/confidential-containers/trustee/blob/512fed65642015b849f38fb13bfdec7806639987/deps/verifier/src/tdx/claims.rs).

The resource policy contains one rule per approved generic bundle. `cvm/common/policy.py` generates Rego v1 for the pinned engine; it requires a fresh token (at most five minutes old and five minutes total lifetime), the configured issuer, favorable CPU appraisal, approved launch measurements, and exact resource path. For GPU profiles it additionally requires all configured GPU appraisals, distinct identities and favorable trust vectors. The binding comes dynamically from evidence; there is no per-vault measurement list. For example, the final path conditions are:

```rego
# SNP: binding_id is the canonical lowercase-hex HOSTDATA.
data.plugin == "resource"
data["resource-path"] == ["keys", "<SNP_BUILD_ID>", binding_id]

# TDX: the binding is the full zero-padded MRCONFIGID.
data.plugin == "resource"
data["resource-path"] == ["keys", "<TDX_BUILD_ID>", ev["init_data"]]
```

These are excerpts; install the complete generated resource policy, including freshness, measurement and appraisal checks.

**Appraisal policy is a required artifact.** The KBS admin installs the profile’s reviewed AS CPU policy and its RVPS reference set before authorizing vaults. These references describe generic boot measurements and approved CPU security configurations; they do not enumerate vault bindings. The AS returns `executables=3`, `hardware=2`, `configuration=2` only when all required checks pass: approved boot measurements; SNP debug and migration-agent permissions disabled plus approved TCB/configuration; or TDX debug disabled, approved module/TCB/XFAM, acceptable DCAP status and unexpired collateral. TDX direct boot also requires a replay-verified CCEL consistent with the pinned boot profile. Missing reference values or claims fail the appraisal. The AS exposes the verified instance field for the KBS's dynamic comparison. Broad fallback branches from sample policies are not part of this production policy.

The upstream `kbs-client` uses the `default` AS policy selector. Install the approved policy as `default_cpu.rego` (and `default_gpu.rego` for GPU profiles), with policy content digests and the profile version recorded in the bundle and deployment receipt. Separate security profiles use separate configured policy/reference storage with the same upstream Trustee distribution. Read-only mounts and upstream role ACLs prevent policy replacement; the policy name alone never establishes approval. KBS verifies signer trust before applying the resource policy. Unfavorable, missing, stale, differently appraised or incorrectly typed claims deny release. AS Rego uses `query_reference_value()` and emits `trust_claims`; a companion RVPS reference carries per-value expiry deadlines, checked by Rego because v0.22 RVPS returns stored values without enforcing metadata expiry.

The reference Trustee configuration explicitly sets `attestation_service.verifier_config.dcap_verifier.tcb_update_type` to `standard`. Upstream v0.22.0 otherwise defaults to `early` and fetches collateral independently of the host's QCNL configuration. Intel's standard channel includes a mitigation deployment grace period; early applies newer TCB recovery requirements. The deployment operator records the selected channel alongside policy and reference approvals. TDX appraisal requires `UpToDate` and unexpired collateral under that channel; there is no fallback that accepts `OutOfDate`. Deployments selecting early must qualify their firmware against that baseline. See [TRUSTEE_GUIDE.md](TRUSTEE_GUIDE.md#4-configure-upstream-kbs).

Production operation requires working HOSTDATA/MRCONFIGID and guest binding verification. Unsupported QEMU/firmware/attester combinations are rejected. A command-line-only binding would require a separately specified per-vault measurement mode; there is no silent fallback that violates G1.

### 6.3 Replay and freshness

The guest deliberately delegates RCAR challenge/ephemeral-TEE-key binding and encrypted resource-response handling to the measured, pinned upstream `kbs-client` and trusted KBS/AS implementation. `validate_token()` independently checks the EAR signature, identity, policy, freshness and TEE claims, but does **not** independently reconstruct the client's RCAR transcript or verify its report-data binding. Correct upstream RCAR processing is an explicit trusted-computing-base assumption, alongside the measured KBS CA and separate AS signing key. The recorded binary digest identifies the selected executable; source checkout checks and that digest do not constitute a signed proof that the binary was built from the checkout.

Trustee issues a challenge nonce and the pinned RCAR implementation binds it and the ephemeral TEE public key into `REPORT_DATA`/`report_data`. Retain the implementation's exact serialization and hash algorithm rather than introducing a new concatenation scheme. KBS verifies that binding and wraps the resource to that public key; stale reports and expired tokens are rejected.

Both the guest token check and the resource policy require numeric timestamps, an unexpired token, `now - 300 <= iat <= now + 5`, `nbf <= now + 5` (defaulting to zero when absent), and `0 < exp - iat <= 300` seconds. The five-second future allowance does not extend expiration.

Disk rollback is a separate limitation: authenticated disk sectors can still be replayed with their previously valid tags and IVs at the same addresses, and a complete snapshot can be replayed. Header binding and fresh attestation do not prove the latest mutable filesystem state. Vault revisions with new secrets and UUIDs support administrative retirement of old revisions, but cannot detect rollback within an active revision or revoke a secret already held by a running guest. A deployment requiring those properties must add an external monotonic state service before relying on them.

### 6.4 Authenticate the confidential application vault

Header binding identifies the vault secret and parameters; it does not authenticate payload bytes. Production operation therefore applies LUKS2 authenticated disk encryption to `vault.qcow2`: `aes-xts-random` with `hmac-sha256`, a 512-bit XTS encryption key and the additional authentication key/IV material managed by cryptsetup. The 64 random bytes held by KBS are the vault's **keyslot unlock secret**, not a manually concatenated encryption/authentication volume key. The stack is ext4 → authenticated dm-crypt → dm-integrity → the vault block device. The three sidecars are outside this confidential boundary by design.

- Use dm-integrity's normal journal mode with initialized authentication metadata. Do not use an unkeyed checksum, recovery mode, integrity recalculation, skipped tag initialization, or a plain-XTS fallback. Record cipher, integrity algorithm, key sizes, sector size, header/payload offsets and tool/kernel versions in the storage profile. Require the actual activated targets to match that profile.
- By default (`vault_prescan: true`), before mounting the vault, read its complete decrypted mapping and fail on any authentication/I/O error. This catches existing corruption before startup. A measured profile may set `vault_prescan: false` to omit that proactive scan: corruption is then detected when the affected sector is read, which can occur after workload startup. Both profiles authenticate every read and require the integrity monitor; the opt-out must pass the dedicated acceptance row in §12.1. The internal manifest validates the vault UUID, generic build id and storage format only after the configured storage checks; it is not a substitute for authenticated reads.
- A generic-root integrity monitor starts before the scan and watches the vault's dm-crypt authentication-error path and device I/O failures, supplemented by target failure counters where exposed. Its own failure or an authentication/I/O failure during bootstrap stops startup; a runtime failure invokes PID 1 forced poweroff. Corrupt bytes must never be returned as usable plaintext. Validate the exact error surfaces, event delivery, monitor supervision and shutdown behavior in the pinned kernel/tool combination; do not assume an untested counter catches external authentication failures.
- Services, scripts, Python environments, the Docker archive, extracted layers and runtime data all stay on this authenticated mapping. Units copied into `/run` are read from the verified mapping before use. Runtime writes receive authentication tags through the same stack.

This preserves the existing mount contract. The vault needs space for tags, IVs and the journal, plus initialization and full-scan I/O costs; sizing and startup estimates must include them. The clear sidecars use their configured ext4 capacity without LUKS overhead. [Cryptsetup's authenticated-encryption documentation](https://gitlab.com/cryptsetup/cryptsetup/-/blob/main/man/cryptsetup.8.adoc) labels the feature experimental, so a production profile is enabled only after the exact builder/guest versions pass the storage and crash-recovery acceptance checks (§12.1). Unsupported profiles fail the build. The [dm-integrity documentation](https://docs.kernel.org/admin-guide/device-mapper/dm-integrity.html) describes the authenticated storage stack; the format does not provide disk rollback protection (§6.3).

**Deployability gate.** The entire payload-integrity story rests on authenticated LUKS2. The builder therefore has **no production profile, and is not deployable**, until a pinned cryptsetup/kernel combination passes the storage and crash-recovery acceptance checks in §12.1 (corruption detection, reboot-after-writes, interrupted journal, interrupted Docker load, runtime monitor failure). If validation of a candidate combination fails, the builder remains non-deployable for that profile version; there is no weaker-but-acceptable interim profile, and a confidentiality-only LUKS2 fallback would reintroduce the unauthenticated-payload gap the authenticated storage contract closes. Do not ship a profile that has not passed the matrix.

---

## 7. Build pipeline

### 7.1 Stage 1 — `cvmctl build [-p <platform>]` (build the generic CVM once per platform)

For the normal one-target workflow, run `cvmctl build` without arguments once on the target host. The command reads `config/cvm_profile.yml`, auto-detects the host platform, constructs the generic root, boots the exact result to collect reference evidence, and leaves a finalized unapproved bundle. The operator installs that exact manifest into an isolated acceptance Trustee with `admin install --candidate`, creates its test vault with `vault --candidate`, runs the matrix, and uses `acceptance-report` plus `admin approve` to publish the signed receipt. This ordering makes the exact manifest available to the tests that authorize it. A trusted site may automate the same steps through an explicit `--acceptance-runner`. The resulting approved profile set is reused by every compatible application and deployment.

**Platform selection.** An explicit `-p amd_sev_snp` or `-p intel_tdx` takes precedence and must name an enabled entry in `config/cvm_profile.yml`. Without `-p`, inspect the build host's CPU vendor and advertised SNP/TDX capability, using host capability interfaces where available; CPU vendor alone is insufficient. Select a platform only when exactly one supported, profile-enabled TEE is identified. An unavailable, unsupported or ambiguous result stops before building and reports that an explicit `-p` is needed; it must not silently choose SNP, TDX, vTPM or development mode. Log and record the resolved platform in the bundle manifest. Detection selects this Stage 1 build only; it does not select Stage 2's target-platform list.

An explicit `-p` with `--defer-measurements` is the advanced workflow for CI, non-CC build hosts and multi-platform profiles. Construction still uses the plain-VM path below, so the builder must not require matching host TEE devices merely to build an explicitly selected target. `./cvmctl finalize` completes the pending bundle on the target platform; `./cvmctl admin approve`, `./cvmctl admin install`, and `./cvmctl admin retire` expose the administrative stages through the same CLI. Reference-measurement validation still requires a suitable target-platform validation host (§7.4), and the delivered runtime launcher still rejects a mismatched host (§4.1).

Inputs: `config/cvm_profile.yml` (profile version, GPU on/off, KBS URL + CA, package pins, one pinned kernel shared across platforms (D4), guest memory, `root_overlay_max_mib`, and a `platforms:` map with per-platform firmware and attester settings), the default `inputs/ubuntu-26.04-server-cloudimg-amd64.img`, installed plain/SNP firmware, `inputs/OVMF.inteltdx.fd`, and `inputs/kbs-client`. A GPU profile also pins a reviewed `gpu_policy`, the upstream NVIDIA attester's `libnvat` shared library, the remote NRAS HTTPS URL and exact driver/container-toolkit packages. The checked-in file and `cvm.build.config.PROFILE_DEFAULTS` give every ordinary value and location. The platform-independent parts of the profile (base image, package pins, layout, storage profile) are shared by every platform build of that version, which is what makes the resulting bundles vault-compatible with each other.

GPU profiles also require `gpu_apt_repositories` (HTTPS source metadata and SHA-256-pinned public keyrings) and `gpu_attestation_provenance`. Stage 1 installs repository-scoped `Signed-By` sources before updating apt. The contract records those inputs plus the NVAT source revision, reviewed libxml2 compatibility patch and library digest; a soname or filename does not identify the source. [GPU_BUILD.md](GPU_BUILD.md) documents the Ubuntu 26.04 recipe. Trustee and guest-components remain unmodified.

1. Boot the base image as a plain VM (`cvm.build.cvm.plain_build`, no CC needed).
2. Send a fixed payload containing the source snapshot and reviewed public inputs to the disposable guest over temporary SSH. The standard-library Python provisioner `cvm/build/provisioning.py` verifies Ubuntu 26.04/x86-64, installs the exact package pins without starting daemons, and installs pinned Docker/containerd, the optional GPU driver + container toolkit, platform attestation tools, the unlock helper with local binding verification, authenticated-storage tools/modules, KBS settings in `/etc/cvm/runtime.json` with the public CA and AS verification key, and generic units (`cvm_bootstrap.service`, `cvm_integrity.service`, `cvm_app.service`). GPU profiles install the pinned `libnvat` bytes for the unmodified upstream client from the build payload; runtime never downloads or upgrades appraisal code. A precompiled kernel-module plus compute-library package set is valid and preferred for headless guests, while a complete driver meta package remains supported. It installs the storage profile and generic units described in §§5–6, disables swap and core dumps, masks crash collectors and login services, applies firewall default-deny at boot, locks passwords, and writes `/etc/cvm_build_id` and `/etc/cvm_profile_version`. It removes the temporary SSH credentials and masks **both** `ssh.service` and `ssh.socket` before shutdown. Build and Trustee hosts must also reject piped core collection before handling keys. **No** application image, framework package, application configuration or secrets. The build requires no configuration-management runtime.
3. Rebuild initramfs with `initramfs/scripts/local-top/verity_root`, overlay-only `initramfs/scripts/local-bottom/overlay_root` and modules `dm_verity overlay`. Platform attestation and vault crypto drivers remain on the normal root. Fetch `vmlinuz`/`initrd.img`.
4. Shut down; `cvm.build.storage.build_verity` produces `verity_root.qcow2`, the root hash and hash-tree offset. The measured command line includes `systemd.verity=no` because the initramfs already activates the verified root.
5. Compute reference measurements (§7.4) and write `cvm_manifest.json`. The normal call leaves it unapproved. Candidate vault and candidate administration modes accept this exact finalized manifest only for isolated acceptance. `acceptance-report` rejects evidence for another manifest or platform and requires the full CPU/platform/GPU matrix before `admin approve` signs the receipt (D18). An explicit trusted runner may automate the same flow. Stage 2 and production `admin install` verify the signature against the public keys pinned in `cvm_project.yml` and the administration configuration; a receipt file without a trusted signature is not approval. A manifest with `production_ready: false` can never receive that receipt.
6. Emit the versioned AS policy, full generic boot/TCB/configuration reference set and this bundle's reusable resource-policy rule (§6.2). The KBS admin validates their provenance, provisions the reference values in RVPS and installs the rule alongside other active bundles (§7.5). Validate the effective AS policy selection and resource policy. Stage 2 requires this bundle to be enabled; vault creation never adds reference values or changes policy.
7. Update `profile_set.json`: add this platform's `build_id`, measurements and cmdline digest; recompute and verify the shared vault-facing contract (storage profile, header bytes, layout, kernel version (D4), Docker/containerd and Python versions, `root_overlay_max_mib`, Trustee commit). If the new bundle's contract differs from bundles already in the set, the build fails; the profile version must be bumped instead.
8. Package the platform bundle and single-platform profile set as `cvm_<profile_version>_<platform>.oci.tar`. Normalize layer ordering, ownership and timestamps so unchanged content has the same layer digest. Record the manifest and archive digests in `oci_artifacts.json`. A deferred artifact is an intermediate transfer object; finalization and approval regenerate it, and only the approved digest may be published as a production bundle. For a multi-platform Vault Build, materialize the first finalized artifact and import later platform artifacts with `./cvmctl pull --merge`; refuse a different profile contract, duplicate platform identity or manifest mismatch.

Runtime ~ today's full build per platform. Re-run only to create a new generic infrastructure version, such as for an OS/kernel/firmware, Docker, driver, attestation-tooling or baked-in KBS configuration change, or to add a platform to an existing profile version. Adding an application or releasing an application update uses Stage 2 with the existing profile set.

### 7.2 Stage 2 — `cvmctl vault` (content once per application release; one sealed copy per target platform)

Run `cvmctl vault <vault_build.yml>` for each application release and deployment. `cvm_image` selects a local generic CVM folder or an immutable OCI registry reference; reuse its verified profile set unchanged. The builder generates one UUID-based deployment ID shared by the run's output folder, manifests and archive names. A rebuild produces new vault copies and KBS key resources; it does not rebuild or re-measure any generic CVM, change generic reference values or update the resource policy. One run populates the content once and seals it into one independent copy per target platform (`platforms:` is optional; default: every platform available in the selected CVM image). Copies are sealed inside the same run from the populated mapping; no plaintext vault image is written to the builder's disk.

Inputs: `vault_build.yml` (§9.2), a pinned profile set, a Docker save archive containing the application's image, optional encrypted application files, and optional clear source directories for the read-only `user_config` and `user_data` sidecars. The builder rejects private-key material and symlinks in either clear input tree. It validates the remaining generic schema, copies supplied content and renders generic runtime configuration; it neither invokes application-specific provisioning playbooks nor installs an application framework or venv. Container dependencies belong in the supplied image. Optional guest-side executables must be prepared by the application owner for the profile's shared base image, package pins and final paths; host-dependent binaries or relocated venvs are not assumed compatible. Enforce the configuration and service rules from §4.2 on the installed launch material.

Byte-identity of application content across copies, excluding per-copy identity manifests and filesystem metadata, is guaranteed only within this one run (D3): there is no retained master, so a later run for a newly added platform re-populates from the same pinned inputs. The builder must document the content-stability guarantees it relies on (pinned input archives and package versions, reproducible timestamps) and records each copy's content digest in its manifest, so drift between separately built copies of the same release is detectable rather than assumed away.

1. For the first target platform, create `<platform>/vault.qcow2` and attach it through NBD. Generate a fresh 64-byte binary unlock secret in locked memory; format authenticated LUKS2 with the validated profile (§6.4), initialise integrity metadata, open the mapping and create ext4. Populate this encrypted mapping directly, avoiding a persistent `vault_plain.img` scratch copy. Route build caches/scratch to this mapping or protected temporary memory.
2. Copy the Docker archive into `/vault/docker/` and optional application files into `/vault/application/`. Derive the root or numeric runtime UID/GID from the authenticated Docker image configuration and apply it to `/vault/application` inside the mounted encrypted vault; named image users are rejected, and source staging remains owned by the invoking user. Write validated runtime settings to `/vault/config/application.json`; this is the authenticated runtime projection of the generic YAML build configuration, with build-host paths omitted. Render the generic container launch service and any declared optional services. Use `EnvironmentFile=/run/cvm/platform.env` when TEE settings are needed. Do not synthesize framework files, install framework packages or replace the supplied image's entrypoint unless the application configuration explicitly overrides it. Record a content digest of the populated tree (excluding the manifest).
3. For each additional target platform, create `<platform>/vault.qcow2`, generate its own fresh unlock secret, format and initialise it exactly as in step 1, and copy the populated tree mapping-to-mapping (`tar | tar`; plaintext exists only in transit through builder memory). Verify the content digest matches. Then, for every copy, write its internal manifest (`platform`, `profile_version`, that platform's `cvm_build_id` from the profile set, UUID, storage format), flush/unmount/close the mapping and freeze its header. Compute each copy's `vault_bind` from its own logical header bytes and record that platform's QEMU and policy encodings (§6.2).
4. Reopen and verify each completed authenticated vault, including its internal manifest. Confirm opening/closing does not change the bound header. Create one clear sidecar set per copy: an empty writable `applog` output image plus read-only `user_config` and `user_data` images populated from their optional source directories. Re-scan the completed input images for private keys and symlinks, then write each copy's informational external manifest.
5. Render each copy's launch and shutdown wrappers for its platform. Copy the exact approved generic bundle into `cvm_bundle/`. The launcher discovers that embedded bundle and verifies its build id; `--cvm-bundle` remains an override for advanced shared-bundle placement. A GPU profile auto-detects every assignable NVIDIA display controller and requires the exact measured-profile `gpu_count`. It binds every function in each isolated PCI slot to `vfio-pci`, attaches each GPU through a dedicated PCIe root port with a 256 GiB 64-bit prefetchable MMIO reserve, and restores the original host drivers after QEMU exits; repeatable `--gpu` options select an explicit set. TDX attaches the VFIO devices through a shared QEMU IOMMUFD object so private guest pages are not registered through the legacy VFIO DMA-mapping path. Refuse mixed-slot IOMMU groups, a mismatched host TEE, and attachment while another CVM holds the vault (§4.1, D8). Record launcher/QEMU PIDs plus Linux process start times in a root-owned runtime directory so `shutdown_cvm.sh` can signal the exact instance and wait for lock release. Copy the generic command line byte-for-byte and reject overrides of measured inputs/launch shape. No per-site cmdline fields are added.
6. Derive each copy's `keys/<cvm_build_id>/<vault-binding-id>` path from the trusted profile set and frozen header (§6.2). Finish image verification, launcher generation and delivery metadata before uploading any secret. Install the bundle's reference/policy revision through the operator's approval workflow first. Stage 2 validates the path locally and does not accept a path supplied by an untrusted operator. It does not install policy or register measurements.
7. Upload the exact 64-byte binary secret at that path through native Trustee resource POST using a scoped administrative token (§7.5). **Successful storage activates access immediately** under the existing bundle rule. Independent copies use different paths, so vault builds do not share a policy-update lock. An ambiguous response is retained as an uncertain outcome without an automatic retry. Native POST can overwrite an existing key; native DELETE has no permanent tombstone. The CoCo operator controls credential scope, upload fencing, storage durability and backup recovery.
8. Package each verified copy separately as `vault_<deployment_id>_<platform>.oci.tar` after its own key upload succeeds. Use one deterministic generic-CVM layer and one vault/runtime layer under a single `application/vnd.nvidia.cvm.delivery.v1` manifest. Include the complete copied `cvm_bundle/`, sealed vault, sidecars, launchers and launcher modules so the artifact alone is sufficient to launch on its target host after materialization. Record the OCI manifest digest and outer archive SHA-256 in `oci_artifacts.json`. Publication may copy the same layout to an OCI registry; production consumers select and verify its immutable signed digest. Copies are independent; there is no transaction spanning all target platforms. If the run is abandoned after an upload, that resource is already authorized: delete/disable it through the trusted backend operation, and retain its path in build-job diagnostics if cleanup cannot complete (§7.5). Never claim a partially completed run's uploaded keys are still denied. Close mappings, detach NBD and erase transient secrets on success and failure. If a key file is needed, it is an owner-only memory-backed FD; prevent swapping and core dumps rather than treating `shred`/unlink as sufficient protection. Existing user-supplied input archives remain the builder operator's responsibility.

No CC hardware or QEMU boot is needed. Work scales with content population plus, per copy, one capacity-dependent integrity initialisation, one content copy and one validation pass; measure build time for supported vault sizes and platform counts rather than promising a fixed number of minutes.

### 7.3 Docker image loading

Today the image is `docker load`ed at build time so it lands in `/var/lib/docker` on the root. With a read-only root and `/vault` as the only persistent writable store, the Docker data-root must live in `/vault`:

- Set `"data-root": "/vault/docker/data"` in `/etc/docker/daemon.json` on the generic image; `docker.service` gets `After=cvm_bootstrap.service cvm_integrity.service` and `Requires=cvm_bootstrap.service cvm_integrity.service` (matching §10).
- Start with an archive-only vault: after vault verification, load the image on first boot and launch the expected image id. The load runs in the owning CVM's ordered startup path. Record success on authenticated storage only after `docker load` succeeds and the expected image id is present; later boots reuse that verified Docker state. If loading is interrupted before success is recorded, the next boot verifies the state and retries the load before starting the application. Partial Docker state or an unfinished progress record never counts as completion, and no workload starts on a failed or mismatched load. The previous CVM must have exited and released the vault before this next boot (§4.1, D8). No cross-CVM load lock, loading-marker heartbeat or stale-holder election is needed. Pre-populating a data-root is deferred until daemon-version compatibility is tested.
- Pin the Docker/containerd storage configuration. Any independently configured containerd content/snapshot store must also live under `/vault`; moving Docker's `data-root` alone is insufficient for such a configuration. These daemons and socket activation paths require successful vault initialization and active integrity monitoring.

### 7.4 Reference measurement computation

Today the builder boots the CVM once in CC mode and scrapes `Measurement:` from the console. Keep that as a fallback but prefer offline computation, as the CoCo guidance does:

- **SNP:** `sev-snp-measure --mode snp --vcpus N --vcpu-type EPYC-v4 --ovmf OVMF.amdsev.fd --kernel vmlinuz --initrd initrd.img --append "<cmdline>"`. The vCPU count and type are launch parameters, so `launch_cvm.sh` pins them and the manifest records them. Note `host-data` does not affect the launch digest.
- **TDX:** compute `mr_td`, `rtmr_0`, `rtmr_1` and `rtmr_2`, and the expected CCEL measurements for the exact TDVF/kernel/initrd/generic cmdline. Select a pinned measurement tool compatible with that firmware, or collect a quote and CCEL from a trusted Stage 1 validation boot and replay the log. Do not assume RTMR1 is only a constant separator. `rtmr_0` depends on the TD HOB; this design pins the launch shape and therefore records and enforces RTMR0 with the other TDX measurements.

All required values are written into `cvm_manifest.json`, the profile set and the reference set so Stage 2 needs no CC hardware of either platform. Validate offline values against an actual quote/report once per platform bundle. Console-scraped values alone are not trusted registration evidence on an untrusted operator host. Two independently sealed vault copies on one bundle must have identical measured boot inputs and different instance-binding fields (a byte-for-byte file copy retains its binding under D8); the two copies of one site, sealed for different platforms, must present different UUIDs, bindings and measurements, and neither may satisfy the other platform's rule (§12.1).

### 7.5 Trustee provisioning and lifecycle without a per-vault registry (D9)

The backend installation and key lifecycle procedure, including sample TLS and native administration configurations, is in [TRUSTEE_GUIDE.md](TRUSTEE_GUIDE.md).

Kata/CoCo separates [reference values](https://confidentialcontainers.org/docs/attestation/reference-values/), [secret resources](https://confidentialcontainers.org/docs/attestation/resources/) and [resource policies](https://confidentialcontainers.org/docs/attestation/policies/). Reference values remain necessary to appraise a trusted guest; a separate application-builder registration file is not required by that architecture. This design adopts that separation and adds the binding-addressed resource convention in §6.2. It does not claim that Kata supplies this exact convention or that an affirming CPU appraisal alone isolates application keys.

| State | Authority and lifecycle |
|---|---|
| Generic bundle identity, measurements and policy artifacts | Trusted, versioned Stage 1 bundles and profile sets. Changed when approving or retiring generic infrastructure or its appraisal requirements. No list of individual vaults. |
| Approved boot/TCB/configuration reference values | Trustee RVPS, provisioned from approved Stage 1 artifacts. Maintained per generic bundle/security revision, with no per-app header-digest registration. |
| Active resource policy | KBS configuration assembled from approved bundle rules. Every rule checks appraisal, exact bundle measurements and dynamic path-to-evidence binding. Unchanged when applications or vault copies are added. |
| Per-vault unlock secret | KBS resource backend at `keys/<cvm_build_id>/<vault-binding-id>`. Trusted creation authorizes future release under the existing rule; deletion/disablement prevents future retrieval. |
| Vault delivery manifest and build-job audit | Records intended bundle, image identity and resource path for delivery, retry, cleanup and revocation. Not an authorization input or a globally reconciled inventory used to regenerate policy. |

**Access control.** Bundle-policy and AS/RVPS administration belongs to the trusted CoCo deployment operator. Vault builders receive pre-issued bearer JWTs scoped to native KBS resource endpoints, never the token issuer's signing key. The upstream endpoint ACL permits resource POST and DELETE; it does not enforce create-only uploads or prohibit deletion by a resource-role holder. Because a leaked token can therefore replace or delete every key its role reaches, the reference ACL confines paths to canonical binding identifiers, deployments should issue one bundle-scoped `cvm-resources-<build_id>` role per approved bundle (`cvmctl admin acl`), and the builder refuses tokens that are expired or valid for more than 30 days. Guest attestation credentials permit policy-authorized retrieval, never administration. Do not embed administrative tokens in guests or deliveries.

**Bundle approval and policy changes.** Start default-deny with no resources (D6). Provision and validate a bundle's RVPS references and AS policy, then install its reusable resource rule before accepting vault key uploads. Assemble the complete policy from the active *bundle* artifacts, preserving other approved bundles. Serialize these infrequent administrative configuration changes and read back the installed policy; there is no global policy mutation on the Stage 2 path. Keep versioned administrative audit/configuration for recovery. A rollback must preserve current retirements and security requirements; a formerly restricted policy is not safe to restore if it re-enables a retired bundle. In Trustee v0.22.0, `GET /kbs/v0/resource-policy` returns policy IDs. Administration verifies that listing, then compares the exact Rego bytes in the shared `storage/kbs/resource-policy.rego` file with the intended artifact. The configured storage must be the same volume used by KBS; silently normalizing policy bytes is not permitted.

**Production Trustee requirements.** Use the existing unmodified Trustee v0.22.0 deployment managed by CoCo v0.23.0. CVM Builder installs no backend services or systemd units. Configure authenticated resource and policy roles, separate AS signing trust, reviewed CPU/GPU default policies and endorsed RVPS references. Native resource storage must be writable by KBS. Record source revision, clean-source provenance, binary/image digest, policy hashes and references. Test that resource tokens cannot change policy and policy tokens cannot mutate keys, and test actual CPU/GPU appraisal. The v0.22 resource-policy GET returns policy IDs, so the offline publisher verifies bytes in the mounted policy namespace; named RVPS values use `/reference-value/<name>`.

**Resource activation, retry and recovery.** After validating each freshly sealed vault, Stage 2 sends one native resource POST with its 64-byte secret. Trustee's resource API permits replacement; the builder supplies a fresh binding-addressed path and never implements a racy read-before-write check. It does not automatically retry uncertain uploads. A request may have succeeded before a timeout or crash, so retain `provisioning.json`, `build_failure.json` and the encrypted artifact for operator recovery. If the secret is lost, build a fresh vault and resolve the abandoned resource instead of replacing its key. Backend durability, concurrent administrative writes and backup recovery follow the selected CoCo storage backend's guarantees; immutable create-only publication is no longer supplied by a custom service.

**Revocation.** Fence active uploads, then delete a sealed vault's exact resource through native KBS DELETE. Deletion prevents future retrieval while the resource remains absent, but creates no permanent tombstone: an authorized POST or backup restore can reintroduce it. The CoCo operator must preserve deletions, current policy and credential restrictions across recovery. Byte-for-byte vault copies share a resource and are revoked together (D8). Retiring a generic bundle removes its reusable allow rule and records retirement in the publisher's admin state; stored keys need not be deleted. A failed publication is not a completed retirement; retry and verify readback before reporting success. Deletion cannot retract secrets already released to a running CVM; periodic authorization and shutdown retain the limits in §§6.3 and 10.

This removes the previously proposed per-vault inventory and policy publisher. It does **not** remove Trustee's persistent reference values, secrets or administrative configuration. The existing Stage 1 artifacts and KBS backend supply the required state; adding an application does not require a new measurement-registration record.

---

## 8. Platform specifics

### 8.1 AMD SEV-SNP

- QEMU: `-object sev-snp-guest,id=sev0,cbitpos=…,reduced-phys-bits=1,policy=<launch_shape.snp_policy>,kernel-hashes=on,host-data=<b64>` + `-bios OVMF.amdsev.fd -kernel -initrd -append`. The policy word is derived at Stage 1 from the approved `snp_smt_allowed`, `snp_single_socket` and guest ABI references, recorded in the manifest's launch shape, and validated by the launcher to exclude debug and migration; the AS policy compares the reported fields with the same references.
- Guest evidence: `/dev/sev-guest` via the selected `snp` attester. Trustee verifies against AMD endorsements; EAR exposes `snp.measurement`, top-level `init_data` (lowercase-hex HOSTDATA), `snp.policy_debug_allowed`, `snp.policy_migrate_ma` and TCB/configuration claims. The guest locally checks HOSTDATA before requesting a key; KBS enforces the appraisal as well as measurement and binding.
- **VCEK availability.** Use CoCo Trustee's upstream SNP collateral handling, including its documented offline certificate store when KDS outage tolerance is required. Do not assume the removed CVM-specific cache or its timeout constants. Qualify initial and periodic appraisal on the selected deployment, including KDS loss and changed chip/TCB values; freshness and bounded guest shutdown remain required.
- For a GPU profile, `launch_cvm.sh` detects exactly `gpu_count` assignable NVIDIA display GPUs, binds every function in each isolated PCI slot to `vfio-pci`, and passes each slot through a dedicated QEMU PCIe root port. Each port reserves a 256 GiB 64-bit prefetchable MMIO window so a device with a 128 GiB BAR, including the validated H800, can be assigned. The measured GPU-profile command line includes `pci=realloc,nocrs` so Linux allocates the devices' large BARs. TDX connects every device to a shared IOMMUFD backend; legacy VFIO attempts to map private TDX RAM and can exhaust the host DMA-mapping limit. The measured guest verifies the visible count, configures `nvidia-container-runtime`, appraises all GPUs, and starts the application container with `docker run --gpus all`. Repeat `--gpu PCI_ADDRESS` exactly `gpu_count` times for explicit placement. TEE-device mapping into the container remains a separate application opt-in.

### 8.2 Intel TDX

**Mechanism (as used by CoCo/Kata on bare metal).** The TD boots TDVF (`OVMF.inteltdx.fd`) with QEMU direct boot. TDVF measures the launch configuration and boot chain across RTMR0/RTMR1/RTMR2 and publishes the **CCEL** event log. The guest's pinned `tdx` attester obtains a quote using the supported TDX/configfs-tsm interface and includes `/sys/firmware/acpi/tables/data/CCEL` in its evidence. Trustee verifies the quote and replays the supplied log. The production AS policy requires that log; the verifier alone can accept evidence without one. Quote generation requires a tested QGS connection through the host's configured vsock/QEMU quote-generation path.

- QEMU: `-machine q35,kernel-irqchip=split,confidential-guest-support=tdx0 -object tdx-guest,id=tdx0,mrconfigid=<b64>,quote-generation-socket.type=vsock,… -bios OVMF.inteltdx.fd -kernel -initrd -append`. TDX has no `kernel-hashes` flag; measurement of kernel/initrd/cmdline is intrinsic to TDVF direct boot.
- Guest kernel ≥ 6.7 for configfs-tsm; Ubuntu 24.04 HWE or Canonical's TDX kernel. Per D4 this one pinned kernel also serves the SNP bundle; per-platform modules (`tdx_guest`, `tsm` here; `sev_guest` on SNP) come from the same kernel build.
- The bundle rule pins `mr_td`, `rtmr_0`, `rtmr_1`, `rtmr_2` and the approved AS policy, then matches the requested resource binding against the hex-encoded padded `init_data`. The guest checks all 48 MRCONFIGID bytes before requesting a key. RTMR0 is pinned because the profile fixes the launch shape; any HOB or launch-configuration drift is denied.

**About TPM-based TDX measurement.** CoCo supports separate attestation paths with different trust assumptions:

1. `az-tdx-vtpm` / `tpm` attesters — used on Azure CVMs and other platforms whose TD *contains* a paravisor-hosted vTPM. There the TPM quote over PCRs 0–15 is the evidence (PCR 4/11 carry kernel+initrd+cmdline; CoCo binds initdata into PCR 8). The vTPM is inside the TCB because the cloud provider's paravisor is measured into `MRTD`.
2. Bare-metal Kata/CoCo TDX — no TPM at all; RTMR + CCEL as described above.

On our bare-metal QEMU deployment a `swtpm`-backed vTPM runs **on the untrusted host**, outside the TD. Its PCRs are not bound to the TD's hardware measurement. Production bare-metal deployments therefore selects the `tdx` attester explicitly. Reserve `intel_tdx_vtpm` as an extension for platforms with an attested vTPM trust chain; enable it only with its own evidence/claim contract, PCR binding procedure, appraisal and acceptance fixtures. The SNP/TDX policy above must never silently fall back to a generic `tpm` attester.

### 8.3 Non-CC / development mode

`launch_non_cc_vm.sh` keeps a mode without a TEE object for CI and debugging: the same root/initramfs boots, and `cvm_bootstrap.service` detects no TEE device. A *dev* generic build is marked by `/etc/cvm/dev_mode` on the measured root and, for dev builds only, uses an **unencrypted** vault (`vault.qcow2` is a plain ext4 image, no LUKS, no KBS call). Production generic builds never contain `dev_mode` and never read a key from a sidecar. The KBS never releases keys to a dev image: in a plain non-CC boot there is no TEE object and hence no evidence to appraise at all, and if the host attaches a TEE to a dev image, the dev root's measurement is not registered under any production profile (rule 2 below), so appraisal and policy deny the request.

**Hard rules for dev mode:** (1) a dev generic build cannot be converted to a production build without re-measuring — `dev_mode` lives on the measured root, so flipping it changes the launch digest and the production KBS rule no longer matches; (2) a dev build and a production build of the same platform must **never share a `profile_version`**, so a dev bundle can never satisfy a production vault's resource rule (which pins the production measurement); (3) the dev vault carries no real secrets — operators must not place production secrets in a dev vault, since it is unencrypted and host-readable. The earlier "key-on-`user_config`" dev path is dropped permanently: `user_config` is a clear read-only application input and is never a key transport. An unencrypted dev vault is strictly simpler and removes that path.

---

## 9. Configuration

### 9.1 Generic CVM profile — `config/cvm_profile.yml` (new; replaces most of `cc_build.yml`)

The excerpt below highlights shared settings and platform selection. The checked-in
configuration is a complete sample with conventional input locations and Ubuntu
26.04 pins. `cvm.build.config.PROFILE_DEFAULTS` supplies the same values when fields
are omitted; explicit profile values always take precedence. The quick test writes
its fully resolved generated YAML to `target/build-test/cvm_profile.yml` so every
selected value is inspectable.

```yaml
profile_version: cpu-2026.09      # shared by every platform bundle built from this file
production_ready: false           # checked-in failed pins cannot be approved
# acceptance_runner: /usr/local/bin/site_acceptance  # optional site automation
gpu: none                       # nvidia_cc requires additional validated GPU inputs
gpu_count: 1                    # exact count passed through and rechecked inside the guest
guest_release: '26.04'
base_image: ../inputs/ubuntu-26.04-server-cloudimg-amd64.img
build_firmware: /usr/share/ovmf/OVMF.fd
kbs_url: https://kbs.example.org:8443
kbs_cert: ../inputs/kbs-ca.pem
bootstrap_egress: [443, 8443]     # KBS + HTTPS egress baked into the image firewall; immutable pre-unlock path
vcpus: 4                          # pinned; part of the SNP measurement input / TD shape
memory_gib: 8
root_overlay_max_mib: 4096        # optional; defaults to half of memory_gib (4096 here)
trustee_commit: 512fed65642015b849f38fb13bfdec7806639987
attestation_policy_id: default  # upstream selector; approved content digest in manifest
vault_storage_profile: luks2-xts-random-hmac-sha256-v1
vault_header_bytes: 16777216       # fixed logical block range; validated at format/open
kernel_version: 7.0.0-31-generic   # ONE kernel for both platforms (D4)
required_system_packages: [...]    # complete exact values are in config/cvm_profile.yml

platforms:                         # auto-detected by cvmctl build; optional -p <name> override (§7.1)
  amd_sev_snp:
    firmware: /usr/share/ovmf/OVMF.amdsev.fd
    attester: snp
    kbs_client: ../inputs/kbs-client
    cpu_model: EPYC-v4
  intel_tdx:
    firmware: ../inputs/OVMF.inteltdx.fd
    attester: tdx
    kbs_client: ../inputs/kbs-client
    cpu_model: host
    quote_generation: {type: vsock, cid: 2, port: 4050}
```

Shared runtime, package, trust and vault-format settings form the vault-facing
contract (`cvm.build.cvm.contract`) and must match every bundle in a profile version.
Launch shape, firmware and measurements are recorded separately per platform.
Stage 1 refuses to add a bundle whose top-level profile version or contract differs
from the existing profile set.

### 9.2 Generic application and vault inputs — `vault_build.yml`

The builder accepts a standalone, application-neutral configuration. Required per-build inputs are `cvm_image`, `docker_archive` and the expected immutable `image_id`. Production builds also load shared `trustee` settings from the project configuration described below. `cvm_image` accepts a folder containing `profile_set.json` and fixed platform subdirectories, or a generic CVM OCI reference pinned as `registry/repository@sha256:<digest>`; `oci://` and `https://` prefixes are accepted. Registry input is verified and materialized temporarily through ORAS, then held until packaging completes; it must be a finalized generic CVM artifact with a valid approval receipt for production. `deployment_id` is generated automatically as UUID hex, printed, and recorded in `vault_set.json` and the vault manifests; it is not a configuration field. The four image sizes have checked-in defaults. Optional `platforms` selects a non-empty subset and defaults to every available platform; it is not inferred from the machine running Stage 2. A registry artifact supplies one platform; a local folder may aggregate several. `application_files` optionally supplies a prepared directory tree inside authenticated encrypted storage. `user_config` and `user_data` may supply content for their clear, read-only sidecars, and `hosts_entries` may supply literal host mappings. The builder creates either input sidecar empty when its source is omitted. It rejects symlinks, private-key filenames and containers, and private-key PEM markers anywhere in either clear input tree. None of these inputs requires a participant role, startup kit or framework installation.

`container` carries application launch settings: optional `entrypoint` and `command` arrays, environment values, declared volume mappings and port mappings. When overrides are absent, use the supplied image's own entrypoint and command. Preserve the generic mount contract (§10); reject settings that bypass vault verification, override measured boot inputs, disable workload failure propagation or make either input sidecar writable. TEE-device access is optional and resolved through `/run/cvm/platform.env`, never a hard-coded platform device. The builder does not inject an NVFlare command or assume any application protocol. Confidential application configuration and secrets are stored only in the authenticated vault; the clear sidecars and public delivery manifest contain no secret values.

Generic resource settings include `applog_drive_size`, `user_config_drive_size`, `user_data_drive_size`, `vault_drive_size`, `allowed_ports`, `allowed_out_ports` and `requires_gpu` (default `false`). The generic profile's `gpu_count` is an integer from 1 through 8 (default `1`) and becomes part of the measured-root contract even when the CPU-only profile leaves GPU support disabled. Eight GPUs cover the reference architecture's validated topology while keeping one dedicated QEMU PCIe root port per GPU. The project-level `trustee` mapping supplies the existing HTTPS endpoint, CA and scoped bearer-token file used to store each copy's key. The checked-in project and build samples supply every field; the drive-size defaults are 1, 1, 1 and 8 GiB respectively. As specified by D5, assert `requires_gpu == true` ⟺ profile `gpu: nvidia_cc`; a mismatch fails the build. Every selected platform must exist in the profile set. Additional workload capabilities are the application owner's profile-selection responsibility; the builder imposes no framework-specific validation.

Shared builder settings belong in `cvm_project.yml` at the calling project's root. Its `trustee` mapping requires `url`, `ca` and `admin_token_file`; credential paths resolve relative to that file. Stage 2 searches upward from the build YAML directory for the nearest project file, or uses `--project-config <path>` (relative to the invocation directory). It loads one file without merging or fallback from an invalid selection. Missing configuration fails before image retrieval or key creation. Per-build `trustee` is rejected. Candidate builds use the project Trustee configuration; plaintext `--dev` builds skip discovery and reject an explicit project file. Project configuration and client credentials remain on the build host and are excluded from runtime settings and deliveries.

Validate container port mappings against `allowed_ports`, which defines the guest inbound firewall allowance. Apply application inbound and outbound rules only after authenticating `/vault/config/application.json`; define the profile's immutable bootstrap egress separately (`bootstrap_egress:` in §9.1) so KBS remains reachable before unlock. Application rules must satisfy the profile's `bootstrap_egress` constraints. Guest time synchronization always permits UDP 123 and TCP 4460 for Ubuntu's NTS key exchange, before and after unlock; a first boot must not depend on cached NTS cookies from the build. The container forwarding rules retain their separate application egress allowance. Keep these rules out of the measured command line.

The builder owns the generic `/vault` layout in §4.2. Application-internal paths and optional payload contents are chosen by the application owner. Build-host source paths stay in `vault_build.yml`; only validated runtime values are copied into `/vault/config/application.json`.

### 9.3 Boundary with application provisioning

Any caller can invoke `cvmctl vault <vault_build.yml>` with a prepared application image, optional files and `cvm_image` pointing to a generic CVM folder or pinned registry artifact. Application provisioning owns application-specific configuration generation, credentials and payload preparation. Generic bundles are built separately with `cvmctl build` (auto-detect) or its optional `-p` override, once per platform and profile version; provisioning a new application or release must reuse existing compatible bundles.

NVFlare is the main consumer of this interface. Its provisioning layer owns the startup kit, `cc_params.yml`, participant roles, project schema and any translation into the generic builder inputs. It supplies the startup kit and any generated credentials under encrypted `application_files`, while non-secret operator inputs may use `user_config` or `user_data`, and selects an NVFlare container image, just as another application supplies its own image and files. The CVM Builder neither reads the NVFlare schema nor generates or interprets the startup kit. Changes to NVFlare provisioning and the details of that adapter are outside this design; they must not become mandatory arguments, services or package dependencies of the generic builder.

---

## 10. Runtime services

The image ships exactly three CVM unit files:

| Unit | Type | Ordering | Role |
|---|---|---|---|
| `cvm_bootstrap.service` | notify; long-lived with phase watchdog | After network-online, chrony, nftables, local filesystems, module loading and finalrd; before Docker/containerd | Emit reference frames; verify firewall and clock; bind, attest, unlock, scan and mount the vault and sidecars; apply app configuration and NFS; install app units; notify readiness; start the workload and supervise periodic re-authorization |
| `cvm_integrity.service` | notify; 15-second watchdog | Started synchronously before scan; before Docker/containerd; no bootstrap dependency | Monitor authenticated storage. Failed or clean exit invokes PID 1 forced poweroff |
| `cvm_app.service` | simple; `KillMode=mixed` | Requires/After bootstrap and Docker; After integrity; BindsTo bootstrap | Load and run the authenticated Docker image with `--cap-drop ALL` plus the admitted capabilities, `no-new-privileges` and a PID limit. On SIGTERM the wrapper stops the container itself (`docker stop --time 15`) and exits 0, so orderly shutdown and quarantine are not workload failures; an unexpected container exit keeps its status and invokes PID 1 forced poweroff |

Application-supplied `app_*.service` units receive `Requires=`, `After=` and `BindsTo=cvm_bootstrap.service`, `After=cvm_integrity.service`, `FailureAction=poweroff-force` and `/run/cvm/platform.env`. Bootstrap starts them explicitly after its readiness notification. They are generated from authenticated vault inputs and are not additional shipped CVM units.

The pinned distro `nftables.service` loads `/etc/nftables.conf` before networking. Stage 1 renders this file from `bootstrap_egress` with the same rule generator used for the application firewall; dm-verity measures it. Bootstrap verifies the table before KBS traffic. Chrony still runs as the distro service, with the bounded synchronization gate performed inside bootstrap and each periodic child.

Docker and containerd are disabled at boot and start through `cvm_app.service` only after the vault is ready. `docker.socket` is masked. Because the stock Docker unit requires that socket and uses `-H fd://`, provisioning removes that dependency and changes its listener to `-H unix:///var/run/docker.sock` in the measured vendor unit. No Docker/containerd CVM drop-ins are installed. Persistent daemon storage remains on `/vault`; system-wide core-dump restrictions remain enabled.

The locked bootstrap supervisor schedules check starts five minutes apart using a monotonic deadline and launches a fresh `periodic` Python child with a 300-second timeout. Workload startup and each child's runtime count toward the interval; an overrun causes an immediate next check without overlapping children. SIGUSR1 is blocked before `READY` and consumed synchronously, allowing a pending acceptance-test request to trigger a check as soon as startup completes and begin a new interval. Each tick atomically records its sequence, result and timing in `/run/cvm/periodic.json`. A failed tick attempts to revoke GPU readiness and exits nonzero even if revocation fails. Audit allow/deny records remain emitted by the measured bootstrap/periodic actions.

All three units carry `FailureAction=poweroff-force`. Bootstrap and integrity also carry `SuccessAction=poweroff-force`, since neither may exit while the guest runs. PID 1 performs forced shutdown without starting another Python process or enqueueing workload-stop transactions. Security failures bypass the application's graceful stop hook; PID 1 kills processes and unmounts filesystems. For ordinary shutdown, reverse unit ordering stops the application services, Docker and containerd before either supervisor can force the final shutdown phase.

The measured distro `finalrd.service` prepares its RAM-backed shutdown environment at boot, before bootstrap can attest or unlock the vault. It no longer waits for `ExecStop`, which a security power-off can bypass. The measured `cvm_shutdown.finalrd` hook includes `libmount`, `libblkid` and their dependencies; systemd loads these libraries dynamically, so finalrd's ordinary dependency scan misses them. Startup checks reject a shutdown image missing either library. This environment lets PID 1 pivot away from the overlay root, unmount its lower filesystem and detach dm-verity. The three CVM unit files and their fail-closed actions remain unchanged in number and purpose. Crash consistency must be revalidated on rebuilt images.

Development images use the same three unit files. Bootstrap omits TEE/KBS and integrity-monitor work, mounts the plain development vault, configures the application and starts the same supervisor. No unit-file dependency rewriting is needed.

## 11. Component responsibilities

| Area | Behavior |
|---|---|
| Build granularity | Generic CVM built once per platform per profile version; vault content populated once per application release/site and sealed into one independent copy per platform, without CC hardware |
| Platform selection | Stage 1 auto-detects the build host platform; `-p` is an optional explicit override. Stage 2 seals copies for the independently selected platforms in its generic configuration; the host runs the copy sealed for its TEE |
| Attestation location | `cvm_bootstrap.service` in verified root; KBS settings on measured root |
| Key resource | `keys/<cvm_build_id>/<vault-binding-id>` per sealed copy; one reusable rule per generic bundle checks positive appraisal, exact measurements and path binding equal to signed `init_data` |
| Platform-specific content in `/vault` | None; TEE device and attestation scripts come from the generic root via `/run/cvm/platform.env` |
| Vault ↔ CVM binding | `vault_bind` = digest of LUKS header, in HOSTDATA/MRCONFIGID and verified in-guest |
| Vault payload | Authenticated LUKS2 with dm-integrity; bootstrap scan and supervised runtime failure handling |
| Per-site cmdline | Exact generic cmdline; vault-specific binding only in the TEE configuration field |
| Platforms | SNP + TDX (RTMR/CCEL), optional TDX-vTPM profile |
| Measurement source | Offline `sev-snp-measure` / RTMR precompute; console fallback |
| Docker data-root | `/vault/docker/data` |
| Key handling at build | Binary unlock secret in locked memory/protected FD; populate encrypted mapping directly; cleanup on all exits |
| Resource policy updates | Install reusable rules when generic bundles are approved/retired; Stage 2 only uploads keys. No per-vault inventory or policy publisher (D9); restricted KBS remains (D6/D7) |
| Application inputs and execution | Generic Docker image, launch configuration and optional payload; NVFlare provisioning remains an external consumer |
| Firewall inbound ports | Generic root default-deny; application ports applied from authenticated `/vault/config/application.json` at boot |
| Entry points | `cvmctl` → `cvm.__main__` dispatches build, vault, OCI and administration commands; stage implementations live in `cvm.build.cvm` and `cvm.build.vault`, with shared storage helpers in `cvm.build.storage` |

The operator surface includes the overlay root, device and mount layout, Docker
application volume mapping, NFS configuration, GPU setup, and firewall and login
lockdown. Application provisioning supplies the image and opaque payload.
Stage 1 supplies bundle policy artifacts; Stage 2 provisions key resources.

---

## 12. Rollout plan

Provision each deployment with fresh vaults and explicitly approved bundle references.

1. **Prepare the restricted Trustee KBS (D6).** Start with default-deny policy and no key resources. Validate the pinned revision, AS policy selection, authenticated administration, policy read-back and backend key lifecycle (§7.5). Approve generic bundle references and reusable rules before uploading vault keys. Run the lifecycle, concurrency, revocation and rollback checks in §12.1.
2. **Validate the verified-root boot flow.** Verify the overlay, fstab and unit ordering, and confirm that unlock runs in the verified root. Every bootstrap failure must keep the workload stopped.
3. **Validate storage, binding and appraisal together.** Exercise authenticated LUKS2, frozen-header TEE checks, the unchanged generic command line, pinned AS policy/reference sets and resource policy. Create fresh vault UUIDs, secrets and bindings (D7).
4. **Validate the generic and application stages.** Exercise manifests and the profile set, platform selection, generic vault inputs, image/payload ingestion, Docker storage on `/vault` and the key-only provisioning sequence. Complete the SNP acceptance matrix and a non-NVFlare workload test.
5. **Validate the TDX profile.** Pin TDVF/QEMU/kernel/attester/QGS/DCAP versions; verify MRTD, RTMRs, padded binding and TDX appraisal. Run the same storage and isolation tests on TDX. Copies sealed together must share the application-content digest while having independent keys. Later runs detect input drift (D3); vTPM has separate acceptance requirements.
6. **Validate operations.** Policy rollback must preserve bundle retirements, and backup restoration must preserve revocations. Verify periodic authorization, structured auditing, recovery and operator procedures before approving a production profile.

Publication depends on the complete security acceptance matrix.

### 12.1 Acceptance checks required before production

These are implementation acceptance criteria, not claims of completed hardware validation. Run policy fixtures against the pinned engine, test the real KBS HTTP endpoint contract, and exercise boot/storage behavior on each supported SNP/TDX profile.

| Check | Required outcome |
|---|---|
| Two applications and an application update on one profile set | Stage 1 runs once per platform. Stage 2 creates each app/release/site vault and uploads keys only; generic bundles, measurements, RVPS references, rule count and policy bytes remain unchanged |
| Non-NVFlare container with no optional application payload | Builds from a Docker archive and generic configuration, then runs its declared image/entrypoint without NVFlare packages, roles, startup kits or provisioning files |
| Application provisioning boundary | The core builder accepts only the generic schema; external provisioning supplies opaque payload and translated settings. No NVFlare playbook or startup-kit generator is invoked by either build stage |
| Application storage encryption | NVFlare provisioning passes its generated kit through encrypted `application_files`. The completed `vault` requires a LUKS key and exposes an authenticated dm-integrity mapping only after attestation. `applog` is clear writable CVM output; `user_config` and `user_data` are clear inputs exposed read-only by QEMU, the guest and the container. A stopped sidecar can be read by the operator without a KBS key. |
| Directional sidecar enforcement | QEMU opens `user_config.qcow2` and `user_data.qcow2` read-only; the guest mounts both `ro,noload,nosuid,nodev,noexec`; the container receives read-only bind mounts. `/applog` remains the only writable clear sidecar, and its stopped image is directly readable without a KBS key. |
| Stage 1 without `-p` on an unambiguous supported host | Detects the profile-enabled SNP or TDX platform from host capabilities and records the resolved platform; an explicit matching selection produces the same target configuration |
| Stage 1 without `-p` when detection is unavailable, unsupported or ambiguous | Fails before building with a diagnostic requiring explicit selection; no silent platform, vTPM or dev fallback |
| Stage 1 with an explicit platform on a different or non-CC build host | Valid enabled `-p` takes precedence and permits construction; invalid or disabled values fail. Target-platform measurement validation remains required before approval |
| Omitted, explicit, zero or over-RAM `root_overlay_max_mib` | Omission resolves to half of `memory_gib`; an explicit positive value becomes the measured `/cow` tmpfs capacity; invalid values fail before construction. A boot missing or duplicating the measured command-line parameter fails in initramfs. |
| One site sealed for both platforms | One run yields two copies with identical application-content digests (excluding per-copy manifests) but different UUIDs, secrets, bindings, launchers and sidecars; each boots only on its bundle and receives only its own secret |
| Byte-for-byte duplicate of the SNP copy booted on the TDX bundle (and vice versa) | Launcher refuses the host mismatch. If forced, the key is absent under the running bundle's prefix, and a request under the original prefix fails its measurement rule. If a key were mistakenly provisioned there, the guest rejects the internal platform/build-id mismatch |
| Copy on a different generic bundle | Key absent under that bundle's prefix; requesting the intended bundle's prefix fails its measurement rule. If a key were incorrectly duplicated, the guest still rejects the internal build-id mismatch |
| Adding a platform to an existing profile version | Install the new bundle's references/rule once; existing copies and rules stay unchanged. New vault copies need key uploads only; a differing vault-facing contract is refused by Stage 1 |
| Vault A instance requests B's resource (in-guest test harness; same generic bundle) | Denied: the shared rule derives A's permitted path from signed evidence, so it cannot match B's requested path. No individual A/B bindings are stored in the policy |
| Header A with HOSTDATA/MRCONFIGID B; wrong TDX padding; unsupported binding adapter | Local check fails before key request; workload stays stopped |
| Header replaced between verification and open | Frozen checked header is used; substituted payload fails authentication |
| Builder (NBD) vs guest (the stable `cvm-vault` device) header digest for one sealed copy | Both compute `SHA-256` over `[0, header_bytes)` of the same LUKS block device and produce identical `vault_bind`; mismatch is a build failure |
| Real request to `/kbs/v0/resource/keys/<build_id>/<binding_id>` | Policy sees `data.plugin == "resource"` and `data["resource-path"] == ["keys", "<build_id>", "<binding_id>"]`; wrong namespace/prefix, extra segments, legacy UUID paths, noncanonical encodings and absent claims fail closed |
| Policy read-back after bundle-rule installation (§7.5) | Check the policy ID listing from `GET /kbs/v0/resource-policy`, then compare persisted Rego bytes in the shared KBS storage with the intended artifact. Mismatch blocks approval; Stage 2 performs no policy writes |
| Bootstrap starting application services (§5/§10) | `READY=1` precedes a synchronous start of `cvm_app.service` and generated app units; no activation deadlock and no workload before every gate succeeds |
| Reordered SCSI targets and asynchronous disk probing | Stable disk serials select the verified root, encrypted vault and each clear directional sidecar correctly; generic measurements and workload behavior remain unchanged |
| Integrity monitor active before the first authenticated scan (§5/§6.4) | `cvm_integrity.service` is started synchronously (no `Requires=cvm_bootstrap.service`) and signals readiness before the full-device scan begins; no scan read occurs before the monitor is watching |
| Measured `vault_prescan: false` profile with preexisting payload corruption and unchanged header | Corrupt an initialized, initially unread payload sector before boot; startup may succeed. A controlled uncached read must reject the corrupt sector and the integrity monitor must trigger bounded poweroff. Record the pinned kernel/cryptsetup versions and detection-to-poweroff timing; never claim proactive pre-startup detection for this profile |
| SNP 64-character and TDX 96-character hex binding fixtures | Correct canonical claims and derived paths accepted; SNP 96-character hex launch measurement verified. Wrong lengths/types, uppercase/short/base64 binding values, nonzero padding and wrong measurements fail closed |
| Debug enabled, migration-agent permitted, unapproved TCB, bad/expired DCAP collateral, missing/bad TDX event log | AS does not yield the required affirmative appraisal; KBS denies even when measurement/binding match |
| Signed negative EAR, wrong policy id, missing/incorrect trust-vector fields, expired token or stale evidence | Denied; no key released |
| Guest clock unavailable, unsynchronized or outside the configured correction/skew bounds | `bootstrap clock gate` fails before first CPU/GPU appraisal and before workload startup. A periodic clock-gate failure stops the workload and powers off. A synchronized clock is recorded in hardware evidence. |
| KBS/AS endpoint silently drops packets during periodic appraisal | The single CPU attest-plus-resource deadline expires within 60 seconds; including workload teardown, the CVM powers off within 105 seconds. No retry layer can reset or multiply that budget. |
| Provisioned upstream SNP offline collateral followed by AMD KDS egress block | Fresh genuine SNP appraisal succeeds from approved collateral for that chip/TCB; invalid or missing collateral cannot affirm. Verify guest timeout and shutdown behavior when appraisal cannot finish. |
| GPU appraisal denial or NRAS silent DROP on a GPU profile | KBS denies key release before vault unlock. A periodic composite failure revokes CUDA readiness and powers off. Confirm CC-disabled, tampered/replayed/missing evidence, exact count/policy, no allow record or vault mapper, and bounded shutdown on the exact profile; hardware evidence is pending. |
| `ssh.service`, `ssh.socket` and TCP port 22 in the final guest | Both units are stopped, disabled and masked; neither IPv4 nor IPv6 has a port-22 listener. A connection cannot socket-activate sshd. |
| Modify one 4 KiB logical block of `verity_root.qcow2`, then update only the untrusted outer artifact hash | Launch measurements and the recorded dm-verity root hash stay unchanged, but dm-verity rejects the root before `switch_root`; no application or login starts. This exercises the root-integrity requirement. |
| Payload, tag, IV or storage metadata corruption with unchanged header | Authentication/activation failure; no unauthenticated plaintext or workload startup |
| Corruption after bootstrap scan; integrity monitor failure | Failed reads remain errors; workload stops and guest powers off |
| Integrity monitor process crash/killed or clean exit | PID 1 executes `FailureAction=poweroff-force` or `SuccessAction=poweroff-force`; no Python failure handler or workload target is involved |
| Dev generic build booted against a production vault's KBS rule | Denied: dev and production share no `profile_version`, so the production rule's measurement never matches the dev image's |
| Reboot after legitimate writes, crash during a journal update or interrupted Docker load | Valid state recovers with intact header binding; uncertain/corrupt state fails closed |
| Attempt to attach one vault image file to a second CVM | Launcher/QEMU refuses the second attachment before the guest can access the vault; exclusive image locking remains enabled |
| Slow shutdown followed by an immediate restart request | No overlapping attachment: the new launch is refused until the previous QEMU process exits and releases the image; a subsequent launch succeeds |
| Copy a stopped, detached vault for separate CVMs on the same authorized platform | Each CVM uses a separate writable file copy and sidecars; writes do not affect the other copy. UUID, secret, binding and KBS resource path are retained; no new cryptographic identity is claimed |
| Boot 1 crashes during `docker load` before success is recorded | After the old QEMU process exits and releases the image, boot 2 retries the interrupted load, verifies the expected image id and records success before starting the app; partial state is never treated as complete |
| Unsupported credential or image requests a vault key | Denied; the restricted KBS accepts only approved bundle measurements and exact binding-addressed resources (D6/D7). |
| Concurrent builds for distinct vaults | Upload distinct key paths with no policy mutation or lost updates; both authorized vaults retrieve their own keys |
| Unknown binding with valid appraisal and matching requested path | The rule may match, but absent key resource prevents release; attestation alone cannot create or enumerate keys |
| Upload failure/timeout/crash before or after durable commit | Each new seal uses a fresh secret/header/binding and resource path. The builder records uncertain upload state and never automatically retries. Native Trustee POST can overwrite an existing resource; no overwrite guard is claimed. A committed resource is already authorized. Fence abandoned writers, reconcile delivery or delete the abandoned resource, and preserve that fencing across recovery before rebuilding with a fresh identity (§7.5) |
| Fence uploads, delete a vault key, then restore a backend backup | Fresh retrieval fails for that identity and its byte-for-byte copies, while independent vaults still work. Verify that operator fencing and recovery preserve the deletion and current policy; native Trustee supplies no tombstone and an authorized POST or an unfiltered restore can recreate the resource. Previously released keys remain a documented limitation |
| Retire a bundle or roll back policy/configuration | All keys under its prefix stay denied; other approved bundles remain usable. Rollback cannot restore retired rules or broaden release |
| Unauthenticated policy/reference/resource mutation; builder attempts policy write | Denied at every exposed administrative route. Resource-only builder access cannot alter AS/RVPS/resource policy |
| Configured AS policy and immutable content | Actual EAR uses the configured `default` selector and required appraisal outputs. Reviewed CPU/GPU policy digests match the bundle; unauthorized replacement is denied by role ACLs and read-only mounts. |
| Replay of previously valid sectors/snapshot | Recorded as the known rollback limitation (§6.3), never reported as rollback protection |

Also measure build/startup time and usable capacity with authenticated storage at representative vault sizes, and verify keys/plaintext scratch cannot reach persistent build files, swap, core dumps or any clear location other than explicitly public sidecar content.

---

## 13. Decisions and open questions

### Decisions (recorded)

- **D1. Per-platform copies are re-sealed** (§6.2, 2026-09-11). Each platform copy gets its own LUKS header, UUID, secret, binding and KBS resource; a single image with two platform rules was rejected. Copying an already sealed image within its authorized platform is allowed under D8 and retains its existing cryptographic identity.
- **D2. Dev mode uses an unencrypted vault** (§8.3, 2026-09-11). The key-on-`user_config` path is dropped; dev and production never share a `profile_version`, and a dev vault carries no real secrets.
- **D3. No retained vault master; later-platform copies are re-populated from pinned inputs** (owner decision, 2026-09-11). The builder does **not** keep an encrypted master per vault revision, so §7.2's "no plaintext vault image is written to the builder's disk" stands unamended and there is no builder-held master key to manage. A copy sealed for a platform added later re-populates from the same pinned inputs; byte-identity of application content (excluding per-copy identity manifests and filesystem metadata) is guaranteed only across copies sealed in a **single** Stage 2 run. The builder documents the content-stability guarantees it relies on (pinned input archives and package versions, reproducible timestamps) and records each copy's content digest in its manifest so drift between separately built copies is detectable.
- **D4. One kernel per profile version** (owner decision, 2026-09-11). Both platform bundles of a profile version use the same pinned kernel (≥ 6.7, configfs-tsm — §8.2); per-platform `kernel_modules` (`sev_guest` vs `tdx_guest`, `tsm`) come from that same kernel build. Kernel version joins the vault-facing contract (§4.1, §7.1 step 7, §9.1), so dm-crypt/dm-integrity validation (§6.4, §12.1) runs once per profile version, not once per platform kernel.
- **D5. GPU capability is uniform across a profile version** (owner decision, 2026-09-11). A `gpu: nvidia_cc` profile is GPU-capable on every platform in its set; the pinned driver and container toolkit must support CC mode on both SNP and TDX hosts — select a driver version that does, rather than forking the profile. GPU-on-one-platform profile versions are not allowed. The application declaration remains the `requires_gpu` field in the generic build configuration (§9.2); its capability-matching rule is unchanged.
- **D6. Use a restricted Trustee KBS instance** (owner decision, 2026-09-11; provisioning revised by D9). Start default-deny with no key resources. Approved generic bundles supply references and reusable rules (§7.5). Import no unreviewed rules or resources. Bundle-level policy is shared across independently addressed vault keys.
- **D7. Provision fresh vault identities** (owner decision, 2026-09-11). Each deployment uses newly sealed vaults and newly provisioned key resources. Vaults are not relabeled to claim a different storage or security contract, and KBS policy is installed from approved bundle references.

- **D8. Vault images may be copied; each runtime file is exclusive to one CVM** (owner clarification, 2026-09-11). A stopped, detached vault may be copied for deployment, transfer or backup. Each runtime file copy has its own writable state and sidecars and is used by at most one CVM at a time. Concurrent attachment of one file is prohibited, including during restart or reassignment. Launchers require exclusive image locking; a new launch can proceed only after the previous QEMU process exits and releases the image. Byte-for-byte copies retain the same UUID, key, binding and platform authorization; independent cryptographic identities require re-sealing. Docker loading needs ordinary interrupted-load recovery, with no coordination between CVMs sharing a vault (§4.1, §7.3, §12.1).

- **D9. Reuse bundle policy; address vault keys by attested binding** (owner-approved approach, 2026-09-11). Adopt the Kata/CoCo separation of reference values, secret resources and policy. Stage 1 provisions generic measurements in RVPS and reusable resource rules once per bundle. Stage 2 only uploads each key at `keys/<cvm_build_id>/<vault-binding-id>`; KBS dynamically compares that path with signed evidence and the guest checks the frozen attached header. No per-vault measurement file, authorization inventory or policy publisher is required. Trusted key creation activates access; key deletion/disablement revokes future retrieval for that vault identity. This supersedes the per-vault publisher/inventory portions of earlier review passes, while preserving per-platform sealing (D1), exclusive attachment (D8), CPU appraisal and authenticated storage.

- **D10. General-purpose container runtime; application provisioning stays external** (owner clarification, 2026-09-11). NVFlare is the main use case, but any compatible Docker application can supply an image, generic launch configuration and optional payload. The builder defines `vault_build.yml` and the authenticated runtime configuration; it does not require framework packages, startup kits or participant roles. NVFlare provisioning owns its startup-kit generation, `cc_params.yml` and translation into generic inputs. This supersedes earlier NVFlare-specific population tasks and provisioning-schema changes in this design (§§1, 4.2, 7.2, 9, 10).

- **D11. Optional Stage 1 platform override** (owner clarification, 2026-09-11). `cvmctl build` auto-detects the build host's supported TEE when `-p` is omitted. Explicit `-p` takes precedence for CI and cross-platform construction. Unknown or ambiguous auto-detection fails with a diagnostic instead of choosing a fallback. Resolved target identity, target-platform validation, runtime host checks and Stage 2's independent platform list remain explicit (§§7.1, 9.1, 12.1).

- **D12. Security findings become explicit acceptance gates** (2026-09-14; sidecar role clarified by D17). Close the remaining R1/R3/R4/R5/R8/R10 gaps with private-key rejection for clear `user_config` and `user_data` input trees, explicit SSH service/socket masking, a shared 60-second CPU appraisal budget, a synchronized-clock gate, qualified upstream SNP collateral availability, forced fail-closed poweroff and direct root/GPU/network-fault acceptance cases. R12 remains covered by exact binding-addressed resource policy. Application admission/escape issues listed against NVFlare core stay with that component and do not expand the generic builder schema.

- **D13. Stage 1 uses a purpose-built construction provisioner** (owner decision, 2026-09-14). The generic root has one supported Ubuntu release and a small, fixed installation contract, so the builder sends its source snapshot, exact package pins and reviewed public inputs to the disposable plain VM and invokes `cvm/build/provisioning.py`. The provisioner uses only the Python standard library present in the base image, suppresses daemon starts during package installation, exports and hashes the pinned kernel/initramfs before removing build access, then masks login paths, deletes construction state and powers off. The provisioner source is part of `runtime_source_sha256`; no separate configuration-management runtime is installed or required.

- **D14. Simple target-host construction and explicit exact-manifest acceptance** (owner decision, 2026-09-14; acceptance flow revised 2026-09-22). The normal Stage 1 call performs construction, local measurement and finalization. Candidate administration and vault construction then exercise that exact manifest before report aggregation and signed approval; an optional trusted runner may automate those explicit stages. The checked-in schemas have complete Ubuntu 26.04 sample values and code defaults. Each platform/application OCI artifact materializes as one self-contained folder with the complete copied generic bundle under `cvm_bundle/`; delivery launch and shutdown require no arguments and no separate bundle download. Launch verifies the embedded bundle, detects and temporarily VFIO-binds the profile's exact GPU count when required, and retains repeatable explicit GPU overrides for advanced placement (§§4.1, 7.1, 7.2, 9.1).

- **D15. Native CoCo key administration** (updated 2026-09-18). Use the existing Trustee resource API with project-level `trustee` credentials. CVM Builder has no key-adapter or reconciliation service. Scoped resource tokens permit native POST and DELETE, while policy and reference administration remain separate. Native overwrite and deletion semantics apply; operator-controlled fencing and backup recovery replace the former adapter's create-only and permanent-tombstone guarantees (§7.5).

- **D16. OCI is the CVM and vault delivery format** (owner decision, 2026-09-15). Stage 1 emits one `application/vnd.nvidia.cvm.bundle.v1` OCI artifact per platform. Stage 2 emits one `application/vnd.nvidia.cvm.delivery.v1` artifact per platform with separate reusable CVM and vault/runtime layers. The identical OCI image layout can be carried offline as `.oci.tar` or copied to a registry. Runtime materialization verifies all content-addressed descriptors; production distribution separately authenticates the selected digest by signature or a trusted release channel. Registry consumers use digest references rather than mutable tags (§§4.1, 7.1, 7.2).

- **D17. Preserve directional clear sidecars and support measured multi-GPU cardinality** (owner clarification, 2026-09-15). Only `vault.qcow2` uses authenticated LUKS2 and the KBS secret. `/applog` is a clear writable output disk for keyless operator log access. `user_config` and `user_data` are clear operator-supplied input disks enforced read-only by QEMU, the guest mount and the container. Clear sidecars carry no secrets and remain outside the confidential boundary. GPU profiles record `gpu_count`; the launcher passes exactly that many complete isolated PCI slots and the guest rechecks the visible count before appraising all GPUs. TDX profiles pin RTMR0 because their launch shape is fixed.

- **D18. Approval is a signature, not a file** (review fix, 2026-09-20). `approval.json` carries an Ed25519 signature by an acceptance authority over its canonical body. Stage 2 trusts the public keys in `cvm_project.yml`'s `approval.public_keys`; `admin install` trusts `approval_public_keys` in its configuration. An unsigned, tampered or foreign-signed receipt selects nothing. `cvmctl pull` authenticates the publisher of an OCI artifact through `--archive-sha256` or `--cosign-key` unless the operator passes `--allow-unverified`.

- **D19. Quarantine before power-off on periodic failure** (review fix, 2026-09-20). A failed periodic appraisal stops the workload and closes the vault, which removes the key from the kernel, and retries authorization for a bounded window before PID 1 powers off. No secret is reachable while quarantined, so the fail-closed property is preserved while a transient network or backend fault no longer destroys in-memory training state. Initial bootstrap, integrity and supervisor failures still power off immediately.

- **D20. The measured guest kernel is locked down** (review fix, 2026-09-20). The command line adds `lockdown=integrity`, `module.sig_enforce=1`, `loglevel=3` and `printk.console_no_auto_verbose=1`; the provisioner installs sysctls that disable kexec, SysRq, unprivileged BPF and kernel-address disclosure. Root in the guest cannot replace the measured kernel while keeping the attested identity, and register or stack dumps no longer reach the host-visible console. The activated vault mapping must reference a kernel logon key; an inline volume key is rejected. The container runs with `--cap-drop ALL` plus an admitted allowlist, `no-new-privileges` and a PID limit, `/host/bin` is opt-in, and admitted `app_*.service` units receive systemd sandboxing.

- **D21. Deterministic vault KDF and approved SNP launch policy** (review fix, 2026-09-20). The keyslot KDF is pinned to PBKDF2-SHA256 with a fixed iteration count, since the 512-bit random secret needs no stretching and a benchmarked argon2 cost would make guest unlock time and memory depend on the build host. The SNP guest policy word is derived from the approved references at Stage 1 and recorded in the launch shape rather than hard-coded.

### Open questions

1. **Supported release matrix.** Select and validate exact QEMU/firmware/kernel/guest-components/cryptsetup versions for each production profile. Instance binding and authenticated storage are mandatory; incompatible combinations fail instead of downgrading.
2. **Production Trustee revision and backend.** Use CoCo v0.23.0’s unmodified Trustee v0.22.0. Validate the configuration, resource/ref APIs, default CPU/GPU policies and local_fs adapter against that exact release before rollout and on every upgrade.
3. **TDX kernel interface qualification.** The Ubuntu 26.04 candidate uses `/dev/tdx_guest` report ioctls and the pinned Trustee TDX attester with QGS and CCEL. This combination is exercised by the hardware tests in [VALIDATION.md](VALIDATION.md). Qualify the exact interface again for a different production kernel/attester combination; kernel version alone is insufficient to establish quote support.
4. **Vault maintenance.** The builder treats header/keyslot/layout changes as a new vault revision with a new UUID/secret/binding and key resource. An in-place resize/re-encryption workflow is deferred until it preserves authenticated storage and defines the key-resource transition.
5. **Multi-vault per CVM.** Deferred. A future format can hash a canonical list of vault identities into the fixed-width binding field; it needs corresponding guest and resource-policy checks, not merely additional attached disks.
6. **GPU-dependent resource release — implemented; complete hardware acceptance pending.** CoCo’s upstream client collects composite evidence; Trustee performs NVIDIA verification and emits per-device GPU submods. CVM Rego requires exact GPU count, distinct NVIDIA identities, matching default policy, favorable status and integer trust vectors. Driver/VBIOS approvals and expiry remain in RVPS. Both policy digests are recorded. The September 21 positive SNP/H800 workload and periodic-appraisal run at `95bf53889` is recorded in [VALIDATION.md](VALIDATION.md); it does not qualify later source revisions or establish the required physical GPU-denial cases. See [TRUSTEE_GUIDE.md](TRUSTEE_GUIDE.md) for upstream NRAS configuration and egress.


---

## 14. References

- Current implementation: [cvmctl](cvmctl), [CLI dispatcher](cvm/__main__.py), [cvm/build/cvm.py](cvm/build/cvm.py), [cvm/artifacts/oci.py](cvm/artifacts/oci.py), [cvm/build/provisioning.py](cvm/build/provisioning.py), [cvm/build/vault.py](cvm/build/vault.py), [cvm/runtime/bootstrap.py](cvm/runtime/bootstrap.py), [cvm/common/luks.py](cvm/common/luks.py), [cvm/host/launcher.py](cvm/host/launcher.py), [cvm/trustee/admin.py](cvm/trustee/admin.py), [cvm/trustee/client.py](cvm/trustee/client.py); conformance assessment: [CONFORMANCE.md](CONFORMANCE.md)
- CoCo, "Building Trust into OS images for Confidential Containers" (dm-verity root hash on cmdline; `sev-snp-measure`; vTPM PCR reference values) — https://confidentialcontainers.org/blog/2024/03/01/building-trust-into-os-images-for-confidential-containers/
- CoCo Trustee state and policy model: [Reference values](https://confidentialcontainers.org/docs/attestation/reference-values/), [Resources](https://confidentialcontainers.org/docs/attestation/resources/), [Policies](https://confidentialcontainers.org/docs/attestation/policies/) — architecture basis for D9; the exact path/claim API is versioned separately in §6.2.
- CoCo guest-components attestation-agent attesters (`snp`, `tdx`, `tpm`, `az-tdx-vtpm`; TDX evidence = quote + CCEL; RTMR extension mapping) — https://github.com/confidential-containers/guest-components/tree/main/attestation-agent
- Trustee KBS initdata (SNP `hostdata`, TDX `mr_config_id`, vTPM PCR 8) — https://github.com/confidential-containers/trustee/blob/main/kbs/docs/initdata.md
- Trustee AMD certificate-cache design — https://github.com/confidential-containers/trustee/blob/main/attestation-service/docs/amd-offline-certificate-cache.md
- Trustee TDX verifier claims and CCEL replay — https://github.com/confidential-containers/trustee (deps/verifier/src/tdx, deps/eventlog)
- chrony `waitsync` command — https://chrony-project.org/doc/4.5/chronyc.html
- CoCo measured boot / rootfs RFCs — https://github.com/confidential-containers/documentation/issues/40 , https://github.com/confidential-containers/confidential-containers/issues/116
- Kata Containers guest assets and confidential rootfs — https://github.com/kata-containers/kata-containers/blob/main/docs/design/architecture/guest-assets.md
- Intel TDX measured boot: MRTD/RTMR ↔ PCR mapping, CCEL, RTMR precompute — https://mahaocheng.me/blog/2025/tdx-measure-boot/
- Linux TDX guest documentation — https://docs.kernel.org/arch/x86/tdx.html
- NVIDIA Deployment Guide for SecureAI (SNP) — https://docs.nvidia.com/cc-deployment-guide-snp.pdf
- NVIDIA Ubuntu driver installation — https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html
- NVIDIA Container Toolkit installation — https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- NVIDIA C++ GPU attestation SDK and CLI — https://github.com/NVIDIA/attestation-sdk and https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/sdk-cli/command-reference.html
- NVIDIA GPU claims — https://docs.nvidia.com/attestation/advanced-documentation/latest/claims-guide/gpu_claims.html
- Ubuntu 26.04 cloud image and checksums — https://cloud-images.ubuntu.com/releases/26.04/release/
- Ubuntu 26.04 OVMF packages — https://packages.ubuntu.com/resolute/ovmf
- TianoCore edk2 stable release used by the recorded validation — https://github.com/tianocore/edk2/releases/tag/edk2-stable202605
- OCI Image Manifest artifact guidance — https://github.com/opencontainers/image-spec/blob/main/manifest.md#guidelines-for-artifact-usage
- OCI Distribution Specification — https://github.com/opencontainers/distribution-spec/blob/main/spec.md
- ORAS copy between registries and OCI layouts — https://oras.land/docs/commands/oras_cp/
- Cosign signatures for arbitrary OCI artifacts — https://docs.sigstore.dev/cosign/signing/other_types/

---

## 15. Revision history

- **2026-09-09 — initial revised draft.** Two-stage build, HOSTDATA/MRCONFIGID binding, authenticated LUKS2, Trustee contract pinned to `a2570329`.
- **2026-09-11 — review pass 2.** Frozen-header keyslot contract; G4 key-isolation argument; deployability gate (§6.4); dev-mode resolution (§8.3); policy publisher (§7.5); integrity→workload propagation made concrete (§10); first-boot load single-flight (§7.3); builder/guest digest equality; branch reference corrected; new §12.1 rows.
- **2026-09-11 — review pass 3.** Workload-target start must be `systemctl start --no-block` (activation-deadlock fix, §5/§10) with a matching §12.1 row; G4 argument rewritten around host-set instance field vs attached header, with the same-bundle discriminator made explicit; dev-mode denial wording corrected for the no-TEE case (§8.3); build/populate/seal terminology sweep; `requires_gpu` moved to the new-fields list (§9.2); publisher read-back byte-equality requirement (§7.5); §13 split into recorded decisions and open questions.
- **2026-09-11 — review pass 6.** Split the `--no-block` rule: the workload target (requires `cvm_vault.service`) still uses `--no-block`, but `cvm_integrity.service` is started synchronously with a readiness signal so the monitor is active before the authenticated scan (§5, §6.4, new §12.1 row). Reconciled the policy-install actor model: builders submit copies + secrets to the publisher, which alone calls `set-resource-policy` under its lock (§7.2 step 6–7, §7.5). First-boot Docker load marker gained liveness (boot-id + heartbeat) with a stale-marker reclamation path and a §12.1 row (§7.3). Renamed the profile-level `allowed_out_ports` to `bootstrap_egress` to avoid the clash with the site field (§9.1, §9.2). Provisioned `/etc/cvm_profile_version` (§7.1). Corrected the trust-vector Rego comment (executables affirms at 3, hardware/config at 2). D1 cross-referenced in §6.2; D2 dated; "migration-agent permitted" and "policy transition" wording; byte-identity scoped to one run (§1); build/populate terminology in G1a/G5; `docker.service` ordering filled in; script rename noted (§4.2).

- **2026-09-11 — owner clarification D8.** Explicitly allow vault file copies while requiring one CVM per runtime image file. Added exclusive-attachment and restart rules to the launcher, lifecycle and acceptance checks; clarified file copying versus per-platform re-sealing (D1). Removed the concurrent-boot Docker load protocol and its marker/heartbeat tests introduced in passes 2 and 6; retained sequential interrupted-load recovery.

- **2026-09-11 — owner-approved approach D9.** Replaced the per-vault registration inventory and policy publisher with RVPS references and reusable rules per generic CVM bundle, plus binding-addressed key resources. Updated boot-time path derivation, Rego, build stages, resource activation/retry/revocation, rollout and acceptance checks. Corrected policy read-back decoding and recorded production AS selection/admin-auth requirements for the cited Trustee baseline. Clarified that content equality excludes per-copy identity manifests. Earlier publisher/inventory requirements are historical and superseded.

- **2026-09-11 — application scope and platform selection (D10–D11).** Made the Docker workload and vault inputs application-neutral, with NVFlare as the main use case. Moved startup kits, `cc_params.yml` and application-specific provisioning outside the builder contract. Defined generic configuration/payload ingestion and preserved the image's own entrypoint by default. Made `-p` optional with capability-based auto-detection, an explicit override and failure behavior. Updated the architecture, boot flow, runtime contract and acceptance checks.

- **2026-09-11/12 — Implementation and CPU validation.** Implemented the generic/vault pipeline, authenticated storage, exclusive launcher, measured runtime and restricted Trustee provisioning boundary. Selected Ubuntu 26.04 for the guest; documented concrete artifact names and tested kernel interfaces. Added piped-core-collector protection for secret-handling hosts and the guest. Exact test candidates, hardware evidence and remaining production gates are recorded in [VALIDATION.md](VALIDATION.md).
- **2026-09-14 — Security findings closure pass (D12).** Added streaming private-key rejection for clear input sidecars, made SSH socket shutdown explicit, required chrony synchronization before appraisal, bounded CPU/GPU appraisal and fail-closed shutdown, backported a verified SNP VCEK cache into the pinned Trustee patch, and added production evidence gates for sidecar roles, root-sector tampering, SSH, KDS/NRAS DROP behavior, clock state and GPU poweroff.
- **2026-09-14 — root overlay capacity.** Added the optional `root_overlay_max_mib` generic-profile setting, defaulted it to half of guest RAM, bounded it by configured guest memory, placed the resolved value on the measured kernel command line and required production evidence that `/cow` has the configured capacity.
- **2026-09-14 — construction provisioning simplification (D13).** Replaced the Stage 1 configuration-management playbook and inventory with a standard-library Python provisioner transferred over the existing temporary SSH channel. Split installation/artifact export from final hardening so the host verifies kernel/initramfs hashes before the guest removes build access and powers off. Removed the old dependency and directory; added exact payload validation and provisioner contract tests.
- **2026-09-14 — build and runtime workflow simplification (D14).** Added a one-call target-host path with an exact-manifest acceptance runner, complete Ubuntu 26.04 configuration defaults, shell wrappers for finalization and KBS administration, pip/uv setup examples, standard-layout bundle discovery, automatic NVIDIA GPU VFIO preparation, and a zero-argument shutdown wrapper backed by exact PID/start-time state.
- **2026-09-14 — user workflow follow-up.** Made `site_acceptance` the automatic simple-build command, documented official input download locations and the exact delivery archives, renamed the runtime guide to `USER_GUIDE.md`, and documented the complete detected-GPU path from host VFIO binding through QEMU to Docker `--gpus all`.
- **2026-09-22 — executable acceptance flow and failed-profile gate.** Replaced the implicit runner prerequisite with exact-manifest candidate administration/vault commands and a manifest-bound report aggregator. The checked-in failed kernel/storage candidate now carries `production_ready: false`, which mechanically prevents production approval. Candidate mode accepts the exact production-named manifest under test; production vaults still require its signed approval.
- **2026-09-14 — self-contained vault delivery.** Moved the copied, verified generic platform bundle under `cvm_bundle/` in every platform delivery and archive. A recipient needs only the delivered tarball; zero-argument launch finds the embedded bundle without another download.
- **2026-09-15 — OCI delivery format.** Replaced Stage 1 CVM and Stage 2 vault tarball deliverables with single-manifest OCI image-layout tar files; split the final delivery into reusable CVM and vault/runtime layers; added digest records, registry publication, immutable registry retrieval, offline materialization and publisher-authentication requirements (D16).

- **2026-09-15 — RA conformance closure.** Added exact multi-GPU passthrough/cardinality enforcement and pinned TDX RTMR0. Kept the generic root and directional sidecars clear by owner requirement: `/applog` is writable public output, while `user_config` and `user_data` are read-only public inputs; only the vault is authenticated-encrypted (D17).

- **2026-09-15 — reference-architecture conformance assessment and profile-scoped GPU key policy.** Recorded the NVIDIA self-hosted GPU-inference reference-architecture assessment in [CONFORMANCE.md](CONFORMANCE.md), including remaining implementation, deployment and acceptance gaps. Clarified that CPU-only NVFlare and other `gpu: none` deployments require no GPU evidence. GPU appraisal remains a workload gate in the current implementation; only a key protecting a confidential-GPU workload needs an additional GPU-dependent release policy before that deployment can claim reference-architecture conformance.
- **2026-09-15 — current NVIDIA GPU appraisal path and validation updates.** Replaced the deprecated Python attestation dependency with a pinned NVIDIA C++ `nvattest`/`libnvat` payload, nested v3 claim validation, signed-EAT and fresh-nonce checks, and support for a headless precompiled driver-module package set. Kept cryptographic RIM checks mandatory while handling two schema fields omitted by current NRAS responses. Added a 256 GiB prefetchable MMIO reserve per passed-through GPU for 128 GiB data-center BARs and a bounded 180-second appraisal window for NRAS/RIM/OCSP processing. Added NTS key-exchange egress, removed build-time clock cookies, and bounded cold-clock initialization without bypassing the final correction/skew gate. Added [TRUSTEE_GUIDE.md](TRUSTEE_GUIDE.md) with backend installation and key lifecycle instructions. Timings, registry round trips and hardware test results are recorded in [VALIDATION.md](VALIDATION.md).
- **2026-09-16 — simplified profile-set metadata.** Removed the duplicate `contract.profile_version` and per-bundle `path`. The profile version remains at the top level of the profile set and each CVM manifest; bundle directories are fixed to the platform names. Build, finalization, vault loading and OCI merging require matching profile versions and contracts.
- **2026-09-16 — simplified vault inputs.** Replaced `cvm_profile` with `cvm_image` accepting a local generic CVM folder or digest-pinned registry reference. Vault Build retrieves and verifies registry artifacts before use, generates its deployment ID, and defaults optional `platforms` to every available bundle. Generated IDs remain in output metadata and archive names; callers can select a stable output location with `--output`.
- **2026-09-16 — project-wide key service.** Moved shared `key_service` settings into `cvm_project.yml`, discovered above the build YAML or selected with `--project-config`. Credential paths are relative to the project file. Per-build key-service settings are rejected; plaintext development builds do not load project credentials. Superseded by native Trustee configuration on 2026-09-18.

- **2026-09-17 — PR security boundary review and composite GPU authorization.** Added backend CPU/GPU key gating, strict NRAS signature/digest/nonce checks, exact GPU EAR cardinality and policy checks, immutable GPU policy receipts, and RVPS driver/VBIOS approvals. Isolated AS signing trust from transport identities, prevented cross-profile TCB unions and implicit expiry renewal, excluded private hardware reports and plaintext content hashes from delivery, authenticated NFS configuration, restricted writable application mounts and service executable paths, bounded key-service TLS work, and documented fresh TDX deployment prerequisites. GPU hardware conformance remains Evidence pending.

- **2026-09-17 — align with CoCo v0.23.0.** Removed the Trustee patch and custom Rust NVIDIA verifier/attester. Use unmodified Trustee v0.22.0, upstream NVAT collection, Rego v1 and RVPS query APIs, role ACLs and local_fs namespaces. Freshness and reference expiry are enforced in CVM Rego; historical patched-backend test results do not approve the migrated profile.

- **2026-09-18 — existing CoCo deployment only.** Removed the three Trustee systemd units and the custom Python key server. Vault builds now use native resource POST with a pre-issued bearer token; revoke uses native DELETE. Retained offline bundle approval/retirement and reference tools. Updated the lifecycle contract to native overwrite/delete semantics and operator-managed backup recovery.

- **2026-09-18 — three-unit guest supervision.** Consolidated bootstrap, reference emission, clock/NFS gates and periodic checks; retained the independent notify/watchdog integrity monitor and application service. Measured nftables rules load through the distro unit; Docker socket activation is disabled. PID 1 handles security poweroff directly. New image measurements and hardware acceptance remain required.

- **2026-09-20 — architecture and security review fixes (D18–D21).** Signed approval receipts and publisher-authenticated pulls; quarantine with vault close and bounded retry on periodic failure; kernel lockdown, signature enforcement, console loglevel and hardening sysctls on the measured root; keyring-only volume keys and a deterministic keyslot KDF; SNP launch policy from approved references; QMP-based orderly shutdown, `-nodefaults`, no terminal monitor, detached QEMU session and a forwarding bind address; container capability, privilege and PID confinement with opt-in `/host/bin`, systemd sandboxing for admitted services, CIDR allowlists and resolver-scoped DNS; NTS-only `time_servers` and a measured `vault_prescan` switch; construction-account, machine-id, host-key and cache scrubbing at finalize; `kbs-client` provenance; 30-day token lifetime and bundle-scoped resource roles; root-only lock directory; `--diagnostics`; git-ignored `inputs/`; package entry point `nvflare-cvmctl` and ownership-checked delivery wrappers. Every measured root must be rebuilt, remeasured and reapproved; no hardware validation of these changes has been performed.

- **2026-09-21 — PR review follow-up.** Added nonblocking audit queuing and PID 1 phase deadlines, moved NFS to the guest-owned `/nfs_data` mountpoint, isolated container variables from the privileged Docker CLI, reformatted public output without a journal at boot, fixed filesystem types and empty-resolver DNS denial, tightened container defaults, bounded anti-forensic splitter parameters, pinned NTS time sources and disabled GPT automount. Build-host Python dependencies are hash-locked. Documented Trustee tenant isolation, workload/data-owner trust asymmetry and the RCAR delegation. See [VALIDATION.md](VALIDATION.md#pr-security-review-follow-up--2026-09-21-utc) for Linux and live upstream Trustee evidence and the remaining hardware acceptance requirement.
