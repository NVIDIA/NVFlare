# CVM Builder

> **Candidate profile only (checked 2026-10-28).** The checked-in
> `cpu-2026.09-r5` profile has `production_ready: false`. Its pinned
> kernel `7.0.0-31-generic` passed the previously failing
> `test_corruption_after_scan_stops_workload` case in the rc4 and rc5 runs. That
> removes the recorded kernel/storage blocker, but the complete current-source
> acceptance matrix and signed production approval have not been recorded for
> this profile. Production approval therefore still deliberately refuses it; a
> kernel version alone is not a qualified security profile. Use the
> [dev-mode walkthrough](#30-minute-dev-mode-walkthrough) for evaluation. A
> production deployment requires a new `profile_version` whose exact manifest
> passes the complete hardware acceptance matrix; do not relabel this profile.

All builder and administration commands use `./cvmctl <command>` (or
`python3 -m cvm` from this directory). Run `./cvmctl --help` for the command list
and `./cvmctl <command> --help` for options. The wrapper selects
`CVM_BUILDER_PYTHON`, the local `.venv/bin/python`, or `python3`, in that order.
The delivered `launch_cvm.sh` and `shutdown_cvm.sh` remain standalone host commands.

Build a reusable, application-free Ubuntu 26.04 confidential VM and a separate
authenticated vault for each application release and site. Docker workloads are
generic: the image's entrypoint and command are preserved by default. NVFlare is
the main use case; its startup kit and `cc_params.yml` belong to NVFlare
provisioning, which supplies ordinary application files to this builder.

The security contract and acceptance requirements are in [DESIGN.md](DESIGN.md).
Production delivery requires an approved generic bundle. `--candidate` creates
an acceptance-only vault from the exact unapproved manifest that can later be
approved; it never creates production approval by itself.
Both reusable CVM bundles and final CVM-plus-vault deliveries are emitted as OCI
image-layout tar files. They can be moved offline or published to a registry.
The normal workflow uses `build`, `vault`, and `run`; each command combines the
safe consecutive operations for its role. The lower-level `finalize`,
`acceptance-report`, `admin`, `publish`, and `pull` commands remain available for
separate trust roles, offline transfer, and recovery. Publisher authentication
uses `--archive-sha256` for a local artifact or `--cosign-verification-key` for
a registry artifact. `run` always requires one of those authenticated inputs;
the separate `pull` command permits an explicit lab-only `--allow-unverified`. Add
`--diagnostics DIR` before any command to retain stderr of failed commands that
never handle secrets.

This builder is part of NVFlare at `nvflare/lighter/cc/image_builder`.
Run the commands below from this directory in a source checkout. NVFlare
provisioning invokes `cvmctl vault` for participants whose unified `cc_config`
selects `cc_deployment_mode: bare_metal_cvm`; see the
[provisioning example](../../../../examples/advanced/cvm_builder/README.md) and
[common CC guide](../../../../docs/user_guide/confidential_computing/deployment.rst).

For operational instructions, see:

- [TRUSTEE_GUIDE.md](TRUSTEE_GUIDE.md): deploy standalone upstream Trustee or
  extend a CoCo deployment, enable approved CVM bundles, and manage vault keys.
- [BUILD_GUIDE.md](BUILD_GUIDE.md): build the reusable CVM bundle and a fresh
  vault for each application release and site.
- [USER_GUIDE.md](USER_GUIDE.md): verify, launch, operate, and stop a delivered CVM.
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md): diagnose build, Trustee, SNP, vault,
  and launch failures.
- [TDX_TROUBLESHOOTING.md](TDX_TROUBLESHOOTING.md): diagnose Intel quote-generation
  failures, recover QGS after PCK cache expiry, and select the Trustee TCB channel.

## Start here

Choose one path, then complete its checklist in order:

- **Evaluate locally without a TEE or Trustee:** install the dependencies, save
  an application image as `inputs/application.tar`, and run the
  [30-minute `--dev` walkthrough](#30-minute-dev-mode-walkthrough). Both `build`
  and `vault` must use `--dev`.
- **Build a production candidate:** deploy Trustee first using the
  [standalone or CoCo path](TRUSTEE_GUIDE.md#2-choose-a-deployment-path), fix its
  HTTPS CA, AS signing key, DNS name, and policy selectors, then copy those
  public trust inputs into `config/cvm_profile.yml`. Prepare and preflight the
  target TDX or SNP host, approve TCB references, build/finalize the generic
  CVM, and complete exact-manifest acceptance before approval.
- **Seal and run an application:** obtain an approved generic OCI artifact and a
  short-lived `cvm-resources-<build_id>` token, save the application with
  `docker save -o application.tar`, run `cvmctl vault`, then authenticate and
  launch the resulting delivery with `cvmctl run`.

Before production traffic, verify that the deployed Trustee identity is the one
pinned by the bundle, the current profile-specific RVPS record is unexpired,
policy selection and unauthorized administration tests pass, and every required
hardware acceptance result is present. A disposable `prepare_lab.py` deployment
is useful for protocol tests, but its generated PKI cannot approve a bundle that
pins another Trustee identity.

## Inputs and prerequisites

Run builds as root on an x86-64 Linux host with KVM, swap disabled, and sufficient
locked memory. Shared testing targets Ubuntu 26.04. The construction VM itself
runs without confidential-computing mode, so `-p` allows cross-platform builds.
Reference collection and runtime launch require the selected SNP or TDX hardware.
Disable piped core collectors on build and Trustee hosts before handling keys;
the tools check this explicitly. Linux ignores `RLIMIT_CORE` for pipe handlers.
See [core(5)](https://man7.org/linux/man-pages/man5/core.5.html). The measured guest
masks crash collectors and sets `kernel.core_pattern=/dev/null` with
`kernel.core_uses_pid=0`, in addition to process-level dump protection.

Install Python 3, PyYAML, cryptography, QEMU system/utilities, cloud-image-utils,
libguestfs-tools, cryptsetup-bin, e2fsprogs, tar, util-linux, and OpenSSH. Storage
requires kernel dm-crypt, dm-integrity and NBD support. TDX requires an operational
QGS and TDVF supporting direct measured kernel boot. SNP requires SNP-enabled
KVM and compatible OVMF. GPU profiles additionally need pinned driver/toolkit
packages from authenticated repositories, the upstream NVIDIA attester’s pinned
`libnvat` library and provenance, and an approved GPU claim policy.
See [GPU_BUILD.md](GPU_BUILD.md) for the clean-base package inputs and SDK recipe.

Before the first TDX build, complete the [host prerequisites](BUILD_GUIDE.md#1-prepare-the-build-host):
kernel/module enablement, explicit QGS vsock configuration, the selected
PCS/PCCS service's credentials and platform registration, and a signed-quote
smoke test. Then [discover and approve TCB references](BUILD_GUIDE.md#discover-and-approve-tdx-tcb-references).
An active QGS service or a local TDREPORT alone does not establish working remote attestation.
For SNP, the same section documents firmware enablement, KVM/device/IOMMU
preflight, KDS throttling, and optional installed-GPU screening. Production SNP
finalization requires attestation report version 3 or later so CPU-generation
specific TCB policy can be enforced; hosts that emit version 2 reports require a
firmware update before they can be qualified.

Install the builder requirements with pip or uv:

```sh
python3 -m pip install --require-hashes -r requirements.txt

# Or:
uv pip install --require-hashes -r requirements.txt
```

All input paths are relative to their YAML file. Production inputs are supplied
explicitly; the builder does not fetch firmware, private keys, or mutable
container tags. The conventional `inputs/` and `credentials/` directories hold
bearer tokens, CAs, signing keys and multi-gigabyte images and are ignored by
git. Production builds also require a `kbs_client_provenance` record for the
selected platform, produced by `./cvmctl provenance` beside the clean Trustee
checkout that built `kbs-client`. CPU and GPU profiles use distinct client and
provenance filenames, and the reusable CVM build verifies whether the binary links NVAT.
Use [config/cvm_profile.yml](config/cvm_profile.yml) and
[config/vault_build.yml](config/vault_build.yml) as configuration templates.
They provide concrete Ubuntu 26.04 defaults, conventional locations, and sample
values. A bare `sudo ./cvmctl build` reads that profile and auto-detects the one
local target platform; no platform argument is required.

## Optional site-automated flow

An authorized operator can automate the lifecycle without manually chaining
lower-level commands. Production normally separates build, acceptance-signing, Trustee
administration, and registry-publishing roles. Use the combined options only
when one operator is explicitly authorized to hold all of those credentials;
otherwise transfer the printed output to the role that runs the corresponding
lower-level command. `build` screens the host, constructs and finalizes the generic CVM,
runs the trusted acceptance adapter, signs its approval, enables
the approved bundle in Trustee, and publishes the signed OCI artifact. The
`site_acceptance_runner` name below is a site-supplied adapter, not a shipped
command. Called as `RUNNER BUNDLE REPORT`, it must install that exact candidate,
create and launch its candidate vault, run the complete checked-in acceptance
suite, and write the report only on success:

```sh
sudo ./cvmctl build config/cvm_profile.yml \
  --quote-probe /usr/local/sbin/site_tdx_quote_probe \
  --acceptance-runner /usr/local/sbin/site_acceptance_runner \
  --approval-key /secure/acceptance-signing.key \
  --install-config /srv/trustee/production-admin.json \
  --publish registry.example.org/cvm/profiles:cpu-2026.10-intel_tdx \
  --cosign-signing-key /secure/release-signing.key
```

The command stops at the first failed gate. It never enables an unapproved
production bundle or publishes an unsigned artifact. Omit `--install-config` or
`--publish` when those credentials belong to another role; the command prints
the output needed by that operator. `--quote-probe` is the TDX signed-quote
smoke test; omit it for an SNP build, whose host screening uses the SNP devices
and kernel capabilities directly.

The default, fully documented path is the split exact-manifest flow in
[BUILD_GUIDE.md](BUILD_GUIDE.md#21-exact-manifest-acceptance-and-approval). The
checked-in `acceptance_runner` is its lower-level evidence generator after the
candidate Trustee and candidate vault exist; passing it directly to `build`
cannot perform those prerequisite steps.

`vault` seals every requested platform delivery and publishes each signed
artifact. A multi-platform destination must contain `{artifact}`, which expands
to the generated artifact name without `.oci.tar`:

```sh
sudo ./cvmctl vault config/vault_build.yml \
  --publish 'registry.example.org/cvm/deliveries/{artifact}:1.0' \
  --cosign-signing-key /secure/release-signing.key
```

Give runtime operators the immutable digest references printed by that command.
They can authenticate, materialize, and launch a delivery in one operation:

```sh
sudo ./cvmctl run \
  registry.example.org/cvm/deliveries/vault_DEPLOYMENT_intel_tdx@sha256:DIGEST \
  --cosign-verification-key /etc/cvm/release-signing.pub \
  --output /srv/cvm/site-1
```

`run` remains attached to the launcher and returns when QEMU exits. Use `pull`
followed by the printed absolute launch command when transfer and launch must be
separate, or use the advanced commands below when construction, acceptance,
Trustee administration, and release signing are assigned to different people.
For a restart, invoke the existing materialized `launch_cvm.sh` by absolute path;
do not rematerialize the delivery, because a new output restores a pristine
vault and discards prior guest writes.

## Command details and advanced handoffs

The lower-level commands are the default operator path for sites that separate
construction, acceptance, Trustee administration, release signing, transfer,
and runtime operations. The automated options above compose those same gates.

### Build and approve the reusable CVM

Run `build` once for each platform and profile version. On a target TEE host, it
screens the host, constructs the generic root, boots it to collect reference
evidence, and finalizes the bundle. With the acceptance and release options from
the three-command example, the same invocation also runs acceptance, signs the
approval, enables the bundle in Trustee, and publishes it.

The output contains one workspace per platform and one
`cvm_<profile>_<platform>.oci.tar`. Application changes do not rebuild this
artifact. A changed kernel, firmware, measured runtime, attester, trust identity,
or policy requires a new profile version and fresh acceptance.

When another host or operator owns measurement, use the split flow:

```sh
sudo ./cvmctl build config/cvm_profile.yml \
  -p amd_sev_snp --defer-measurements

# On the authenticated SNP target after materializing the pending artifact:
sudo ./cvmctl finalize /srv/cvm/cvm_PROFILE/amd_sev_snp
```

When acceptance, Trustee administration, or signing belongs to another role,
omit those credentials from `build`. The command retains the completed
intermediate output. The next operator uses `acceptance-report`, `admin approve`,
`admin install`, or `publish` as described in
[BUILD_GUIDE.md](BUILD_GUIDE.md#21-exact-manifest-acceptance-and-approval).
Hardware reference collection alone never approves a bundle.

### Configure Trustee once per project

Use upstream **Trustee v0.22.0**, paired with **CoCo v0.23.0**. Share the CoCo
instance when one exists, or deploy the same upstream KBS with the checked-in
standalone configuration. [TRUSTEE_GUIDE.md](TRUSTEE_GUIDE.md) contains both
paths, including the production trust identity, policy mapping, reference
installation, deployment receipt, scoped resource token, and retirement flow.
CVM Builder contains no Trustee server fork or service unit.

A generic bundle installs references and a reusable resource rule once. Each
application vault uploads only its new binding-addressed key. There is no
per-vault measurement history. Native KBS uploads may replace a resource and
deletions have no permanent tombstone, so reconcile uncertain uploads and fence
writers before revocation or restore.

### Seal and publish an application delivery

Create an uncompressed Docker save archive, record the exact image ID, and set
`docker_archive`, `image_id`, and `cvm_image` in `config/vault_build.yml`:

```sh
image=my-application:release
docker pull --platform linux/amd64 "$image"
docker image inspect --format '{{.Id}}' "$image"
docker save --output inputs/application.tar "$image"
sudo ./cvmctl vault config/vault_build.yml \
  --publish 'registry.example.org/cvm/deliveries/{artifact}:1.0' \
  --cosign-signing-key /secure/release-signing.key
```

`cvm_image` is an approved local profile directory or an immutable generic-CVM
registry digest. Omit `platforms` to build every platform in that profile, or
select a subset. Each invocation creates a fresh deployment ID, encrypted vault,
key, binding, sidecars, and self-contained OCI delivery. It reuses the approved
generic CVM bytes and does not update AS policies or reference values.

NVFlare provisioning performs this `vault` operation for each participant whose
`cc_config` selects `bare_metal_cvm`; users configure the archive and generic
CVM in the unified provisioning files and run `nvflare provision` rather than
calling `vault` separately. See the
[bare-metal provisioning guide](../../../../docs/user_guide/confidential_computing/on_premises/cvm_builder.rst).

### Run or stage a delivery

Use `cvmctl run` for the normal runtime path. It authenticates the publisher,
materializes the delivery, and launches QEMU for the first time. Restart through
the existing materialized `launch_cvm.sh` so persistent guest writes survive.
Use `cvmctl pull` only when staging
and launch belong to different operations; it prints the absolute launcher
command. [USER_GUIDE.md](USER_GUIDE.md) covers archive checksums, registry
signatures, GPU selection, shutdown, logs, and exclusive vault attachment.

## Guest service supervision

The measured root ships three CVM units; their ordering, the periodic
re-attestation cadence and the quarantine behavior on a failed periodic check are
described in [USER_GUIDE.md](USER_GUIDE.md#guest-service-supervision). Rebuild
and reapprove generic CVM bundles after any change to the guest runtime; earlier
hardware acceptance records do not cover a changed boot or shutdown sequence.

## Python package layout

The standalone `cvm` namespace has no NVFlare imports. Its packages separate
construction, guest execution, host launch, Trustee administration and artifact
transport:

| Package | Responsibility |
|---|---|
| `cvm.build` | Generic CVM construction, application vault sealing, build inputs and construction provisioning |
| `cvm.runtime` | Guest bootstrap, attestation, application lifecycle, GPU readiness, integrity and audit |
| `cvm.host` | Host capability checks, QEMU launch/shutdown and GPU assignment |
| `cvm.trustee` | Bootstrap/client/admin tools for standalone or CoCo-managed upstream Trustee; no server fork |
| `cvm.artifacts` | Bundle/approval verification, OCI packaging and registry transport |
| `cvm.common` | Shared binding formats, measurement parsing, validation, pure policies and Linux primitives |

Guest images receive only `cvm.runtime` and the shared modules listed in
`cvm/build/payload.py`. Deliveries receive `cvm.host`, bundle verification and
their shared dependencies. Neither includes build or Trustee administration
code. The construction provisioner runs separately and is not installed in the
finished guest. Public shell commands and YAML fields are unchanged.

The package migration changes installed guest bytes and launcher templates.
Rebuild generic bundles with a new profile version, collect fresh measurements
and obtain new approval before creating deliveries with the reorganized tools.
Existing deliveries remain self-contained. The source fingerprint still covers
all construction sources, provisioning logic, payload lists and launch assets.

## Development and validation

### 30-minute dev-mode walkthrough

This path exercises reusable CVM construction, application sealing, and launch
on an x86-64 Linux host without TEE hardware or a running Trustee. It still
validates the production-shaped public build inputs in `cvm_profile.yml`: the
Ubuntu base image, build and guest firmware, reviewed `kbs-client-cpu` plus its
provenance matching `trustee_commit`, KBS CA, AS public key, attestation policy,
and syntactically complete TCB references for every enabled platform. Obtain or
build those inputs using sections 1 and 2 of the build guide before starting;
`--dev` skips live attestation and key release, not input validation. It produces a separate `dev-` profile and an
unencrypted development vault. With the base image, dependencies, and Docker
application image already available, the walkthrough is intended to fit a
30-minute evaluation window.

```sh
cd nvflare/lighter/cc/image_builder
mkdir -p inputs target
image=my-application:release
docker save "$image" -o inputs/application.tar
export CVM_DEV_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$image")"

python3 - <<'PY'
import os
from pathlib import Path
import yaml

profile = yaml.safe_load(Path("config/cvm_profile.yml").read_text())
profile["profile_version"] = "dev-local"
profile["attestation_policy"] = "../config/attestation_policy.rego"
Path("inputs/cvm_profile-dev.yml").write_text(yaml.safe_dump(profile, sort_keys=False))

vault = yaml.safe_load(Path("config/vault_build.yml").read_text())
vault["cvm_image"] = "../target/cvm_dev-local"
vault["docker_archive"] = "application.tar"
vault["image_id"] = os.environ["CVM_DEV_IMAGE_ID"]
Path("inputs/vault_build-dev.yml").write_text(yaml.safe_dump(vault, sort_keys=False))
PY

sudo --preserve-env=CVM_BUILDER_PYTHON ./cvmctl build \
  inputs/cvm_profile-dev.yml -p intel_tdx --dev
sudo --preserve-env=CVM_BUILDER_PYTHON ./cvmctl vault \
  inputs/vault_build-dev.yml --dev
```

Copy the `OCI artifact:` path and `Archive SHA-256:` value printed by `vault`,
then launch it through the same authenticated runtime command used in production:

```sh
sudo ./cvmctl run ARTIFACT_PATH --archive-sha256 PRINTED_ARCHIVE_SHA256 \
  --output /srv/cvm/dev-local
```

The `--dev` switches are mandatory on both build commands. Dev
artifacts cannot be approved, combined with a production bundle, or used as
evidence of TEE/KBS behavior.

Use a separate `dev-` profile plus `--dev` on both builders for an unencrypted
development vault and plain VM. Do not provide Trustee administration credentials to a dev
vault: `--dev` skips project discovery and rejects `--project-config`.
Dev artifacts cannot receive production approval or be mixed with production
deliveries. A test requiring real TEE/KBS behavior uses the exact finalized
bundle plus the explicit `--candidate` flag on `vault` and bundle administration.

Run the builder contracts on Linux with Python unittest. NVFlare's regular unit
suite also runs these contracts through
[`cvm_builder_test.py`](../../../../tests/unit_test/lighter/cvm_builder_test.py).
Use an isolated Linux test host and upstream Trustee for storage, HTTPS and hardware
acceptance; these tests are opt-in. Run these commands from the repository root:

```sh
export PYTHONPATH="$PWD/nvflare/lighter/cc/image_builder:$PWD/tests/unit_test/lighter/cc/image_builder${PYTHONPATH:+:$PYTHONPATH}"
cargo build --locked --release --manifest-path tests/unit_test/lighter/cc/image_builder/policy_engine/Cargo.toml
python3 -m unittest discover -s tests/unit_test/lighter/cc/image_builder -v
sudo env PYTHONPATH="$PYTHONPATH" CVM_STORAGE_TESTS=1 \
  python3 -m unittest discover -s tests/integration_test/lighter/cc/image_builder -p test_storage.py -v
```

To check release packaging on Linux or macOS, install the build dependencies in
`pyproject.toml`, then run:

```sh
CVM_DISTRIBUTION_TESTS=1 python3 -m unittest discover \
  -s tests/integration_test/lighter/cc/image_builder -p test_distribution.py -v
```

This builds an sdist and a wheel from that sdist in a temporary directory,
compares the packaged builder assets with the source, and runs the CLI and GPU
input validation against the wheel's contents, including the NVAT patch.

`tests/integration_test/lighter/cc/image_builder/prepare_lab.py` and
`tests/integration_test/lighter/cc/image_builder/lab_kbs.py` create an isolated
test deployment with disposable PKI and explicit loopback ports. For production
acceptance, `--identity-directory` and `--kbs-url` instead stage the eventual
production CA, AS signer, server identity, and measured hostname as described in
the Trustee guide. Its generated profile takes the
apt pins from the validated checked-in profile, because the reusable CVM build installs them
inside the guest base image and a build host whose mirror lags the guest
repository would force an apt downgrade. Use `--package-profile` to supply a
different pinned set, or `--pins-from-host` to derive them from the build host
and fail when the two sources disagree. Its HTTPS tests require
`CVM_HTTP_TESTS=1` and exercise actual Trustee encryption and policy evaluation
using signed fixtures. Such fixtures are protocol tests, not hardware acceptance.
Hardware acceptance must additionally exercise the exact sealed guest on each
platform, including corruption, interruption, binding, revocation and recovery.

Select the lab platforms when generating its profile. For a TDX-only lab, put
the validated TDVF in the lab's `inputs/OVMF.inteltdx.fd` (or set
`CVM_TDX_FIRMWARE`), then run from the repository root:

```sh
PYTHONPATH=nvflare/lighter/cc/image_builder python3 \
  tests/integration_test/lighter/cc/image_builder/prepare_lab.py \
  /path/to/isolated-lab --platform intel_tdx
```

Use `--platform amd_sev_snp` for an SNP-only profile, which does not require TDVF.
Repeat the option to select both; omitting it preserves the two-platform default.
The generated TCB reference file is deliberately empty. Populate reviewed,
approved references for the selected platforms before building. Selecting a
platform at build time does not exempt other enabled profile platforms from
reference validation. Also copy the Ubuntu cloud image to
`inputs/ubuntu-26.04-server-cloudimg-amd64.img`, install/build
`inputs/kbs-client-cpu`, and record its reviewed
`inputs/kbs_client_cpu_build.json`; `prepare_lab.py` now creates the policy files
under `config/` itself. Build the pinned Trustee checkout and binary under
`trustee-source`, then start the isolated service from the repository root:

```sh
sudo env PYTHONPATH="$PWD/nvflare/lighter/cc/image_builder" \
  python3 tests/integration_test/lighter/cc/image_builder/lab_kbs.py \
  /path/to/isolated-lab
```

The command writes `lab-kbs/kbs.json`, `admin.json`, `resources.json`, and the
protected KBS log. `--http-only` prepares PKI and policy files without guest
inputs; it is sufficient only for protocol fixtures, not hardware acceptance.

Recorded versions, test results, artifact hashes and remaining production gates
are in [VALIDATION.md](VALIDATION.md). The checked-in profile is a template, not
an already approved production configuration.
