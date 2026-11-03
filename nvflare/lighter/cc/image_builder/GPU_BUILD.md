# Reproduce GPU build inputs

Run these steps in a disposable **Ubuntu 26.04 x86-64 build environment**.
The construction base image remains the authenticated, unmodified Ubuntu cloud
image from [BUILD_GUIDE.md](BUILD_GUIDE.md). No NVIDIA repository or library needs
to be preinstalled in that image. Keep the resulting public inputs and build
record with the reviewed profile.

## Authenticated package sources

The Ubuntu repositories supply the guest kernel and NVIDIA driver packages.
NVIDIA Container Toolkit requires its own repository, described in
[NVIDIA's installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Fetch the public signing key and verify the reviewed keyring bytes before use:

```sh
(
  set -eu
  mkdir -p inputs
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey -o inputs/nvidia_container.asc
  gpg --batch --yes --dearmor -o inputs/nvidia_container.gpg inputs/nvidia_container.asc
  printf '%s\n' '425822bb25bfa7f5ce96e598a7bbd27db128649e4113017b3ff765b98b43b166  inputs/nvidia_container.gpg' | sha256sum -c -
)
```

The primary key fingerprint is
`C95B321B61E88C1809C4F759DDCAE044F796ECB0`. A rotated key requires a reviewed
profile update; do not replace the digest automatically when verification fails.
Set these fields in a copy of `config/cvm_profile.yml`. Update the existing
`platforms` entries shown below; do not add a second `platforms` mapping:

```yaml
gpu: nvidia_cc
gpu_count: 1
gpu_policy: gpu_policy.json
gpu_attestation_url: https://nras.attestation.nvidia.com/v4/attest/gpu
gpu_attestation_library: ../inputs/libnvat.so.1
gpu_attestation_provenance: ../inputs/nvat_build.json
platforms:
  amd_sev_snp:
    kbs_client: ../inputs/kbs-client-gpu
    kbs_client_provenance: ../inputs/kbs_client_gpu_build.json
  intel_tdx:
    kbs_client: ../inputs/kbs-client-gpu
    kbs_client_provenance: ../inputs/kbs_client_gpu_build.json
gpu_apt_repositories:
  - url: https://nvidia.github.io/libnvidia-container/stable/deb/amd64
    suite: /
    components: []
    keyring: ../inputs/nvidia_container.gpg
    keyring_sha256: 425822bb25bfa7f5ce96e598a7bbd27db128649e4113017b3ff765b98b43b166
gpu_packages:
  - linux-modules-nvidia-580-open-7.0.0-31-generic=7.0.0-31.31+1
  - nvidia-kernel-common-580=580.178.04-0ubuntu0.26.04.1
  - libnvidia-compute-580=580.178.04-0ubuntu0.26.04.1
  - nvidia-compute-utils-580=580.178.04-0ubuntu0.26.04.1
  - nvidia-utils-580=580.178.04-0ubuntu0.26.04.1
  - libnvidia-cfg1-580=580.178.04-0ubuntu0.26.04.1
  - libnvidia-container1=1.20.0-1
  - libnvidia-container-tools=1.20.0-1
  - nvidia-container-toolkit-base=1.20.0-1
  - nvidia-container-toolkit=1.20.0-1
  - libxml2-16=2.15.2+dfsg-0.1ubuntu0.3
```

Choose a new `profile_version` and approve the driver/VBIOS/TCB references for
the intended devices. These package pins form one concrete build example, not
approval for every GPU or TCB. Keep all four Container Toolkit package versions
aligned. Repository URLs, suites, components and keyring hashes enter the CVM
contract. Stage 1 copies the keyrings, checks their hashes again, and writes
deb822 sources with a repository-specific `Signed-By` **before** `apt-get update`.
Update failures stop construction. No `trusted=yes` or global `apt-key` trust is
used. For retained releases, an authenticated immutable apt mirror can replace
the URL without changing the input schema.

APT pins can disappear from rolling security repositories. The libxml2 example
above includes the [Ubuntu security update](https://ubuntu.com/security/notices/USN-8910-1);
the older `0.1ubuntu0.1` pin is no longer a reproducible default. Before using
this example, inspect `apt-cache policy libxml2-16` and the signed repository's
available versions in the **guest build environment**, then simulate the exact
install with `apt-get -s install 'libxml2-16=2.15.2+dfsg-0.1ubuntu0.3'`.
If the pin is unavailable, either use the authenticated immutable mirror that
retains it or review the vendor's security notice and select a currently
available fixed version. Record the signed repository metadata and package
digests, check the complete pinned dependency set, and use a **new profile
version** followed by full acceptance and approval. Do not silently substitute
the host's candidate version, remove version pins, or relabel an old approval.

## Review GPU driver and VBIOS references

The platform security administrator supplies `gpu_driver_versions` and
`gpu_vbios_versions` in the profile's `approved-tcb-references.json`, alongside
the independently approved CPU TCB fields. The values are **strings**, preserving
their exact spelling. Inventory the intended device inside a trusted guest with:

```sh
nvidia-smi --query-gpu=driver_version,vbios_version --format=csv,noheader
```

Inventory is a discovery aid, not attestation or approval. Obtain a fresh,
signature- and nonce-verified NVIDIA appraisal through the deployed Trustee.
For each GPU, the signed EAR's
`submods.gpuN["ear.veraison.annotated-evidence"].nvidia` contains
`x-nvidia-gpu-driver-version` and `x-nvidia-gpu-vbios-version`. Validate the EAR
signature, issuer, lifetime, protocol binding, and affirmative CPU/GPU appraisal
before using those values. Compare them with the vendor-supported CC hardware,
driver and firmware/RIM releases and the intended device's reviewed inventory.
The administrator then explicitly approves the matching string lists; never
populate them automatically from an untrusted host or a merely decoded JWT.

The generated GPU policy requires the verified claims to match those lists and
also enforces the report/RIM signature, certificate, measurement, nonce, secure
boot and debug-state checks. Keep approval evidence outside public deliveries.
Import the approved lists through the [Trustee reference workflow](TRUSTEE_GUIDE.md#7-install-references-and-policies).
Changing supported versions requires a new reviewed profile and acceptance.

## NVAT source and provenance

The reviewed [`kbs-client` selector patch](cvm/build/kbs_client_policy_selector.patch)
updates the client lockfile to the upstream guest-components selector API. That
lockfile selects NVIDIA attestation-sdk tag `2026.06.09`, commit
`9d12801cea8a198ea0f29640dfaf8a4017c841c5`. Its CMake library version is
**1.2.2** and its ABI soname is `libnvat.so.1`. This revision already contains
the `const xmlError *` fix required by Ubuntu's `libxml2.so.16`, so no local
NVAT source patch is applied. The profile requires a clean-source provenance
record for the exact runtime library linked by the selector-capable client.

From the CVM Builder directory, build in fresh directories:

```sh
(
  set -eu
  sudo apt-get update
  sudo apt-get install -y build-essential cmake pkg-config git curl ca-certificates \
    perl cargo rustc libclang-dev libxml2-dev libxmlsec1-dev zlib1g-dev
  mkdir -p inputs
  git clone --no-checkout https://github.com/NVIDIA/attestation-sdk.git inputs/nvat_source
  git -C inputs/nvat_source checkout --detach 9d12801cea8a198ea0f29640dfaf8a4017c841c5
  test -z "$(git -C inputs/nvat_source status --porcelain --untracked-files=all)"
  CMAKE_BUILD_PARALLEL_LEVEL=8 cmake -S inputs/nvat_source/nv-attestation-sdk-cpp \
    -B inputs/nvat_build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
  CMAKE_BUILD_PARALLEL_LEVEL=8 cmake --build inputs/nvat_build --target nvat -j8
  install -m 644 inputs/nvat_build/libnvat.so.1.2.2 inputs/libnvat.so.1
  python3 - <<'PY'
import hashlib
import json
import pathlib
import platform
import subprocess

root = pathlib.Path("inputs")
source = root / "nvat_source"
def git(*args):
    return subprocess.check_output(["git", "-C", str(source), *args])
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
revision = git("rev-parse", "HEAD").decode().strip()
assert revision == "9d12801cea8a198ea0f29640dfaf8a4017c841c5"
assert not git("status", "--porcelain", "--untracked-files=all")
assert not git("ls-files", "--others", "--exclude-standard")
release = platform.freedesktop_os_release()
assert release["ID"] == "ubuntu" and release["VERSION_ID"] == "26.04"
assert platform.machine() == "x86_64"
packages = subprocess.check_output(["dpkg-query", "-W"], text=True)
(root / "nvat_build_packages.txt").write_text(packages)
record = {
    "source_clean": True,
    "source_repository": "https://github.com/NVIDIA/attestation-sdk.git",
    "source_commit": revision,
    "library_sha256": digest(root / "libnvat.so.1"),
    "build_environment": "ubuntu-26.04-x86_64",
    "build_packages_sha256": digest(root / "nvat_build_packages.txt"),
    "cmake_cache_sha256": digest(root / "nvat_build/CMakeCache.txt"),
    "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
}
(root / "nvat_build.json").write_text(json.dumps(record, indent=2) + "\n")
PY
)
```

Retain the source, dependency downloads, build directory, package inventory and
JSON together. Toolchain/dependency differences can change the binary hash; this
recipe records the resulting artifact rather than promising bit-identical
outputs across environments. Stage 1 checks the record's clean source revision,
build environment and library hash, and includes the entire record in the
contract. This is an operator-reviewed build record, not a signed remote-build
attestation.

To build the upstream GPU-enabled `kbs-client` in this disposable environment,
install the matching header and library where its unmodified build script expects
them, then use the clean Trustee checkout from [TRUSTEE_GUIDE.md](TRUSTEE_GUIDE.md):

```sh
sudo install -m 644 inputs/nvat_build/include/nvat.h /usr/include/nvat.h
sudo install -m 644 inputs/libnvat.so.1 /usr/lib/x86_64-linux-gnu/libnvat.so.1
sudo ln -sfn libnvat.so.1 /usr/lib/x86_64-linux-gnu/libnvat.so
sudo ldconfig
NVAT_USE_SYSTEM_LIB=1 cargo build --locked --release --manifest-path /tmp/trustee/Cargo.toml \
  -p kbs-client --bin kbs-client --features tdx-attester,snp-attester,nvidia-attester
install -m 755 /tmp/trustee/target/release/kbs-client inputs/kbs-client-gpu
./cvmctl provenance /tmp/trustee inputs/kbs-client-gpu inputs/kbs_client_gpu_build.json
```

The guest receives the pinned library bytes and its runtime dependencies, not the
SDK source or development packages. Validate GPU evidence collection and encrypted
key retrieval with these exact artifacts before approving the generic CVM.
The builder verifies that a GPU profile's client links `libnvat.so.1`; the
separate filename prevents this binary from replacing the CPU-only client.
