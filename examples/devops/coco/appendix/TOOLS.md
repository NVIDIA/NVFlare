# Common tools and workflow inputs

This is part of the optional [beginner prerequisite appendix](../PREREQUISITES.md).
It helps beginners bring up the prerequisite systems for the NVFlare CoCo
workflow. Consult the official documentation for each component and the version
you deploy. If there is any discrepancy, the component's official documentation
takes precedence.

## Common tools — both hosts

Run this section after creating `inputs.env` in the main appendix. Load it in
each terminal. On fresh hosts, install the shared command-line tools and clone
the public NVFlare 2.9 source:

~~~bash
source "$HOME/coco-private/inputs.env"
sudo apt-get update
sudo apt-get install -y git curl ca-certificates openssl jq skopeo zstd \
  python3 python3-venv python3-pip python3-yaml iproute2
git clone --branch 2.9 --single-branch https://github.com/NVIDIA/NVFlare.git \
  "$HOME/NVFlare"
python3 -m venv "$HOME/coco-venv"
"$HOME/coco-venv/bin/python" -m pip install "$HOME/NVFlare"
source "$HOME/coco-venv/bin/activate"
cd "$COCO_ROOT"
~~~

Use the same source revision and NVFlare release on both machines and in later
application images. If the clone destination already exists, inspect it instead
of replacing it. For Python installation, see
[Python virtual environments](https://docs.python.org/3/library/venv.html);
for image-copy options, see the
[Skopeo documentation](https://github.com/containers/skopeo/tree/main/docs).

**Check — both hosts:**

~~~bash
python -c 'import sys, nvflare, yaml; assert sys.version_info >= (3, 11); print(nvflare.__version__)'
git -C "$HOME/NVFlare" rev-parse HEAD
skopeo --version
curl --version | head -n 1
openssl version
test -r "$COCO_ROOT/trusted_system/capture-platform.py"
~~~

Expected: Python 3.11+, importable NVFlare/PyYAML, identical Git revisions on
both hosts, and working commands. Resolve missing packages or revision
differences before proceeding. Activate `coco-venv` again in new terminals.

### Host A: install Cosign v2

The image builder uses Cosign v2's signing/verification interface. Download its
versioned binary and checksum list from the
[official v2.6.1 release](https://github.com/sigstore/cosign/releases/tag/v2.6.1):

~~~bash
mkdir -p "$COCO_DIR/downloads/cosign"
cd "$COCO_DIR/downloads/cosign"
curl -fL --proto '=https' --tlsv1.2 -O \
  https://github.com/sigstore/cosign/releases/download/v2.6.1/cosign-linux-amd64
curl -fL --proto '=https' --tlsv1.2 -O \
  https://github.com/sigstore/cosign/releases/download/v2.6.1/cosign_checksums.txt
grep -E '  \*?cosign-linux-amd64$' cosign_checksums.txt > selected.sha256
test "$(wc -l < selected.sha256)" -eq 1
sha256sum -c selected.sha256
sudo install -m 0755 cosign-linux-amd64 /usr/local/bin/cosign
~~~

**Check — Host A:** `cosign version` must report `v2.6.1`. Stop on a checksum
mismatch or a different major version. The services section performs actual
registry signing and verification; a version command only checks installation.

### Host B: install Helm

Use the [official Helm installation instructions](https://helm.sh/docs/intro/install/)
with the versioned archive below:

~~~bash
mkdir -p "$COCO_DIR/downloads/helm"
cd "$COCO_DIR/downloads/helm"
curl -fL --proto '=https' --tlsv1.2 -O https://get.helm.sh/helm-v3.19.0-linux-amd64.tar.gz
curl -fL --proto '=https' --tlsv1.2 -O https://get.helm.sh/helm-v3.19.0-linux-amd64.tar.gz.sha256sum
sha256sum -c helm-v3.19.0-linux-amd64.tar.gz.sha256sum
tar -xzf helm-v3.19.0-linux-amd64.tar.gz
sudo install -m 0755 linux-amd64/helm /usr/local/bin/helm
~~~

**Check — Host B:** `helm version --short` must report `v3.19.0`. A checksum
failure means the archive must not be installed. Cluster access is checked after
Kubernetes is initialized, not at this stage.

Return to [host preparation](../PREREQUISITES.md#3-prepare-confidential-hardware--host-b).

## Prepare the workflow inputs

Complete the services and cluster sections first. This section bridges their
outputs to the existing measurement and provisioning guides.

### Host A: matching genpolicy artifacts

Obtain the official [Kata 3.29.0 tools archive](https://github.com/kata-containers/kata-containers/releases/tag/3.29.0).
Extract it in a private staging directory, not over Host A's installed runtime:

~~~bash
source "$HOME/coco-private/inputs.env"
source "$HOME/coco-venv/bin/activate"
mkdir -p "$COCO_DIR/downloads/kata-tools" "$COCO_DIR/kata-tools"
cd "$COCO_DIR/downloads/kata-tools"
curl -fL --proto '=https' --tlsv1.2 -O \
  https://github.com/kata-containers/kata-containers/releases/download/3.29.0/kata-tools-static-3.29.0-amd64.tar.zst
printf '%s  %s\n' \
  d7b2b5d846fce3cd8136a28b7626d7b543bfcab5edea2bf8b37591af122f63fb \
  kata-tools-static-3.29.0-amd64.tar.zst | sha256sum -c -
tar --zstd -xf kata-tools-static-3.29.0-amd64.tar.zst -C "$COCO_DIR/kata-tools"
export GENPOLICY="$COCO_DIR/kata-tools/opt/kata/bin/genpolicy"
export RULES="$COCO_DIR/kata-tools/opt/kata/share/defaults/kata-containers/rules.rego"
export SETTINGS="$COCO_DIR/kata-tools/opt/kata/share/defaults/kata-containers/genpolicy-settings.json"
"$GENPOLICY" -j "$SETTINGS" --version
cat "$COCO_DIR/kata-tools/opt/kata/VERSION"
test -r "$RULES" && test -r "$SETTINGS"
~~~

Use the resulting absolute paths in `admin/platform.env`; all three files must
come from the same version. Keep the tools archive separate from the selected
worker's installed Kata runtime.

**Check — Host A:** the archive's `VERSION` file must identify Kata `3.29.0`,
both accompanying rules/settings files must exist, and the version command must
succeed. Genpolicy has its own version (`0.1.0` in this archive), rather than
printing the Kata release number. Its CLI also loads settings before handling
`--version`, so retain the explicit `-j` argument. The workload builder
validates their supported format during provisioning. If a file is missing,
check the release archive layout; do not substitute files from another release.

### Host B: privileged capture dependencies

Capture uses an ordinary user for Kubernetes and evidence verification, and
`sudo python3` for QEMU inspection. The latter needs NVFlare imports as well as
Python 3.11+. A user virtual environment alone does not configure sudo's PATH.
On this dedicated reference worker, create a root-owned environment and add it
to this trusted operator's sudo `secure_path`:

~~~bash
source "$HOME/coco-private/inputs.env"
source "$HOME/coco-venv/bin/activate"
sudo python3 -m venv /opt/nvflare-capture-venv
sudo /opt/nvflare-capture-venv/bin/python -m pip install "$HOME/NVFlare"
printf 'Defaults:%s secure_path="/opt/nvflare-capture-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"\n' \
  "$(id -un)" | sudo tee /etc/sudoers.d/nvflare-capture-path >/dev/null
sudo chmod 0440 /etc/sudoers.d/nvflare-capture-path
sudo visudo -cf /etc/sudoers.d/nvflare-capture-path
~~~

This changes interpreter selection for that operator; it grants no additional
sudo permission. For an existing managed worker, have its administrator arrange
the equivalent supported environment using the site's sudo policy. See the
[sudoers documentation](https://www.sudo.ws/docs/man/sudoers.man/).

**Check — Host B:**

~~~bash
sudo python3 - "$COCO_ROOT/trusted_system/capture-running-launch.py" <<'PY'
import runpy
import shutil
import sys
assert sys.version_info >= (3, 11)
runpy.run_path(sys.argv[1])
assert shutil.which("crictl"), "crictl is missing from sudo PATH"
print("Privileged capture dependencies available")
PY
~~~

Expected: the success message using the same interpreter/path as capture.
Missing imports, a missing `crictl`, or a sudo policy error must be fixed before
capture. Do not install packages into Ubuntu's externally managed system Python.

### Host B: install the selected evidence verifier

For **SNP**, install the same pinned `snpguest` used by collector preparation:

~~~bash
cd "$COCO_DIR/downloads"
curl -fL --proto '=https' --tlsv1.2 -o snpguest \
  https://github.com/virtee/snpguest/releases/download/v0.9.2/snpguest
printf '%s  %s\n' \
  9f008af82c37ad1d152d6b6f22fe4bf892d45f278d8b2500f69591f238905481 \
  snpguest | sha256sum -c -
sudo install -m 0755 snpguest /usr/local/bin/snpguest
snpguest --version
~~~

Expected: version `0.9.2`; stop on checksum mismatch. Review
[snpguest](https://github.com/virtee/snpguest) and the independent platform TCB
baseline before filling the four capture minimums. Signature/collateral and TCB
verification happen during fresh measurement capture.

For **TDX**, use [rootless Podman](https://podman.io/docs/installation) on Host B:

~~~bash
sudo apt-get install -y podman uidmap slirp4netns fuse-overlayfs
grep "^$(id -un):" /etc/subuid /etc/subgid
podman info --format '{{.Host.Security.Rootless}}'
~~~

Expected: nonoverlapping subordinate UID/GID ranges assigned to this user and
`true` from Podman. If ranges are absent, have the administrator allocate unused
ranges, then reconnect. On Ubuntu, keep the packaged AppArmor configuration for
Podman; investigate denied user namespaces rather than disabling AppArmor globally.
Installing Docker on Host B can replace Kubernetes' containerd/runc packages;
use Docker on Host A and Podman for this worker-local verifier.

Build and run as the same ordinary capture user:

~~~bash
cd "$COCO_ROOT"
TDX_VERIFIER_ENGINE=podman bash trusted_system/tdx-verifier/build.sh \
  "$HOME/coco-private/tdx-verifier"
"$HOME/coco-private/tdx-verifier/tdx-evidence-verify" --version
~~~

Expected: a successful pinned-verifier version check. The generated executable
binds the engine and immutable local image ID; retain that user's image store.
Do not run it with `sudo`. Existing Docker-based verifier installations remain
supported through the default engine. Quote verification additionally requires
Intel PCS connectivity and a synchronized clock. See
[measurement verifier details](../MEASUREMENTS.md#make-a-tdx-verifier-available-if-needed).

### Host A: publish the collector, then transfer public inputs

Create a private Skopeo authfile using the publisher account from the services
guide. The commands there create `$COCO_DIR/private/registry-auth.json`.
Then prepare the selected collector:

~~~bash
cd "$COCO_ROOT"
cp trusted_system/collector-publication.env.example \
  "$HOME/coco-private/collector-publication.env"
editor "$HOME/coco-private/collector-publication.env"
~~~

Set these values in that configuration (absolute paths, with this host's values):

~~~bash
CPU_TEE="snp" # or tdx, matching Host B
COLLECTOR_REGISTRY_REPOSITORY="registry.coco.lab/coco/collector"
COLLECTOR_REGISTRY_CA="$HOME/coco-prerequisites/public/ca.pem"
COLLECTOR_REGISTRY_AUTH_FILE="$HOME/coco-prerequisites/private/registry-auth.json"
~~~

~~~bash
python3 trusted_system/prepare-collector.py \
  "$HOME/coco-private/collector-publication.env" \
  "$HOME/coco-private/collector-publication-1"
jq -er '.image | select(test("@sha256:[0-9a-f]{64}$"))' \
  "$HOME/coco-private/collector-publication-1/collector-image.json"
~~~

Expected: a published immutable collector image and `collector-image.json`.
The SNP helper verifies its pinned binary automatically. Use a new output
directory for subsequent publication; do not overwrite retained metadata.

Copy **only** the metadata and the public CA bundle to Host B, using the normal
file transfer method for your machines. Store the metadata at
`$HOME/coco-private/collector-image.json` and the CA at
`$COCO_DIR/public/ca.pem`. Host A's private directory stays on Host A.

**Check — Host B:**

~~~bash
COLLECTOR_IMAGE="$(jq -er .image "$HOME/coco-private/collector-image.json")"
CERT_DIR="$(mktemp -d)"
cp "$COCO_DIR/public/ca.pem" "$CERT_DIR/ca.crt"
printf '{"auths":{}}\n' > "$CERT_DIR/empty-auth.json"
skopeo inspect --authfile "$CERT_DIR/empty-auth.json" --cert-dir "$CERT_DIR" \
  "docker://$COLLECTOR_IMAGE" >/dev/null
rm -rf "$CERT_DIR"
~~~

Expected: successful inspection without publisher credentials. Fix registry
project visibility, DNS, or TLS if it fails. Guest pulls are tested separately
in the cluster instructions; a host pull alone does not prove guest trust.

### Map the resulting inputs

Use these outputs when following the existing guides. Replace paths with the
actual absolute locations on the indicated machine.

| Destination | Inputs from this appendix |
| --- | --- |
| Host B capture configuration | Node/RuntimeClass/config path from the cluster guide; collector image/metadata/public CA; selected verifier; independent SNP TCB floors. |
| Host A `service/platform.env` | `KBS_URL=https://trustee.coco.lab:8443`; `TRUSTEE_PUBLIC_CERT` pointing to public `ca.pem`; private `ADMIN_TOKEN`; compatible `KBS_CLIENT`; canonical `CURRENT_RESOURCE_POLICY_FILE` from Trustee setup. |
| Host A `admin/platform.env` | Matching genpolicy/rules/settings paths; selected RuntimeClass; later capture launch-profile output; Kubernetes service address/port from the actual cluster. |
| Host A `provision/cc_project.yml` | Trustee endpoint/TLS CA, public `as-signing-public.pem`, registry endpoint/TLS CA, `publisher_username_file` pointing to `$COCO_DIR/private/publisher-username`, `publisher_password_file` pointing to `$COCO_DIR/private/publisher-password`, and unchanged guest-local AA token URL. |

Host tools can use existing system CA trust when it already verifies an
endpoint. The provisioning schema still requires actual public CA/bundle files
to embed trust into guests. A registry TLS CA, Trustee TLS CA, and AS signing
public key have different purposes; the AS public key is not a TLS certificate.

**Check — both hosts:** confirm every configured file exists, public/private
files are on the intended machine, and the selected RuntimeClass matches the
CPU/GPU participant settings. On Host B, read the actual service IP using:

~~~bash
kubectl get service kubernetes -o jsonpath='{.spec.clusterIP}{"\n"}'
kubectl get runtimeclass
~~~

Do not invent capture outputs at this stage. They are generated by
[measurement capture](../MEASUREMENTS.md), after which Host A registers the
references, provisions images, and installs release handoffs. Return to the
[final prerequisite checklist](../PREREQUISITES.md#7-return-to-the-four-step-workflow).
