# 1. Capture measurements from the installed runtime

Use an existing selected worker and its installed Kata configuration. The
capture tools do not install Kubernetes, Kata, operators, or Trustee, and do not
download a replacement Kata runtime.

Run baseline capture under trusted platform-owner control. A fresh signed
report/quote verifies the evidence against the configured security baseline and
minimum TCB. Successful capture writes the reference and launch-profile files.

## Configure capture

From the example directory:

~~~bash
CONFIG="$HOME/coco-private/capture.env"
mkdir -p "$(dirname "$CONFIG")"
cp trusted_system/platform.env.example "$CONFIG"
chmod 0600 "$CONFIG"
editor "$CONFIG"
~~~

The configuration describes the existing worker:

| Input | Value |
| --- | --- |
| `NODE_NAME` | Kubernetes worker on which the collector must run. |
| `RUNTIME_CLASS` | Selected RuntimeClass from the README table. |
| `INSTALLED_KATA_CONFIG` | Installed Kata configuration TOML. |
| `KATA_RUNTIME` | Existing Kata runtime executable. |
| `KUBECONFIG_PATH` | Kubeconfig authorized to run the collector. |
| `COLLECTOR_IMAGE` | Available collector image by immutable digest. |
| `COLLECTOR_METADATA_FILE` | Collector publication metadata. |
| `COLLECTOR_REGISTRY_CA` | Registry CA certificate or existing trust-bundle file. |
| `SNP_VERIFIER` or `TDX_VERIFIER` | Existing compatible evidence-verifier executable. |

For SNP, fill the four independent `SNP_MIN_REPORTED_TCB_*` floors for bootloader,
TEE, SNP firmware, and microcode. Each is a decimal integer from 0 through 255.
Choose these minima from an independently maintained platform security baseline,
not merely the worker's reported versions. A zero floor imposes no lower bound
on that component; an all-zero set provides no minimum-TCB protection. Capture
also rejects signed SNP policies that permit debugging or a migration agent.
SNP certificate retrieval uses HTTPS to AMD KDS and relies on the verifier's
system CA trust; capture does not independently pin the AMD root certificate.
For TDX, provide the compatible DCAP verifier wrapper. Use trusted preinstalled
verifier executables from the operator's tool environment. Verifiers check fresh
evidence signatures/collateral and the configured platform constraints.

The selected worker needs Python 3.11+ with PyYAML, Bash, kubectl, skopeo,
OpenSSL, iproute2's `ip`, containerd's `ctr`, and the separate CRI client `crictl`.
The interpreter selected by `sudo python3` must also be Python 3.11+ and able
to import the capture helper and its NVFlare dependencies from this checkout.
`crictl` must be executable in that interpreter's effective PATH, not only in
the caller's shell. Capture checks the privileged interpreter, helper imports,
and `crictl` before creating output, caching images, or launching the collector.
Capture uses `sudo ctr` to cache the image and `sudo python3` to inspect its
QEMU process. The kubeconfig
must permit reading the selected Node, creating the temporary namespace, Pod,
and challenge ConfigMap, reading Pod status/logs, and cleanup.
Run capture on that worker; placement and the observed node must agree.
The collector image must be readable without publisher credentials. Its supplied
CA/bundle is scoped to the Skopeo operations automatically.

## Make a TDX verifier available if needed

Skip this section when `TDX_VERIFIER` already names a compatible verifier.
The supplied helper builds a Docker-based verifier on an x86_64 capture worker;
Docker must be usable directly or through noninteractive `sudo`.

~~~bash
VERIFIER_DIR="$HOME/coco-private/tdx-verifier"
bash trusted_system/tdx-verifier/build.sh "$VERIFIER_DIR"
"$VERIFIER_DIR/tdx-evidence-verify" --version
~~~

Set this absolute generated-wrapper path in `$CONFIG`:

~~~bash
TDX_VERIFIER="$HOME/coco-private/tdx-verifier/tdx-evidence-verify"
~~~

`run-verifier.sh` in the source is a template, not the generated executable.
Build on the capture worker with the same Docker daemon used for verification:
the generated wrapper embeds a local image ID. The version command checks that
wrapper/image; quote verification additionally needs HTTPS access to Intel PCS
and an accurate clock. Capture supplies evidence JSON, challenge bytes, and
InitData TOML to the verifier, which returns verified claims JSON. The supplied
verifier checks DCAP signatures/collateral, challenge and InitData binding,
event-log replay, non-debug mode, unexpired collateral, and `UpToDate` TCB.

## Make a collector available if needed

Skip this section if a suitable collector image and metadata already exist.
On the trusted publishing machine, use Python 3.11+, Docker directly as the
publishing user, skopeo, and curl for SNP collector preparation.

~~~bash
PUBLISH_CONFIG="$HOME/coco-private/collector-publication.env"
mkdir -p "$(dirname "$PUBLISH_CONFIG")"
cp trusted_system/collector-publication.env.example "$PUBLISH_CONFIG"
chmod 0600 "$PUBLISH_CONFIG"
editor "$PUBLISH_CONFIG"
python3 trusted_system/prepare-collector.py \
  "$PUBLISH_CONFIG" "$HOME/coco-private/new-collector"
~~~

Set `CPU_TEE=snp` or `tdx`, the registry repository and CA/bundle, and an existing
private Skopeo authfile in `COLLECTOR_REGISTRY_AUTH_FILE`. The helper uses that
CA/bundle automatically for publication; no separate Skopeo trust setup is needed.

Supply its image reference, `collector-image.json`, and public registry CA to
the capture worker. Keep publisher credentials on the build machine.

## Capture measurements

~~~bash
OUTPUT="$HOME/coco-private/new-capture"
python3 trusted_system/capture-platform.py "$CONFIG" "$OUTPUT"
~~~

Use a new output directory. For TDX only, its basename must match
`[a-z0-9][a-z0-9_.-]{0,63}`: 1–64 characters starting with a lowercase letter
or digit, followed by lowercase letters, digits, dots, underscores, or hyphens.
This naming restriction applies to the basename, not the parent path.
Capture records the installed runtime and actual launch, obtains fresh
challenge-bound SNP or TDX evidence, and verifies it.

Capture writes `capture-summary.json`, the retained verified evidence, and the
two output files directly. For SNP, the signed report supplies the measurement;
the configured minimum TCB floors remain verification inputs. TDX requires its
complete verified measurement tuple and acceptable TCB/collateral.

The generated NVFlare launch shape is fixed: UID/GID 65532, non-root,
no privilege escalation, dropped capabilities, and writable guest rootfs.
GPU profiles request one confidential GPU; CPU-only profiles request none.
No CPU/memory overrides, host mounts, or extra containers are added.

## Use the output

- `platform-reference-values.json` goes to [Trustee](TRUSTEE.md).
- `approved-workload-launch-profile.json` goes to the image builder.

Keep raw evidence and capture records private. Give each recipient the output
it needs and use the received files directly.
Continue with [setting Trustee values](TRUSTEE.md).
