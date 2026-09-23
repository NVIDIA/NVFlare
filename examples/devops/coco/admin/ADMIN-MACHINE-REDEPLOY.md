# Admin machine: build, encrypt, sign, publish, and create handoffs

First complete [public-package configuration](../CONFIGURATION.md), including
`platform.env` and separately received public certificates.

Run this procedure as `operator` on the trusted Ubuntu 26.04 workload-owner
role `provisioning_node` on physical machine `provisioning-node`. This machine may see plaintext workloads and owns the
Cosign private key and per-release image key. It must never receive KBS
administration or any service TLS private key. It never logs in to Kubernetes.

## 1. Receive and verify the admin kit

The local coordinator copies the reviewed `admin/` directory to
`/home/operator/coco-admin`. Before running scripts:

```bash
cd /home/operator/coco-admin
find . -type f -print0 | sort -z | xargs -0 sha256sum
openssl x509 -in public/trustee.crt -noout \
  -subject -issuer -dates -ext subjectAltName -fingerprint -sha256
openssl x509 -in public/registry-ca.crt -noout \
  -subject -issuer -dates -fingerprint -sha256
```

Authenticate both fingerprints independently with the service administrator.
Do not accept certificates delivered through CoCo IT.

## 2. Select and authenticate the target-cluster platform profile

Complete this step before building an image for a different CoCo cluster. The
cluster operator is adversarial, so values reported only by that operator are
not trusted platform authorization inputs. A trusted platform or service
administrator must approve the profile and independently establish the TEE
launch measurement and GPU appraisal policy.

The current scripts source `platform.env` from the directory containing the
scripts. They do not accept a platform-profile option on the command line.
They additionally require `public/approved-workload-launch-profile.json`,
authenticated by `WORKLOAD_LAUNCH_PROFILE_SHA256` in `platform.env`.
Follow [APPROVED-LAUNCH-PROFILE.md](APPROVED-LAUNCH-PROFILE.md) to obtain and
install it from trusted_system. Stage 30 checks the generated Pod before and after
genpolicy, and stage 40 checks it again before packaging. These checks cover
container count, GPU resources, CPU/memory omissions, host namespaces and
runtime overrides. The profile pins these cluster-dependent values:

| Value | Required relationship to the target cluster |
|---|---|
| `RUNTIME_CLASS` | Exact confidential runtime class installed on the target |
| `WORKLOAD_LAUNCH_PROFILE_SHA256` | Trusted SHA-256 of the approved workload launch contract |
| `KUBERNETES_SERVICE_HOST` and `KUBERNETES_SERVICE_PORT` | Exact service environment authorized in the guest process |
| `KBS_URL` and `public/trustee.crt` | Independently trusted Trustee endpoint and certificate |
| `REGISTRY_HOST`, `REGISTRY_PORT`, and `public/registry-ca.crt` | Registry endpoint and authenticated public CA |

If the new cluster has exactly the same reviewed profile, the existing admin
kit can be reused with a new release name. If any value differs, make a
separate hash-verified kit directory instead of editing the profile used by an
existing immutable release:

```bash
cp -a "$HOME/coco-admin" "$HOME/coco-admin-cluster-b"
cd "$HOME/coco-admin-cluster-b"
```

The trusted coordinator updates `platform.env` and public certificates in the
new directory, records their hashes, and gives the workload user only the
approved profile. The workload user must not derive or change platform values.
Release state and secrets remain under `$HOME/coco-workload-owner`; therefore
`RELEASE_NAME` must be unique across every kit and every target cluster.

The SNP launch measurement and minimum reported-TCB floors are deliberately not
in the admin profile or KBS resource fragment. They are enforced by the
independent Attestation Service CPU policy. The resource fragment requires
`cpu0` and `gpu0` to carry the complete approved trust vectors. If cluster B
uses a different Kata kernel, firmware, rootfs, dm-verity configuration, or VM
launch configuration, the service administrator must independently approve
that measurement and implement a reviewed measurement allowlist or a separate
Trustee security domain. Never overwrite the existing approved measurement
with a value learned from the adversarial cluster. Never let the workload owner
or CoCo IT choose the bootloader, TEE, SNP-firmware, or microcode SVN floors.

Before generating a workload, CoCo IT must prove that the target cluster can
perform credential-free registry reads using only the public CA:

```bash
curl --fail --silent --show-error \
  --cacert public/registry-ca.crt \
  "https://secure-services.example.com:5000/v2/" \
  >/dev/null
```

The current registry challenges every request from the admin publisher's
source `/32`, while other origins may perform anonymous GET/HEAD. If cluster B
shares the publisher egress address, the probe will return HTTP 401. Do not
give registry publisher credentials to CoCo. The service administrator must
first separate pull and publication identity, for example with a reviewed
publisher mTLS endpoint or a distinct publication hostname.

Use one new immutable release and one new image key for each image and target
platform authorization tuple. Even when the application bytes are reused,
create a separate release when command, init-data, platform profile, or key
authorization must differ between clusters.

## 3. Install owner tools and public trust

Display/hash and run:

```bash
./00-install-tools.sh
```

This installs Docker, Go, Skopeo, OpenSSL, jq, YAML support, and zstd; installs
the registry CA for Docker; downloads checksum-pinned Cosign 2.6.2 and Kata
3.29.0 `genpolicy`; and probes Trustee and the registry using the supplied
public certificates. Before the publisher credential is installed, the
registry probe must receive HTTP 401 with Basic realm
`CoCo registry publisher`; this proves the service is challenging the exact
publisher origin. An anonymous HTTP 200 at this stage is a configuration
error. The validated script SHA-256 is
`1579dc93be163d697c530f1a5cfe6d71dbad33364d87cd12e3dfaafb627cb7c8`.

## 4. Install the publisher credential

Receive a directory containing exactly `username` and `password` from the
service administrator over a confidential authenticated channel. The username
must be `coco-publisher`. Do not display the password.

```bash
chmod 0700 /path/to/received-publisher
chmod 0600 /path/to/received-publisher/{username,password}
./05-install-publisher-credential.sh /path/to/received-publisher
```

The installed destination is
`$HOME/coco-workload-owner/secrets/registry/`. Remove the received staging copy
after verification. Never transfer this credential to CoCo.

## 5. Prepare the supplied minimal example

The example is a static scratch image running as UID/GID 65532. It accepts no
arguments, exposes no port, emits no output, contains no shell/package manager,
and waits only for termination.

```bash
./example-workload/build-example.sh
release_env="$HOME/coco-admin/live-workload-$(date -u +%Y%m%dT%H%M%SZ).env"
cp workload.example.env "$release_env"
chmod 0600 "$release_env"
$EDITOR "$release_env"
```

Replace the release placeholder with a unique lowercase DNS label and confirm
the absolute build context, Dockerfile, repository, command `[/coco-app]`, and
UID/GID. A release name is immutable and must not be reused for different bytes
or policy. The file `live-workload-20260828-v2.env` records this deployment and
is evidence, not a reusable release template.

For a real workload, replace the example build context. Use a fixed non-root
identity, no SSH/admin service, no host-mounted secrets, no sensitive image
metadata, and owner-controlled mTLS for externally visible success. Never emit
secrets to stdout/stderr because the hostile cluster controls Kubernetes logs.

## 6. Build and review plaintext

```bash
./10-build-plaintext.sh "$HOME/UNIQUE-RELEASE.env"
```

Stop after this command. Review the exact image inspect and untruncated Docker
history paths printed by the script. Run vulnerability and functional testing
on admin. Check for secrets, environment values, unintended entrypoint/command,
shells, package managers, SSH, and UID/GID mismatch. Do not continue until the
plaintext is approved.

## 7. Encrypt, publish, and sign by immutable digest

```bash
./20-encrypt-sign-publish.sh "$HOME/UNIQUE-RELEASE.env" \
  --approve-reviewed-plaintext
./25-verify-published-image.sh "$HOME/UNIQUE-RELEASE.env"
```

The first command generates a new 32-byte image key, encrypts every OCI layer
with A256GCM and a release-specific KBS key URI, pushes ciphertext to the TLS
registry, resolves the immutable digest, creates or reuses the owner Cosign
identity, signs the digest without Rekor, and validates all encryption
annotations. Because the service challenges every request from the publisher
CIDR, the second command uses publisher authentication for the signature and
manifest checks while proving an unauthenticated write fails. Verify a truly
credential-free manifest and blob read independently from CoCo's
non-publisher origin. Plaintext stays only on admin in Docker and the mode-0700
release directory.

Kata `genpolicy` reads Docker-compatible registry credentials automatically.
Stage 30 exports `DOCKER_CONFIG` to the protected Skopeo auth directory; do not
copy that auth file beside the handoff or make it world-readable.

Back up `$HOME/coco-workload-owner/secrets/signing/` as the owner identity. Do
not transfer `cosign.key` or its password to service or CoCo.

## 8. Generate the Pod and measured enforcement policy

```bash
./30-generate-pod-and-policies.sh "$HOME/UNIQUE-RELEASE.env"
```

The script pins the exact encrypted image digest, command vector, non-root
identity, runtime class, and one confidential GPU. Kata `genpolicy` produces
the guest agent policy. The script disables arbitrary environment matching and
verifies default denial of exec, attach/read/write stream, and policy
replacement. It embeds compressed init-data containing Trustee/registry public
trust and default-deny image signature policy into the Pod.

Kata 3.29.0 is affected by CVE-2026-77176 / GHSA-fmg6-v47x-52wr. The script
applies the exact semantics of upstream workaround commit
`788e2df4f4a332b2675317363ba1d2803c591d8b` to the pinned `rules.rego`, then
requires all four repaired mount/storage checks in the final embedded policy.
Generation takes place in a private temporary directory and installs outputs
only after every invariant passes, so an ordinary failure leaves no partial
policy release.

If recovering the one older incomplete run from before transactional output
was implemented, and only if `policy-SHA256SUMS` and the handoff directory are
both absent, display/hash and run:

```bash
./31-recover-failed-policy-generation.sh "$HOME/UNIQUE-RELEASE.env" \
  --remove-incomplete-policy-build
./30-generate-pod-and-policies.sh "$HOME/UNIQUE-RELEASE.env"
```

Never use the recovery script on a complete or handed-off release; it refuses
both conditions.

Review every file in
`$HOME/coco-workload-owner/releases/RELEASE/output/`, especially `pod.yaml`,
`generated-policy.rego`, `final-initdata.toml`,
`image-security-policy.json`, `release-authorization.json`, and
`resource-policy-fragment.rego`. The init-data is integrity protected but
public; do not put secrets in it.

### How two or more image-release rules coexist

The generated `resource-policy-fragment.rego` is the exact authorization block
for one immutable release. It contains no `package`, `import`, or `default`
declaration. The trusted service installer appends complete fragments to one
global policy whose header appears exactly once:

```rego
package policy

import rego.v1

default allow := false

# complete release-A fragment
# complete release-B fragment
```

Multiple top-level `allow if` rules are logical alternatives, but each rule is
bound to its own complete release tuple. Every generated fragment contains a
unique prefix derived from `RELEASE_NAME` and checks all of the following:

```rego
<prefix>_expected_initdata := "<64-lowercase-hex-init-data-digest>"
<prefix>_expected_image := "<registry/repository@sha256:immutable-digest>"
<prefix>_expected_args := ["<absolute-program>", "<optional-argument>"]

<prefix>_authorized_path(path) if {
    path == ["default", "image-key", "<release>"]
}
<prefix>_authorized_path(path) if {
    path == ["default", "sig-public-key", "<release>"]
}
<prefix>_authorized_path(path) if {
    path == ["default", "security-policy", "<release>"]
}

<prefix>_expected_trust_vector := {
    "executables": 3,
    "hardware": 2,
    "configuration": 3,
    "file-system": 0,
    "instance-identity": 0,
    "runtime-opaque": 0,
    "storage-opaque": 0,
    "sourced-data": 0,
}

<prefix>_approved_trust_vector(submod) if {
    submod["ear.trustworthiness-vector"] ==
        <prefix>_expected_trust_vector
}

<prefix>_approved_container(container) if {
    container["OCI"]["Annotations"]["io.kubernetes.cri.image-name"] ==
        <prefix>_expected_image
    container["OCI"]["Process"]["Args"] == <prefix>_expected_args
}

allow if {
    data.plugin == "resource"
    <prefix>_authorized_path(data["resource-path"])
    count(input.submods) == 2
    <prefix>_approved_trust_vector(input.submods.cpu0)
    <prefix>_approved_trust_vector(input.submods.gpu0)

    cpu := input.submods.cpu0
    cpu["ear.veraison.annotated-evidence"]["init_data"] ==
        <prefix>_expected_initdata
    <prefix>_expected_image in
        cpu["ear.trustee.identifiers"]["validated"]["container_images"]

    containers :=
        cpu["ear.veraison.annotated-evidence"]["init_data_claims"]["agent_policy_claims"]["containers"]
    some container in containers
    <prefix>_approved_container(container)
}
```

The angle-bracket form above documents the generated structure; it is not an
installable policy and must never be completed by hand. Obtain the exact image
digest, command, init-data digest, prefix, and paths only from:

```text
$HOME/coco-workload-owner/releases/RELEASE/output/release-authorization.json
$HOME/coco-workload-owner/releases/RELEASE/output/resource-policy-fragment.rego
```

For two images, the final service policy is the single default-deny header plus
the complete first fragment plus the complete second fragment. Do not combine
images with a broad `image_a or image_b` rule, share image-key paths, authorize
a repository tag, or let one release retrieve another release's resources.

The KBS resource policy does not directly approve a SNP launch measurement. It
requires exactly the CPU and GPU submodules with complete approved trust
vectors. The platform-owned Attestation Service policies and reference values
produce those vectors. Both layers are required before a decryption key is
released.

## 9. Create and route the two disjoint handoffs

```bash
./40-create-handoffs.sh "$HOME/UNIQUE-RELEASE.env"
```

Outputs:

- `handoff/trusted-service/`: six files including `image_key`; confidential
  authenticated transfer only to the service administrator;
- `handoff/coco-it/RELEASE-pod.yaml`: the only file delivered to CoCo IT;
- `handoff/expected-pod-sha256.txt`: communicate through a separate
  authenticated channel, never bundle with the YAML.

Wait for the service administrator to review and install the trusted-service
handoff before authorizing launch. The cluster owner can edit the Pod, image,
command, or policy, but then measured attestation/resource policy must refuse
the three KBS resources. The checksum detects transfer tampering; Trustee/KBS
enforcement is the security boundary.

## 10. Owner completion criteria and teardown

The owner records the immutable image digest, Pod SHA-256, service approval,
and an application-level mTLS success signal. Kubernetes readiness alone is not
owner proof. Negative tests after policy/tool upgrades must include modified
command, image digest, init-data, unsigned image, `kubectl exec`, unavailable
Trustee, and missing, extra, or changed CPU/GPU trust-vector evidence.

For maintenance, see [the package boundary](../trusted_system/TEARDOWN.md).
