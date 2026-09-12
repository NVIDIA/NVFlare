# Workload-owner kit for `admin`

To package NVFlare clients and their signed startup kits, use
[project.yaml-based provisioning](../provision/README.md). The manual stages
below remain available for other workloads.

Start with [../CONFIGURATION.md](../CONFIGURATION.md). Configure `platform.env`
from `platform.env.example` before invoking any installer.

This kit is for a workload owner using the trusted `admin` machine. It builds a
plaintext container, encrypts every OCI layer, publishes the ciphertext, signs
the immutable manifest digest, generates a strict Kata agent policy, and creates
the final Pod YAML.

The Kubernetes/CoCo owner is adversarial. No script in this kit logs in to the
cluster or runs `kubectl`. The only cluster handoff is one Pod YAML file.

Read [SECURITY-MODEL.md](SECURITY-MODEL.md) before using the kit. The short IT
procedure is in [COCO-IT-RUNBOOK.md](COCO-IT-RUNBOOK.md).

Before generating a Pod, install and authenticate the required
[approved workload launch profile](APPROVED-LAUNCH-PROFILE.md) from trusted_system.
Stages 30 and 40 reject missing, changed or incompatible profiles; they no
longer rely only on manually matching `platform.env` and Pod resource fields.

## Roles and outputs

| Role | Receives | Must never receive |
|---|---|---|
| Workload owner on `admin` | Build context, signing key, registry publisher credential, temporary image key | KBS admin token or Trustee TLS private key |
| Trusted `service` administrator | Per-release image key, public signing key, image policy, authorization fragment | Cosign private key, registry password, plaintext image |
| Adversarial `coco` IT | One final `*-pod.yaml` plus its SHA-256 over an independent channel | Every key, credential, build input, service bundle, plaintext layer |

The service administrator must install the resources and merge the authorization
fragment before IT launches the Pod. The workload owner must not be given KBS
administrative credentials, and the cluster owner must not administer Trustee.

## Multiple adversarial CoCo clusters

Before targeting another CoCo cluster, use the target-cluster onboarding and
multi-release policy procedure in
[ADMIN-MACHINE-REDEPLOY.md](ADMIN-MACHINE-REDEPLOY.md). In particular:

- authenticate a platform-owned profile rather than trusting values supplied
  only by the adversarial cluster owner;
- use a separate admin kit directory when runtime class, Kubernetes service
  environment, Trustee, registry trust, or other platform values differ;
- independently approve TEE/GPU measurements and minimum SNP reported-TCB
  floors in the service appraisal layer;
- verify credential-free registry pulls from the new cluster without giving it
  publisher credentials;
- create a globally unique immutable release and image key for each image and
  target-platform authorization tuple; and
- merge one complete, uniquely prefixed KBS resource-policy fragment per
  release instead of creating a broad rule shared by multiple images.

## 1. One-time setup

No certificates are included. Receive `public/trustee.crt` and
`public/registry-ca.crt` from the secure-services administrator, then verify
their fingerprints through an independent trusted channel before first use:

```bash
cd /path/to/admin
openssl x509 -in public/trustee.crt -noout -subject -issuer -dates -fingerprint -sha256
openssl x509 -in public/registry-ca.crt -noout -subject -issuer -dates -fingerprint -sha256
./00-install-tools.sh
```

After registry certificate rotation, refresh both the registry-specific CA
directories and `/usr/local/share/ca-certificates/coco-registry-ca.crt`, then
run `sudo update-ca-certificates`. Kata `genpolicy` uses system trust roots.
Stages 20 and 25 explicitly pass `public/registry-ca.crt` to Cosign with
`--registry-cacert`; do not disable TLS verification to work around stale CAs.

Install the publisher credential without placing its password on a command line
or in shell history:

```bash
./05-install-publisher-credential.sh /path/to/authenticated/received-credential
```

The received directory must contain exactly `username` and `password`. The
installer does not display the password and refuses to overwrite an existing
credential. Securely remove the received copy after installation.

The persistent Cosign signing key is created on first publication under
`$HOME/coco-workload-owner/secrets/signing/`. Back it up as an owner signing
identity. Never transfer `cosign.key` or `cosign.password` to `service` or
`coco`.

## 2. Prepare the workload

The image is decrypted only inside the confidential guest, but visible OCI
metadata may remain. The Dockerfile must:

- use a fixed non-root UID/GID;
- contain no `sshd` and expose no administrator interface;
- avoid a shell and package manager where possible;
- contain no secrets in files, `ENV`, labels, build arguments, or history;
- write no secrets or sensitive payloads to stdout/stderr;
- use owner-controlled mTLS for any network interface.

Copy the example config. Use a new DNS-label `RELEASE_NAME` for every change to
the image, command, UID/GID, or policy:

```bash
cd /path/to/admin
cp workload.example.env "$HOME/my-workload-v1.env"
chmod 0600 "$HOME/my-workload-v1.env"
editor "$HOME/my-workload-v1.env"
```

`APP_COMMAND_JSON` is the complete process argument vector. It is not a shell
command. This kit deliberately supports no arbitrary Pod environment variables,
volumes, probes, ports, service-account token, host namespaces, or additional
containers. Bake non-secret configuration into the image and fetch secrets only
through an independently authorized attested channel.

## 3. Build and review plaintext

```bash
./10-build-plaintext.sh "$HOME/my-workload-v1.env"
```

For the example Dockerfile, first run `bash example-workload/build-example.sh`
after stage 00 and set `BUILD_CONTEXT` to its actual absolute directory.

The script stops after the build. Inspect both files printed by the script and
run local functional and vulnerability tests. In particular, check Docker
history and image config for credentials and unintended commands. Do not debug
the plaintext image on `coco`.

## 4. Encrypt, publish, and sign

Only after approving the review output:

```bash
./20-encrypt-sign-publish.sh \
  "$HOME/my-workload-v1.env" --approve-reviewed-plaintext
./25-verify-published-image.sh "$HOME/my-workload-v1.env"
```

The scripts:

1. generate a fresh 32-byte image key for this release;
2. encrypt every layer with the pinned CoCo key-provider image;
3. bind the encryption annotation to this release's KBS key URI;
4. push to the TLS registry with the publisher credential;
5. resolve and sign the immutable registry digest;
6. verify the signature and encrypted layer annotations;
7. verify anonymous read succeeds and anonymous write fails.

Plaintext remains in Docker's local cache and under the private release working
directory on `admin`. It is never pushed to the registry.

## 5. Generate the measured Pod and policies

```bash
./30-generate-pod-and-policies.sh "$HOME/my-workload-v1.env"
```

This uses Kata `3.29.0` `genpolicy`, pinned rules/settings, the immutable encrypted
image digest, the exact command vector, UID/GID, one confidential GPU, and a
default-deny image signature policy. It embeds the compressed init-data into the
Pod and checks that `ExecProcessRequest`, stream requests, and policy replacement
remain default-denied.

### Executable inventory for stages 20 and 30

Every `.sh` file below is interpreted by `/usr/bin/env bash`. The table lists
the principal external executable for each operation separately from supporting
utilities or inputs. Kata `genpolicy` is the executable that generates the Kata
Agent policy and writes the encoded `cc_init_data` annotation into the Pod.

| Step | Shell script | Main executable(s) | Supporting executables or inputs |
|---|---|---|---|
| Validate reviewed plaintext image | `20-encrypt-sign-publish.sh` | `docker image inspect` | `sha256sum`, `awk` |
| Generate per-release image-encryption key | `20-encrypt-sign-publish.sh` | `openssl rand` | `chmod` |
| Generate persistent Cosign signing key, if absent | `20-encrypt-sign-publish.sh` | `cosign generate-key-pair` | `openssl rand`, `chmod` |
| Download the pinned CoCo key-provider image | `20-encrypt-sign-publish.sh` | `docker pull` | None |
| Start the CoCo key provider | `20-encrypt-sign-publish.sh` | `docker run` | CoCo `coco-keyprovider` container |
| Encrypt every OCI image layer | `20-encrypt-sign-publish.sh` | `skopeo copy` | CoCo key-provider protocol |
| Authenticate to the private registry | `20-encrypt-sign-publish.sh` | `skopeo login` | Bash `printf` |
| Publish the encrypted OCI image | `20-encrypt-sign-publish.sh` | `skopeo copy` | Registry authentication file |
| Resolve the immutable manifest digest | `20-encrypt-sign-publish.sh` | `skopeo inspect` | `jq` |
| Sign the immutable encrypted-image digest | `20-encrypt-sign-publish.sh` | `cosign sign` | None |
| Verify the image signature | `20-encrypt-sign-publish.sh` | `cosign verify` | None |
| Inspect and validate encrypted-layer annotations | `20-encrypt-sign-publish.sh` | `skopeo inspect`, `python3` | Python `base64` module |
| Save publication outputs and checksums | `20-encrypt-sign-publish.sh` | `install`, `sha256sum` | `chmod` |
| Harden Kata generation rules | `30-generate-pod-and-policies.sh` | `python3` | Pinned Kata rules and settings |
| Create image-signature policy | `30-generate-pod-and-policies.sh` | `python3` | Immutable image repository and KBS signing-key URI |
| Create `base-initdata.toml` | `30-generate-pod-and-policies.sh` | Bash here-document | Trustee certificate, registry CA and KBS paths |
| Create preliminary Pod YAML | `30-generate-pod-and-policies.sh` | `python3` with PyYAML | Release configuration and immutable image reference |
| Generate Kata Agent policy and `cc_init_data` | `30-generate-pod-and-policies.sh` | **Kata `genpolicy`** | Rego rules, JSON settings, base init-data, Pod YAML and OCI metadata |
| Add Kubernetes-only restrictions | `30-generate-pod-and-policies.sh` | `python3` with PyYAML | None |
| Decode and validate generated init-data | `30-generate-pod-and-policies.sh` | `python3` | Python `base64`, `gzip`, `hashlib`, `tomllib` and `yaml` modules |
| Calculate init-data SHA-256 | `30-generate-pod-and-policies.sh` | `python3` with `hashlib` | `sha256sum` for output-file checksums |
| Generate KBS resource-policy fragment | `30-generate-pod-and-policies.sh` | `python3` | Image, command, init-data digest and authorized KBS paths |
| Generate release authorization | `30-generate-pod-and-policies.sh` | `python3` | Release-specific policy values |
| Install completed policy outputs | `30-generate-pod-and-policies.sh` | `install` | `sha256sum` |

Review these files under
`$HOME/coco-workload-owner/releases/<release>/output/`:

- `pod.yaml` — the only eventual cluster input;
- `generated-policy.rego` — guest enforcement policy;
- `final-initdata.toml` — public measured init-data;
- `image-security-policy.json` — default-deny signature policy;
- `release-authorization.json` — exact service-side authorization values;
- `resource-policy-fragment.rego` — fragment for the service's global policy;
- `expected-initdata-sha256.hex` — lowercase hex expected from the pinned
  post-v0.21 Trustee SNP evidence format.

The init-data is integrity protected, not confidential. The cluster owner can
read the embedded agent policy, endpoints, and public certificates.

## 6. Create the two disjoint handoffs

After the review:

```bash
./40-create-handoffs.sh "$HOME/my-workload-v1.env"
```

The output is:

```text
handoff/
├── trusted-service/       # secret; authenticated transfer to service admin
│   ├── image_key
│   ├── cosign.pub
│   ├── image-security-policy.json
│   ├── release-authorization.json
│   ├── resource-policy-fragment.rego
│   └── SHA256SUMS
├── coco-it/               # contains exactly one file
│   └── <release>-pod.yaml
└── expected-pod-sha256.txt # authenticate independently; do not bundle with Pod
```

Transfer `trusted-service/` only to the independent service administrator over
an authenticated confidential channel. They must:

1. verify `SHA256SUMS`;
2. install `image_key`, `cosign.pub`, and `image-security-policy.json` at the
   three paths in `release-authorization.json`;
3. independently review the image, command, lowercase-hex init-data value, and
   CPU/GPU requirements;
4. merge the uniquely named Rego fragment into the existing global KBS resource
   policy;
5. preserve the platform-owned CPU/GPU attestation policies and trusted RVPS
   reference values;
6. test the complete merged policy and confirm the new three resources are
   releasable only when the EAR contains exactly `cpu0` and `gpu0`, each with
   the complete approved signed trust vector recorded in the authorization.

`set-resource-policy` replaces the global policy. The service administrator
must never upload the fragment by itself or let a workload user replace global
CPU/GPU/RVPS policy.

After service approval, manually deliver only
`handoff/coco-it/<release>-pod.yaml` to IT. Send the SHA-256 from
`expected-pod-sha256.txt` over a separate authenticated channel.

## 7. IT launch and owner-visible success

IT follows [COCO-IT-RUNBOOK.md](COCO-IT-RUNBOOK.md). The workload must signal
success to an owner-controlled endpoint over mTLS. Do not require SSH,
`kubectl exec`, `kubectl attach`, or `kubectl logs`.

The cluster may edit the YAML, substitute an image, alter the command, or replace
init-data. That is expected adversarial behavior. A modified workload must fail
the guest policy or fail attestation and receive no KBS resources. The Pod
checksum provides delivery error/tamper detection, but the service-side measured
authorization is the cryptographic enforcement boundary.

## 8. Mandatory negative tests for each policy/tool upgrade

Use a non-production release and confirm all of these fail closed:

- change one command argument without regenerating policy;
- replace the image digest;
- replace one byte of embedded init-data;
- try an unsigned image in the same registry;
- run `kubectl exec`, attach, or copy;
- make Trustee unavailable;
- request one of the three KBS paths with a missing/extra submodule or any
  changed CPU/GPU trust-vector component.

Do not learn production reference values from a cluster controlled by the
adversary.

## Versioned upstream references

- [Pinned Trustee init-data specification](https://github.com/confidential-containers/trustee/blob/338610fbfed57b66c61a8a3a60e0e4386bdce793/kbs/docs/initdata.md)
- [Pinned Trustee KBS attestation protocol](https://github.com/confidential-containers/trustee/blob/338610fbfed57b66c61a8a3a60e0e4386bdce793/kbs/docs/kbs_attestation_protocol.md)
- [Kata 3.29.0 genpolicy README](https://github.com/kata-containers/kata-containers/blob/3.29.0/src/tools/genpolicy/README.md)
- [Kata 3.29.0 generated policy behavior](https://github.com/kata-containers/kata-containers/blob/3.29.0/src/tools/genpolicy/genpolicy-auto-generated-policy-details.md)
- [CoCo image-rs design](https://github.com/confidential-containers/guest-components/blob/main/image-rs/docs/design.md)

Revalidate this kit whenever Kata, Trustee, guest-components, registry behavior,
the Kubernetes/containerd OCI contract, or certificates change.

## Maintenance

Lab-specific destructive teardown scripts are not included. See
[maintenance boundaries](../trusted_system/TEARDOWN.md).
