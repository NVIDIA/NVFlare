# Install secure services on a fresh host

First review [public-package configuration](../CONFIGURATION.md).

This procedure starts with a fresh Ubuntu VM and ends with Trustee/KBS,
Attestation Service (AS), RVPS, and a private OCI registry running behind TLS.
It does not require an existing service installation, cached images, old keys,
or recovery archives. Run service commands as `service_operator`, not root.

The service scripts are numbered **01–11 in fresh-install execution order**;
see [the command table](README.md#installation-order). Workload authorization
and verification follow as 12/13. Host-specific teardown is not packaged.
Later reference-only updates reuse 02 → 10 → 11, with no `--configure-only`.
These numbers apply only to the service kit; trusted-system stages are unchanged.

The service administrator is trusted. The CoCo cluster administrator is not.
The NVFlare provisioning node (`provisioning_node`) owns workload images and registry
publishing. The trusted platform system supplies the five platform-reference
values. Only public certificates and public workload/runtime handoffs go to
CoCo; never send it publisher credentials or private keys.

## 1. Prepare the secure services host and network

Have the infrastructure administrator provide:

- Ubuntu **24.04**, x86_64, with an SSH-key-authenticated `service_operator` account
  and passwordless sudo. Ubuntu 26.04 is not the service kit's validated OS.
- An OS disk of at least 80 GiB, with **at least 40 GiB free** when running
  preflight. Rust builds consume additional space; monitor it during stage 04
  and keep 40 GiB free for the later platform installation preflight too.
- A deployment-owned DNS name resolving to the host's public address.
  `secure-services.example.com` is only the documentation example. Configure
  and review all role settings, certificate SANs and workload validators for
  the real name before generating a release; changing one variable is not enough.
- A network firewall and any host firewall allowing TCP 22 from trusted management
  origins, and TCP **8443** and **5000** from the intended admin/CoCo origins.
  Do not expose TCP 8080, 5001, 50003, or 50004.
- Outbound DNS and HTTPS to Ubuntu package mirrors, GitHub, GHCR, Docker Hub,
  crates.io and its download hosts. Actual CPU/GPU attestation also needs
  access to the endorsement/verification providers used by the AS verifiers.

On the trusted coordinator, authenticate the VM's SSH host-key fingerprint
through an independent trusted channel before accepting it. Then verify access:

```bash
SERVICE_SSH=service_operator@secure-services.example.com
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$SERVICE_SSH" \
  'id; . /etc/os-release; printf "%s %s\n" "$ID" "$VERSION_ID"; sudo -n true; df -h /'
```

On **provisioning_node**, obtain its current public egress IPv4 address:

```bash
curl -4 --fail --silent --show-error --max-time 15 https://api.ipify.org
echo
```

Record that address as an exact `/32` for `REGISTRY_PUBLISHER_CIDR`. Do not
reuse an address from an old deployment record. This setting causes registry
clients at the publisher origin to receive an authentication challenge during
their initial `/v2/` ping; it does not replace password authentication.

## 2. Deliver the service kit

On the **trusted coordinator**, start with the reviewed local package. Verify
its hash inventory against an independently authenticated package source;
a checksum file alone does not establish who supplied it.

```bash
cd /path/to/coco_deployment
sha256sum --check --strict PACKAGE-SHA256SUMS
SERVICE_SSH=service_operator@secure-services.example.com
ssh "$SERVICE_SSH" 'test ! -e /home/service_operator/coco-service-admin && install -d -m 0700 /home/service_operator/coco-service-admin'
scp -r service/. "$SERVICE_SSH:/home/service_operator/coco-service-admin/"
```

Transfer only `service/` to this VM, not the entire role-separated package.
No certificate snapshots are shipped; later steps generate new certificates.

On **secure services**:

```bash
cd /home/service_operator/coco-service-admin
sudo -n true
umask 077
test ! -e platform.env
cp platform.env.example platform.env
chmod 0600 platform.env
hostname -f
editor platform.env
# Set EXPECTED_HOSTNAME and SERVICE_FQDN; review all paths and pins.
bash ./01-install-host-tools.sh
```

Stage 01 installs Docker/Compose, Nginx, OpenSSL, compiler/build dependencies,
Git, curl, jq, Python/YAML, zstd and supporting tools. It also installs the
required OPA 1.8.0 CLI for x86_64 to `/usr/local/bin/opa`, checking the download
against a pinned SHA-256 before installation, and prints `opa version`.
It enables Docker and Nginx. Install tools before running `03-preflight.sh` on
a fresh OS. Existing installations upgrading this kit must also rerun stage 01.

Do not run stage 03 yet: the distributed TCB fields deliberately have no
defaults. Obtain authenticated platform inputs first.

## 3. Receive the five trusted platform-reference values

The trusted_system operator supplies only `platform-reference-values.json`: one SNP
launch measurement and four minimum TCB integers. Follow
[PLATFORM-REFERENCE-VALUES-HANDOFF.md](PLATFORM-REFERENCE-VALUES-HANDOFF.md)
for export after trusted-system stage 09, authenticated transfer and the exact
five-key format. Do not request a signed archive, Kata artifacts, SNP report,
signing key or AS policy in this handoff.

For multiple approved launch measurements, the same five-key file can contain
a measurement array and one common set of four floors. Both configuration-only
stage 02 and initial policy stage 09 preserve the full array. Follow
[MEASUREMENT-ALLOWLIST.md](MEASUREMENT-ALLOWLIST.md) for approval and exact-list
replacement semantics; all other installation commands below are unchanged.

On **secure services**, receive the file under `~/incoming-platform/`. Validate and
review it, then prepare the existing local environment file:

```bash
cd /home/service_operator/coco-service-admin
VALUES=/home/service_operator/incoming-platform/platform-reference-values.json
python3 ./lib/platform-reference-values.py validate "$VALUES"
bash ./02-install-platform-reference-values.sh "$VALUES" \
  --approve-platform-reference-values --configure-only
```

The secure services owner must trust the sender and approve the values for the intended
platform. The JSON validator checks format and types, not authenticity or
hardware security. The configure-only operation changes only five fields in
`platform.env`, retaining a private backup. It does not access any backend.

Set the admin egress IPv4/32 collected in step 1, without changing other fields:

```bash
read -r -p 'Admin public egress IPv4/32: ' PUBLISHER_CIDR
python3 - ./platform.env "$PUBLISHER_CIDR" <<'PY'
import ipaddress
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
network = ipaddress.ip_network(sys.argv[2], strict=True)
if network.version != 4 or network.prefixlen != 32:
    raise SystemExit("Publisher origin must be one exact IPv4 /32")
text, count = re.subn(
    r'^REGISTRY_PUBLISHER_CIDR=.*$',
    f'REGISTRY_PUBLISHER_CIDR="{network}"',
    path.read_text(),
    flags=re.M,
)
if count != 1:
    raise SystemExit("Expected exactly one publisher-origin assignment")
path.write_text(text)
path.chmod(0o600)
PY
bash ./03-preflight.sh
```

Do not continue unless preflight reports zero failures. AS CPU/GPU policies
come from the reviewed service software/kit already controlled by secure services.

## 4. Build the pinned Trustee executables

On **secure services**:

```bash
cd /home/service_operator/coco-service-admin
bash ./04-build-trustee-main.sh
```

Stage 04 fetches Trustee, checks out
`338610fbfed57b66c61a8a3a60e0e4386bdce793`, and builds KBS, AS with all
verifiers, RVPS, and the SNP/TDX-capable KBS administration client. It vendors
checksum-verified Actix HTTP 3.13.3 with a 128 KiB HTTP request-head buffer.
**Do not use Trustee v0.21.0.** The HTTP header issue requires this pinned
post-v0.21 source and patch.

No NVFlare repository checkout or Kubernetes installation is needed on secure services.
Builds can take tens of minutes. In another SSH terminal, monitor `df -h /`
and `sudo docker stats --no-stream`; do not mistake a quiet build for failure.

Outputs live under `/home/service_operator/trustee-main-338610f/`:

- `built-image-ids.txt`: built KBS, AS and RVPS image IDs;
- `kbs-client-snp-tdx-main-338610f`: administration client;
- the pinned source checkout and build inputs.

## 5. Start Trustee and create its TLS identity

```bash
cd /home/service_operator/coco-service-admin
bash ./05-deploy-trustee.sh
bash ./06-configure-trustee-tls.sh
bash ./07-harden-kbs-admin-audience.sh
```

Stage 05 creates fresh Compose services, persistent data directories, admin
signing material/token, and a default-deny KBS resource policy. The admin issuer
requires audience `KBS`. RVPS uses the unified LocalFs configuration and a
host-mounted persistent directory.

Stage 06 generates a new RSA key and self-signed Trustee leaf certificate with
the service DNS SAN, then installs the Nginx HTTPS endpoint on **8443**. Both
the normal health check and a 40,000-byte Authorization-header probe must pass.
Stage 07 verifies admin audience enforcement without changing release policy.

## 6. Start the private registry and create its TLS identity

```bash
cd /home/service_operator/coco-service-admin
bash ./08-deploy-private-registry.sh
```

This creates an independent registry CA, registry server key/certificate, and
random publisher password. The registry backend uses TLS on loopback port
5001; Nginx exposes verified TLS on **5000**. Anonymous reads are permitted
outside the publisher origin, while mutations require publisher credentials.
The publisher origin is challenged on every method so OCI tools discover
authentication at `/v2/`. CoCo needs no registry password.

Stage 08 verifies an unauthenticated mutation is denied, starts an authenticated
test upload, checks that its URL preserves `https://FQDN:5000/`, and cancels
the upload. It does not publish a workload image.

## 7. Install secure services' policy and the five platform references

The fresh AS needs secure services' reviewed CPU policy installed once. Stage 09 uses
the five approved fields configured in step 3 and installs the CPU policy
from the service kit, not from trusted_system:

```bash
cd /home/service_operator/coco-service-admin
VALUES=/home/service_operator/incoming-platform/platform-reference-values.json
bash ./09-install-platform-policy.sh --approve-pinned-snp-platform
bash ./10-verify-platform-reference-values.sh "$VALUES"
bash ./11-verify-service.sh
```

Stage 09 stages restrictive numeric floors, installs the reviewed CPU policy,
then registers the approved measurement and four scalar TCB floors in RVPS.
The pinned default GPU policy remains active. No workload key is authorized;
KBS remains default deny until a separate workload handoff is installed.

Stage 10 reads every reference back through the authenticated KBS admin API
and compares it against the JSON. Require five `PASS` lines. Stage 11 checks
the CPU policy, expected values, TLS, key permissions and backend bindings.

Verify persistence on **secure services**:

```bash
source ./lib/common.sh
sudo find "$REFERENCE_DIR" -maxdepth 3 -type f -print
sudo docker compose -p "$TRUSTEE_PROJECT" -f "$TRUSTEE_COMPOSE" restart rvps
sleep 3
bash ./10-verify-platform-reference-values.sh "$VALUES"
bash ./11-verify-service.sh
```

The measurement and all four TCB floors must survive the restart. If RVPS is
still starting, wait and rerun verification. Do not rewrite values to mask a
persistence failure.

For subsequent **reference-only** updates, use stage 02 without
`--configure-only`, then stage 10. It requires the reviewed CPU policy already
installed and leaves AS CPU/GPU and KBS release-policy files unchanged. See
[the five-value handoff procedure](PLATFORM-REFERENCE-VALUES-HANDOFF.md).
The legacy signed platform-bundle verifier and installer have been removed;
this workflow accepts only the five-value JSON handoff.

## 8. Locate and distribute the new public certificates

On **secure services**:

```bash
cd /home/service_operator/coco-service-admin
install -m 0644 /home/service_operator/trustee-public.crt public/trustee.crt
install -m 0644 /home/service_operator/.coco-publisher/registry-ca.crt public/registry-ca.crt
sudo install -o service_operator -g service_operator -m 0644 \
  /etc/coco-registry/tls/server.crt public/registry-server.crt
sha256sum public/trustee.crt public/registry-ca.crt public/registry-server.crt
openssl x509 -in public/trustee.crt -noout -subject -dates -ext subjectAltName
openssl verify -CAfile public/registry-ca.crt \
  -verify_hostname secure-services.example.com public/registry-server.crt
```

| Material | Path on secure services | Recipient |
|---|---|---|
| Trustee public certificate | `/etc/trustee/tls/trustee.crt`; copy `~/trustee-public.crt` | Admin and CoCo |
| Registry CA certificate | `/etc/coco-registry/pki/ca.crt`; copy `~/.coco-publisher/registry-ca.crt` | Admin and CoCo |
| Registry server certificate | `/etc/coco-registry/tls/server.crt` | Public; optional inspection copy |
| Trustee private key | `/etc/trustee/tls/trustee.key` | secure services only, root:root 0600 |
| Registry CA private key | `/etc/coco-registry/pki/ca.key` | secure services only, root:root 0600 |
| Registry server private key | `/etc/coco-registry/tls/server.key` | secure services only, root:root 0600 |
| Publisher username/password | `~/.coco-publisher/{username,password}` | Trusted admin only, confidential transfer |
| KBS admin token | `~/trustee-main-338610f/kbs/config/docker-compose/admin-token` | secure services only, root:root 0600 |

Trustee and the registry deliberately use separate private keys and certificate
chains. Clients trust the Trustee leaf and registry CA, never a private key.
Record the freshly generated hashes through an authenticated channel; do not
copy certificate fingerprints from an earlier installation.

On a **trusted coordinator** with authenticated SSH access to all parties,
transfer only the public files. Replace client addresses with their current
FQDNs and make sure host keys have been authenticated:

```bash
SERVICE_SSH=service_operator@secure-services.example.com
ADMIN_SSH=operator@ADMIN_FQDN
COCO_SSH=operator@COCO_FQDN
PUBLIC_STAGE=$(mktemp -d)
scp "$SERVICE_SSH:/home/service_operator/coco-service-admin/public/trustee.crt" "$PUBLIC_STAGE/"
scp "$SERVICE_SSH:/home/service_operator/coco-service-admin/public/registry-ca.crt" "$PUBLIC_STAGE/"
sha256sum "$PUBLIC_STAGE/trustee.crt" "$PUBLIC_STAGE/registry-ca.crt"
# Compare these with the service administrator's authenticated hashes.
ssh "$ADMIN_SSH" 'install -d -m 0755 ~/service-public'
ssh "$COCO_SSH" 'install -d -m 0755 ~/service-public'
scp "$PUBLIC_STAGE/trustee.crt" "$PUBLIC_STAGE/registry-ca.crt" "$ADMIN_SSH:service-public/"
scp "$PUBLIC_STAGE/trustee.crt" "$PUBLIC_STAGE/registry-ca.crt" "$COCO_SSH:service-public/"
```

## 9. Give publishing authority only to provisioning_node

Obtain explicit service-owner authorization before exporting the new publisher
credentials. On the **trusted coordinator**, the following streams only the
two credential files over authenticated SSH, without saving them locally or
printing them. It does not require direct secure services-to-admin SSH access:

```bash
ssh "$ADMIN_SSH" 'test ! -e ~/incoming-registry-credential && install -d -m 0700 ~/incoming-registry-credential'
set -o pipefail
ssh "$SERVICE_SSH" 'tar -C /home/service_operator/.coco-publisher -cf - username password' \
  | ssh "$ADMIN_SSH" 'umask 077; tar --no-same-owner -xf - -C ~/incoming-registry-credential'
```

Never pipe to a terminal, use shell tracing, or copy this directory to CoCo.
On **provisioning_node**, whose separate admin kit is at `~/coco-admin`:

```bash
cd ~/coco-admin
install -m 0644 ~/service-public/trustee.crt public/trustee.crt
install -m 0644 ~/service-public/registry-ca.crt public/registry-ca.crt
bash ./00-install-tools.sh
bash ./05-install-publisher-credential.sh ~/incoming-registry-credential
REGISTRY=secure-services.example.com:5000
for cert_dir in /etc/docker/certs.d /etc/containers/certs.d; do
  sudo install -d -m 0755 "$cert_dir" "$cert_dir/$REGISTRY"
  sudo install -m 0644 ~/service-public/registry-ca.crt "$cert_dir/$REGISTRY/ca.crt"
done
SECRETS="$HOME/coco-workload-owner/secrets/registry"
skopeo login --authfile "$SECRETS/config.json" \
  --username "$(<"$SECRETS/username")" --password-stdin "$REGISTRY" < "$SECRETS/password"
chmod 0600 "$SECRETS/config.json"
```

The parent `certs.d` directories must be traversable by the non-root publisher
(0755); making only the final hostname directory readable is insufficient.
Leave TLS verification enabled. The username/password installer accepts
exactly those two files and refuses to overwrite existing credentials.
Remove the received staging copy according to your secret-handling procedure;
do not remove the installed credential files. See the admin kit for building,
signing, encrypting and publishing workload images.

## 10. Configure CoCo's public trust and verify client access

On **CoCo**, after its separate cluster installation and with the CoCo kit at
`~/coco-it`:

```bash
install -m 0644 ~/service-public/registry-ca.crt ~/coco-it/public/registry-ca.crt
install -m 0644 ~/service-public/trustee.crt ~/coco-it/public/trustee.crt
REGISTRY=secure-services.example.com:5000
sudo install -d -m 0755 /etc/containerd/certs.d "/etc/containerd/certs.d/$REGISTRY"
sudo install -m 0644 ~/service-public/registry-ca.crt "/etc/containerd/certs.d/$REGISTRY/ca.crt"
```

The CoCo cluster setup must configure containerd's registry `config_path` and
`hosts.toml` to use that CA; copying a certificate alone does not configure a
new containerd installation. Follow
`../coco/40-configure-registry-trust.sh` (provided separately)
and the CoCo installation guide for the installed containerd version. The workload owner's generated Pod InitData
must embed the new Trustee certificate. A host CA installation does not add
trust inside a confidential VM or update previously generated Pod InitData.

On **both client nodes**, verify Trustee using the distributed certificate:

```bash
curl --fail --silent --show-error --connect-timeout 10 --max-time 30 \
  --cacert ~/service-public/trustee.crt -o /dev/null -w '%{http_code}\n' \
  https://secure-services.example.com:8443/healthz
```

Require 200. On **CoCo**, verify anonymous registry reads:

```bash
curl --fail --silent --show-error --connect-timeout 10 --max-time 30 \
  --cacert ~/service-public/registry-ca.crt -o /dev/null -w '%{http_code}\n' \
  https://secure-services.example.com:5000/v2/
```

Require 200. Admin's unauthenticated ping normally returns 401 because of the
publisher-origin rule; the authenticated Skopeo login in step 9 must succeed.
If admin and CoCo share one NAT address, the origin rule also challenges CoCo.
Do not solve this by giving CoCo publisher credentials; establish separate
origins or have the trusted service administrator review the authentication
design before proceeding.

## 11. Authorize a workload only after the base services pass

The registry starts empty and KBS initially releases no workload resources.
The workload owner next builds/signs/encrypts/publishes an image and generates
InitData, a one-file Pod handoff for CoCo, and a separate six-file confidential
`trusted-service/` handoff for the service administrator.

Follow [TRUSTED-HANDOFF-RUNBOOK.md](TRUSTED-HANDOFF-RUNBOOK.md) to authenticate
the workload owner and validate the handoff. Its checksum file detects
corruption; it is not itself a sender signature. On **secure services**, review and run:

```bash
cd /home/service_operator/coco-service-admin
bash ./12-install-trusted-service-handoff.sh /home/service_operator/incoming/RELEASE
```

The installer displays the complete resource-policy diff, requires the release
name, and validates all six files. It renders the policy from the secure-services
template and rejects a received fragment that differs. All three resources are
installed and verified before committing the authorization policy last. Only
the approved workload identity and resource paths may release keys. OPA is a
required local syntax and authorization-test CLI installed by stage 01; no
standalone OPA server is needed. KBS performs final policy validation.

After CoCo IT launches the unchanged workload-owner Pod YAML, verify CPU/GPU
attestation and the three attested resource requests on **secure services**:

```bash
cd /home/service_operator/coco-service-admin
bash ./13-verify-workload-release.sh RELEASE 30m
```

A running container or an HTTPS health check is not proof of successful
attestation-gated key release. Retain this verification evidence securely.

## Final service layout

| Component | Public endpoint | Backend binding / persistent state |
|---|---|---|
| Nginx → Trustee KBS | HTTPS 8443 | `127.0.0.1:8080`; Trustee `kbs/data/kbs-storage` and `kbs-policy` |
| Attestation Service | None directly | `127.0.0.1:50004`; Trustee `kbs/data/attestation-service` |
| RVPS | None directly | `127.0.0.1:50003`; Trustee `kbs/data/reference-values` |
| Nginx → registry | HTTPS 5000 | `127.0.0.1:5001`; `/var/lib/coco-registry` |

Inspect on secure services with `sudo docker ps`, `sudo nginx -t`, and `sudo ss -lntp`.
Private keys, publisher credentials, KBS administration and stored workload
keys remain restricted to their trusted owners. Public reachability never
grants attestation approval or resource-release authorization.
