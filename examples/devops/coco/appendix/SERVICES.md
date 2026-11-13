# Prepare the trusted services machine

This is the Host A part of the [beginner prerequisite appendix](../PREREQUISITES.md).
Trustee runs here, on a different machine from Kubernetes. Run commands in Bash
as the same trusted publishing user unless a command explicitly uses `sudo`.
The examples start new services and new storage; do not run the initialization
commands against an existing registry or Trustee installation.

> This optional appendix helps beginners bring up the prerequisite systems for
> the NVFlare CoCo workflow. Consult the official documentation for each component
> and the version you deploy. If there is any discrepancy, the component’s
> official documentation takes precedence.

## 1. Prepare tools on Host A

Keep the settings selected in the parent appendix available in this shell:

~~~bash
set -euo pipefail
source "$HOME/coco-private/inputs.env"
export REGISTRY="$REGISTRY_HOST"
export KBS_URL="https://$TRUSTEE_HOST:8443"
umask 077
mkdir -p "$COCO_DIR"/{downloads,public,private,trustee}
sudo apt-get update
sudo apt-get install -y ca-certificates curl git gnupg jq openssl \
  python3 python3-yaml python3-venv skopeo
~~~

Install Docker Engine and its Compose/buildx plugins using the
[official Ubuntu instructions](https://docs.docker.com/engine/install/ubuntu/#install-using-the-apt-repository).
For a fresh Host A, the repository installation commands are:

~~~bash
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release; echo "$VERSION_CODENAME")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
~~~

Sign out and back in before continuing, then restore the shell environment:

~~~bash
set -euo pipefail
umask 077
source "$HOME/coco-private/inputs.env"
source "$HOME/coco-venv/bin/activate"
~~~

A failed setup command should stop the recipe. Docker group membership
grants control equivalent to root on Host A. Do not install these Docker
packages on Host B: their `containerd.io` package conflicts with the separately
managed Kubernetes runtime.

**Check on A:**

~~~bash
docker info >/dev/null && docker compose version
skopeo --version
python3 -c 'import yaml; print("Python YAML: ready")'
~~~

Expect successful commands and a Compose v2 version. If Docker permission is
denied, complete the new login; the image publisher must be able to use Docker
directly. If its daemon is unavailable, inspect `systemctl status docker` before
proceeding. Record the installed package versions with `dpkg-query -W` alongside
your deployment configuration.

## 2. Create DNS names and TLS certificates

Use names that resolve to Host A's reachable address. For this small lab,
add this line to `/etc/hosts` on **both A and B**, replacing it if an older
mapping exists:

~~~bash
printf '%s %s %s\n' "$SERVICES_IP" "$REGISTRY_HOST" "$TRUSTEE_HOST"
sudoedit /etc/hosts
~~~

The printed line is what to add. Host files alone do not configure guest DNS;
also complete the CoreDNS mapping in [Host B setup](HOST_CLUSTER.md).

The following is a lab CA recipe. If your organization already supplies valid
server certificates, use those instead. Keep their private keys on A and supply
the corresponding CA/bundle wherever the instructions below use `ca.pem`.
See [Harbor HTTPS configuration](https://goharbor.io/docs/2.14.0/install-config/configure-https/)
and [Trustee TLS configuration](https://github.com/confidential-containers/trustee/blob/v0.22.0/kbs/docs/config.md#http-server-configuration).

**On A, once for new keys:**

~~~bash
test ! -e "$COCO_DIR/private/ca-key.pem" || { echo 'CA already exists'; false; }
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:3072 \
  -out "$COCO_DIR/private/ca-key.pem"
openssl req -x509 -new -sha256 -days 3650 \
  -key "$COCO_DIR/private/ca-key.pem" -out "$COCO_DIR/public/ca.pem" \
  -subj '/CN=CoCo lab root CA' \
  -addext 'basicConstraints=critical,CA:TRUE' \
  -addext 'keyUsage=critical,keyCertSign,cRLSign'
for name in "$REGISTRY_HOST" "$TRUSTEE_HOST"; do
  test ! -e "$COCO_DIR/private/$name-key.pem"
  openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:3072 \
    -out "$COCO_DIR/private/$name-key.pem"
  openssl req -new -key "$COCO_DIR/private/$name-key.pem" \
    -out "$COCO_DIR/private/$name.csr" -subj "/CN=$name"
  printf 'subjectAltName=DNS:%s\nbasicConstraints=CA:FALSE\nkeyUsage=digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\n' \
    "$name" > "$COCO_DIR/private/$name.ext"
  openssl x509 -req -sha256 -days 365 \
    -in "$COCO_DIR/private/$name.csr" \
    -CA "$COCO_DIR/public/ca.pem" -CAkey "$COCO_DIR/private/ca-key.pem" \
    -CAcreateserial -extfile "$COCO_DIR/private/$name.ext" \
    -out "$COCO_DIR/public/$name.pem"
done
sudo install -m 0644 "$COCO_DIR/public/ca.pem" \
  /usr/local/share/ca-certificates/coco-lab-ca.crt
sudo update-ca-certificates
sudo install -d -m 0755 "/etc/docker/certs.d/$REGISTRY"
sudo install -m 0644 "$COCO_DIR/public/ca.pem" "/etc/docker/certs.d/$REGISTRY/ca.crt"
sudo systemctl restart docker
~~~

**Check on A; repeat name resolution on B:**

~~~bash
getent hosts "$REGISTRY_HOST" "$TRUSTEE_HOST"
for name in "$REGISTRY_HOST" "$TRUSTEE_HOST"; do
  openssl verify -CAfile "$COCO_DIR/public/ca.pem" \
    -verify_hostname "$name" "$COCO_DIR/public/$name.pem"
done
~~~

Both names must resolve to A, and both certificates must report `OK`. Fix
addresses, certificate names, validity dates, or clock synchronization if they
do not. Copy only `public/ca.pem` to B, install it in B's system trust store using
the same `install`/`update-ca-certificates` commands, and retain its path for
guest/provisioning trust configuration. Never copy the CA or service private keys.

A registry certificate already trusted by Skopeo's system roots needs no
additional host CA installation. The confidential guest has its own trust
configuration; keep a public CA/bundle available for the existing provisioning
and collector inputs rather than assuming host trust changes guest trust.

## 3. Deploy Harbor on Host A

Use the [Harbor 2.14 installation guide](https://goharbor.io/docs/2.14.0/install-config/)
and [official release assets](https://github.com/goharbor/harbor/releases/tag/v2.14.6).
Provide at least the official resource minimums and enough additional storage
for image builds. This example disables the HTTP listener and uses HTTPS 443;
Trustee will use 8443. Restrict published service ports through the deployment's
network firewall; Docker-published ports require Docker-aware firewall rules.

**On A:**

~~~bash
cd "$COCO_DIR/downloads"
curl -fLO https://github.com/goharbor/harbor/releases/download/v2.14.6/harbor-online-installer-v2.14.6.tgz
curl -fLO https://github.com/goharbor/harbor/releases/download/v2.14.6/harbor-online-installer-v2.14.6.tgz.asc
gpg --keyserver hkps://keyserver.ubuntu.com --recv-keys 644FF454C0B4115C
gpg --fingerprint 644FF454C0B4115C
gpg --verify harbor-online-installer-v2.14.6.tgz.asc harbor-online-installer-v2.14.6.tgz
~~~

Check the signing fingerprint against the
[official installer verification instructions](https://goharbor.io/docs/2.14.0/install-config/download-installer/).
Continue only after successful verification. Extract into a new installation
directory, then generate `harbor.yml` from the shipped template:

~~~bash
test ! -e "$COCO_DIR/harbor" || { echo 'Harbor directory already exists'; false; }
tar -xzf harbor-online-installer-v2.14.6.tgz -C "$COCO_DIR"
python3 - "$COCO_DIR" "$REGISTRY_HOST" <<'PY'
import getpass
import sys
from pathlib import Path
import yaml

root, hostname = sys.argv[1:]
path = Path(root) / 'harbor'
config = yaml.safe_load((path / 'harbor.yml.tmpl').read_text())
config['hostname'] = hostname
config.pop('http', None)
config['https'] = {
    'port': 443,
    'certificate': f'{root}/public/{hostname}.pem',
    'private_key': f'{root}/private/{hostname}-key.pem',
}
config['harbor_admin_password'] = getpass.getpass('New Harbor admin password: ')
config['database']['password'] = getpass.getpass('New Harbor database password: ')
if not config['harbor_admin_password'] or not config['database']['password']:
    raise SystemExit('Passwords must not be empty')
config['data_volume'] = f'{root}/harbor-data'
with (path / 'harbor.yml').open('x') as stream:
    yaml.safe_dump(config, stream, sort_keys=False)
PY
cd "$COCO_DIR/harbor"
sudo ./install.sh
~~~

**Check on A and B:**

~~~bash
curl --fail --cacert "$COCO_DIR/public/ca.pem" \
  "https://$REGISTRY_HOST/api/v2.0/health"
curl --cacert "$COCO_DIR/public/ca.pem" -sS -D - -o /dev/null \
  "https://$REGISTRY_HOST/v2/"
~~~

Expect Harbor health `healthy`. `/v2/` normally responds `401` with a Bearer
`WWW-Authenticate` challenge pointing to Harbor's HTTPS token service; it is not
an HTTPS failure. If Harbor is starting, inspect `docker compose ps` and the
failing component's logs from its installation directory. Resolve TLS/DNS or
unhealthy components before creating the project.

### Enable anonymous pulls and private publishing

Open `https://registry.coco.lab` in a browser that trusts your lab CA. Sign in as
the initial `admin` user. Following the
[project instructions](https://goharbor.io/docs/2.14.0/working-with-projects/create-projects/),
create project **coco** and select **Public**. Do not enable a project policy
that requires signatures on the initial test pull.

Create a project robot account named `publisher` with **Pull Repository** and
**Push Repository** permissions, following the
[robot account instructions](https://goharbor.io/docs/2.14.0/working-with-projects/project-configuration/create-robot-accounts/).
Save the generated robot name and secret privately on A; choose an expiry that
covers your deployment. Use the exported account name, including its prefix.

**On A:**

~~~bash
export REGISTRY_AUTH_FILE="$COCO_DIR/private/registry-auth.json"
read -r -p 'Harbor robot account name: ' PUBLISHER_USER
read -r -s -p 'Harbor robot secret: ' PUBLISHER_SECRET; printf '\n'
printf '%s\n' "$PUBLISHER_USER" > "$COCO_DIR/private/publisher-username"
printf '%s' "$PUBLISHER_SECRET" > "$COCO_DIR/private/publisher-password"
printf '%s' "$PUBLISHER_SECRET" | skopeo login --authfile "$REGISTRY_AUTH_FILE" \
  --username "$PUBLISHER_USER" --password-stdin "$REGISTRY"
printf '%s' "$PUBLISHER_SECRET" | docker login "$REGISTRY" \
  --username "$PUBLISHER_USER" --password-stdin
unset PUBLISHER_SECRET
chmod 0600 "$REGISTRY_AUTH_FILE"
chmod 0600 "$COCO_DIR/private/publisher-username" "$COCO_DIR/private/publisher-password"
printf '{"auths":{}}\n' > "$COCO_DIR/public/anonymous-auth.json"
~~~

Test actual publishing with a disposable, locally built image. A successful
login alone does not establish that publishing and guest pulls work.

~~~bash
mkdir -p "$COCO_DIR/registry-check"
printf 'registry test\n' > "$COCO_DIR/registry-check/check.txt"
printf 'FROM scratch\nCOPY check.txt /check.txt\n' > "$COCO_DIR/registry-check/Dockerfile"
docker build -t "$REGISTRY/coco/prerequisite-check:initial" "$COCO_DIR/registry-check"
docker push "$REGISTRY/coco/prerequisite-check:initial"
CHECK_DIGEST=$(skopeo inspect --authfile "$COCO_DIR/public/anonymous-auth.json" \
  --format '{{.Digest}}' "docker://$REGISTRY/coco/prerequisite-check:initial")
export CHECK_IMAGE="$REGISTRY/coco/prerequisite-check@$CHECK_DIGEST"
printf '%s\n' "$CHECK_IMAGE"
skopeo copy --authfile "$COCO_DIR/public/anonymous-auth.json" \
  "docker://$CHECK_IMAGE" "dir:$COCO_DIR/registry-check/pulled"
if skopeo copy --dest-authfile "$COCO_DIR/public/anonymous-auth.json" \
  "dir:$COCO_DIR/registry-check/pulled" "docker://$REGISTRY/coco/prerequisite-check:anonymous" \
  2> "$COCO_DIR/private/anonymous-push-error.txt"; then
  echo 'FAIL: anonymous publishing is allowed'; false
else
  grep -Eqi 'unauthorized|denied|authentication required' \
    "$COCO_DIR/private/anonymous-push-error.txt" || \
    { echo 'FAIL: publishing failed for a reason other than access denial'; false; }
  echo 'Expected: anonymous publishing denied'
fi
if printf 'deliberately-invalid-secret' | skopeo login \
  --authfile "$COCO_DIR/private/invalid-auth.json" \
  --username "$PUBLISHER_USER" --password-stdin "$REGISTRY" \
  2> "$COCO_DIR/private/invalid-login-error.txt"; then
  echo 'FAIL: invalid credentials accepted'; false
else
  grep -Eqi 'unauthorized|denied|invalid username|authentication required' \
    "$COCO_DIR/private/invalid-login-error.txt" || \
    { echo 'FAIL: login failed for a reason other than credential rejection'; false; }
  echo 'Expected: invalid credentials denied'
fi
~~~

Install the pinned Cosign v2 tool described in the parent appendix, then check
its registry signature storage using a disposable key pair:

~~~bash
cd "$COCO_DIR/registry-check"
cosign generate-key-pair
cosign sign --key cosign.key --tlog-upload=false "$CHECK_IMAGE"
DOCKER_CONFIG="$COCO_DIR/registry-check/anonymous-docker"
mkdir -p "$DOCKER_CONFIG"
printf '{"auths":{}}\n' > "$DOCKER_CONFIG/config.json"
DOCKER_CONFIG="$DOCKER_CONFIG" cosign verify --key cosign.pub \
  --insecure-ignore-tlog "$CHECK_IMAGE" >/dev/null
unset DOCKER_CONFIG
~~~

Expect all positive operations to succeed and both negative operations to fail
because access is denied, rather than because of DNS, TLS, or network failure.
`--insecure-ignore-tlog` is limited to this disposable test, which intentionally
does not publish to a transparency log. It does not disable registry TLS.
For details see [Cosign image signing](https://docs.sigstore.dev/cosign/signing/signing_with_containers/).

On B, use the public CA and an empty authfile to repeat `skopeo inspect` and
`skopeo copy` of the printed `CHECK_IMAGE`. Do not send publisher credentials.
If anonymous access fails, check that `coco` is public and that the same registry
hostname is used throughout. Keep the registry test image until guest testing
is complete.

## 4. Build a pinned Trustee service and client on Host A

Follow the [official Trustee installation guide](https://confidentialcontainers.org/docs/attestation/installation/)
and the [v0.22.0 source](https://github.com/confidential-containers/trustee/tree/v0.22.0).
This recipe builds the integrated KBS/AS/RVPS variant. It does not start the
upstream multi-service Compose example or expose AS/RVPS ports.

**On A:**

~~~bash
cd "$COCO_DIR"
git clone --branch v0.22.0 --depth 1 \
  https://github.com/confidential-containers/trustee.git trustee-source
test "$(git -C trustee-source rev-parse HEAD)" = \
  512fed65642015b849f38fb13bfdec7806639987
docker build --pull --file trustee-source/kbs/docker/Dockerfile \
  --tag "$REGISTRY/coco/trustee:v0.22.0" trustee-source
docker push "$REGISTRY/coco/trustee:v0.22.0"
TRUSTEE_DIGEST=$(skopeo inspect --authfile "$REGISTRY_AUTH_FILE" \
  --format '{{.Digest}}' "docker://$REGISTRY/coco/trustee:v0.22.0")
export TRUSTEE_IMAGE="$REGISTRY/coco/trustee@$TRUSTEE_DIGEST"
printf '%s\n' "$TRUSTEE_IMAGE" > "$COCO_DIR/trustee/image.txt"
~~~

Use the same upstream builder stage to build the administration client. This
keeps its Rust toolchain and libraries inside the build container:

~~~bash
docker build --file trustee-source/kbs/docker/Dockerfile --target builder \
  --tag coco-trustee-builder:v0.22.0 trustee-source
docker create --name coco-kbs-client-build coco-trustee-builder:v0.22.0 \
  sh -c 'cd /usr/src/trustee && cargo build --locked --release -p kbs-client'
docker start --attach coco-kbs-client-build
docker cp coco-kbs-client-build:/usr/src/trustee/target/release/kbs-client \
  "$COCO_DIR/trustee/kbs-client"
docker rm coco-kbs-client-build
sudo install -m 0755 "$COCO_DIR/trustee/kbs-client" /usr/local/bin/kbs-client
~~~

**Check on A:**

~~~bash
docker run --rm "$TRUSTEE_IMAGE" /usr/local/bin/kbs --version
kbs-client config --help | grep -- --admin-token-file
~~~

Expect the service binary to execute and the client to advertise bearer-token
administration. If either build fails, retain the error and resolve the pinned
build's requirements before continuing; do not replace the image with `latest`.

## 5. Configure and start Trustee on Host A

Read the [pinned KBS configuration reference](https://github.com/confidential-containers/trustee/blob/v0.22.0/kbs/docs/config.md)
and [persistent storage configuration](https://github.com/confidential-containers/trustee/blob/v0.22.0/deps/key-value-storage/README.md).
The AS signing key is separate from the HTTPS key and the administration key.
The v0.22.0 EAR implementation requires a P-256 signing key (ES256).

**On A, once for new keys:**

~~~bash
test ! -e "$COCO_DIR/private/as-signing-key.pem"
test ! -e "$COCO_DIR/private/admin-signing-key.pem"
openssl ecparam -name prime256v1 -genkey -noout \
  -out "$COCO_DIR/private/as-signing-key.pem"
openssl pkey -in "$COCO_DIR/private/as-signing-key.pem" -pubout \
  -out "$COCO_DIR/public/as-signing-public.pem"
openssl req -new -key "$COCO_DIR/private/as-signing-key.pem" \
  -out "$COCO_DIR/private/as-signing.csr" -subj '/CN=CoCo lab AS signer'
printf 'basicConstraints=CA:FALSE\nkeyUsage=critical,digitalSignature\n' \
  > "$COCO_DIR/private/as-signing.ext"
openssl x509 -req -sha256 -days 365 \
  -in "$COCO_DIR/private/as-signing.csr" \
  -CA "$COCO_DIR/public/ca.pem" -CAkey "$COCO_DIR/private/ca-key.pem" \
  -CAcreateserial -extfile "$COCO_DIR/private/as-signing.ext" \
  -out "$COCO_DIR/public/as-signing-cert.pem"
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:3072 \
  -out "$COCO_DIR/private/admin-signing-key.pem"
openssl pkey -in "$COCO_DIR/private/admin-signing-key.pem" -pubout \
  -out "$COCO_DIR/public/admin-signing-public.pem"
mkdir -p "$COCO_DIR/trustee/storage"/{kbs,attestation_service_policy}
~~~

From the NVFlare checkout's `examples/devops/coco` directory, seed the **new,
stopped** service's policies. Rego v0 is unsupported and must not be used.

~~~bash
cd "$COCO_ROOT"
test ! -e "$COCO_DIR/trustee/storage/kbs/resource-policy.rego" || \
  { echo 'Existing Trustee state: do not initialize again'; false; }
install -m 0600 service/policies/default-deny-resource-policy.rego \
  "$COCO_DIR/trustee/storage/kbs/resource-policy.rego"
install -m 0600 service/policies/default_cpu.rego \
  "$COCO_DIR/trustee/storage/attestation_service_policy/default_cpu.rego"
install -m 0600 service/policies/default_gpu.rego \
  "$COCO_DIR/trustee/storage/attestation_service_policy/default_gpu.rego"
install -m 0600 service/policies/default-deny-resource-policy.rego \
  "$COCO_DIR/private/current-resource-policy.rego"
cat > "$COCO_DIR/trustee/kbs-config.toml" <<'EOF'
[http_server]
sockets = ["0.0.0.0:8443"]
insecure_http = false
private_key = "/run/coco/private/tls-key.pem"
certificate = "/run/coco/public/tls-cert.pem"

[attestation_token]
insecure_header_jwk = false
trusted_certs_paths = ["/run/coco/public/ca.pem"]

[attestation_service]
type = "coco_as_builtin"

[attestation_service.attestation_token_broker]
duration_min = 5
verbose_token = false

[attestation_service.attestation_token_broker.signer]
key_path = "/run/coco/private/as-signing-key.pem"
cert_path = "/run/coco/public/as-signing-cert.pem"

[attestation_service.rvps_config]
type = "BuiltIn"

[admin]
authorization_mode = "AuthenticatedAuthorization"

[[admin.authentication.bearer_jwt.identity_providers]]
issuer = "coco-lab-admin"
audience = "coco-kbs-admin"
public_key_uri = "file:///run/coco/public/admin-signing-public.pem"

[[admin.authorization.regex_acl.acls]]
role = "coco-admin"
allowed_endpoints = "^/kbs/v0/(attestation-policy|resource-policy|reference-value|resource)(/.*)?$"

[storage_backend]
storage_type = "LocalFs"

[storage_backend.backends.local_fs]
dir_path = "/var/lib/coco-trustee"

[[plugins]]
name = "resource"
storage_backend_type = "kvstorage"
EOF
~~~

Create a short-lived administration JWT without placing it in command-line
arguments. Renew it by repeating this command when it expires; keep the signer
and token on A. The token's role, issuer, and audience match the config above.

~~~bash
python3 - "$COCO_DIR" <<'PY'
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
def encode(value):
    return base64.urlsafe_b64encode(value).rstrip(b'=')
now = int(time.time())
header = encode(json.dumps({'alg': 'RS256', 'typ': 'JWT'}).encode())
claims = encode(json.dumps({
    'iss': 'coco-lab-admin', 'aud': 'coco-kbs-admin', 'role': 'coco-admin',
    'sub': 'platform-admin', 'iat': now, 'nbf': now - 30, 'exp': now + 3600,
}).encode())
message = header + b'.' + claims
signature = subprocess.run(
    ['openssl', 'dgst', '-sha256', '-sign', str(root / 'private/admin-signing-key.pem')],
    input=message, stdout=subprocess.PIPE, check=True,
).stdout
(root / 'private/kbs-admin-token').write_bytes(message + b'.' + encode(signature) + b'\n')
PY
chmod 0600 "$COCO_DIR/private/kbs-admin-token"
docker run -d --name coco-trustee --restart unless-stopped \
  --publish "$SERVICES_IP:8443:8443" \
  --mount "type=bind,src=$COCO_DIR/trustee/kbs-config.toml,dst=/etc/kbs-config.toml,readonly" \
  --mount "type=bind,src=$COCO_DIR/trustee/storage,dst=/var/lib/coco-trustee" \
  --mount "type=bind,src=$COCO_DIR/private/$TRUSTEE_HOST-key.pem,dst=/run/coco/private/tls-key.pem,readonly" \
  --mount "type=bind,src=$COCO_DIR/private/as-signing-key.pem,dst=/run/coco/private/as-signing-key.pem,readonly" \
  --mount "type=bind,src=$COCO_DIR/public/$TRUSTEE_HOST.pem,dst=/run/coco/public/tls-cert.pem,readonly" \
  --mount "type=bind,src=$COCO_DIR/public/ca.pem,dst=/run/coco/public/ca.pem,readonly" \
  --mount "type=bind,src=$COCO_DIR/public/as-signing-cert.pem,dst=/run/coco/public/as-signing-cert.pem,readonly" \
  --mount "type=bind,src=$COCO_DIR/public/admin-signing-public.pem,dst=/run/coco/public/admin-signing-public.pem,readonly" \
  "$TRUSTEE_IMAGE" /usr/local/bin/kbs --config-file /etc/kbs-config.toml
~~~

**Check health on A and B:**

~~~bash
curl --fail --cacert "$COCO_DIR/public/ca.pem" "$KBS_URL/healthz"
~~~

Expect HTTP 200. TLS failure means that DNS, CA, SAN, or clock configuration
needs repair. For startup failure inspect `docker logs coco-trustee` on A and
check configuration paths and mounts; health alone does not establish policy
or administration correctness.

**Check administration and persistent policy files on A:**

~~~bash
status=$(curl --cacert "$COCO_DIR/public/ca.pem" -sS -o /dev/null \
  -w '%{http_code}' "$KBS_URL/kbs/v0/resource-policy")
case "$status" in 401|403) echo 'Expected: unauthenticated administration denied';; \
  *) echo "FAIL: unexpected status $status"; false;; esac
python3 - "$COCO_DIR" "$KBS_URL" <<'PY'
import ssl
import sys
import urllib.request
from pathlib import Path

root, endpoint = sys.argv[1:]
root = Path(root)
request = urllib.request.Request(endpoint + '/kbs/v0/resource-policy')
request.add_header('Authorization', 'Bearer ' + (root / 'private/kbs-admin-token').read_text().strip())
with urllib.request.urlopen(request, context=ssl.create_default_context(cafile=str(root / 'public/ca.pem'))) as response:
    assert response.status == 200
print('Authenticated administration: ready')
PY
cmp service/policies/default_cpu.rego \
  "$COCO_DIR/trustee/storage/attestation_service_policy/default_cpu.rego"
cmp service/policies/default_gpu.rego \
  "$COCO_DIR/trustee/storage/attestation_service_policy/default_gpu.rego"
cmp "$COCO_DIR/private/current-resource-policy.rego" \
  "$COCO_DIR/trustee/storage/kbs/resource-policy.rego"
sha256sum "$COCO_DIR/public/as-signing-public.pem" \
  "$COCO_DIR/trustee/storage/kbs/resource-policy.rego" \
  "$COCO_DIR/trustee/storage/attestation_service_policy/"*.rego \
  > "$COCO_DIR/private/trustee-before-restart.sha256"
docker restart coco-trustee
sha256sum --check "$COCO_DIR/private/trustee-before-restart.sha256"
~~~

Repeat the health and authenticated-administration commands after restart.
Expect all policy comparisons and hash checks to pass. If authorization fails,
check token expiry, time, issuer/audience, role, ACL, and the mounted public key.
If files change or disappear, fix persistence before proceeding. Do not reset
policies on a running installation to make a check pass.

The resource-policy GET endpoint lists policy IDs; it does **not** export the
active policy contents. Inspect the trusted host's configured storage files for
the checks above. Retain `private/current-resource-policy.rego` as the canonical
complete export for later release registration, and serialize all policy updates
through that export. Do not recreate it from the initial deny policy after
workload registration has begun.

## 6. Supply the existing workflow's service inputs

On A, set the following in the existing `service/platform.env` as described in
[Trustee configuration](../TRUSTEE.md):

~~~bash
KBS_URL="https://trustee.coco.lab:8443"
TRUSTEE_PUBLIC_CERT="${HOME}/coco-prerequisites/public/ca.pem"
ADMIN_TOKEN="${HOME}/coco-prerequisites/private/kbs-admin-token"
KBS_CLIENT="/usr/local/bin/kbs-client"
CURRENT_RESOURCE_POLICY_FILE="${HOME}/coco-prerequisites/private/current-resource-policy.rego"
~~~

Keep the current export and any recovery files private on A. Copy only the
public CA/bundle and AS signing public key to the recipients that need them.
Set the registry to `registry.coco.lab/coco` in the appropriate repository input;
credentials authenticate to the registry host `registry.coco.lab`. The publisher
authfile stays on A. See the parent appendix for the exact provisioning and
collector configuration fields.

**Final service check:** A and B resolve the same service names and can verify
both HTTPS endpoints; A can publish while B can pull anonymously; only A has
administration credentials; the configured signer and storage survive restart.
No platform references or image keys have been fabricated during setup.
Continue with [Host B setup](HOST_CLUSTER.md), then return to the
[four-step workflow](../README.md#workflow). Capture supplies the actual platform
references. Provisioning generates image resources and a private release
handoff; it does not register that handoff with Trustee automatically.
