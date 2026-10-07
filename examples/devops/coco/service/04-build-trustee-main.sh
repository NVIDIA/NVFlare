#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

require_sudo
for command in curl docker git python3 sha256sum tar; do need_cmd "${command}"; done
[[ "${TRUSTEE_COMMIT}" != "258ea4acb7b9bd865fce5c63a539f2120dba8298" ]] || \
    die 'Trustee v0.21.0 is prohibited'

if [[ ! -e "${TRUSTEE_ROOT}" ]]; then
    git clone "${TRUSTEE_REPOSITORY}" "${TRUSTEE_ROOT}"
fi
[[ -d "${TRUSTEE_ROOT}/.git" ]] || die "not a Git checkout: ${TRUSTEE_ROOT}"
git -C "${TRUSTEE_ROOT}" fetch origin "${TRUSTEE_COMMIT}"
git -C "${TRUSTEE_ROOT}" checkout --detach "${TRUSTEE_COMMIT}"
[[ "$(git -C "${TRUSTEE_ROOT}" rev-parse HEAD)" == "${TRUSTEE_COMMIT}" ]]

CRATE="$(mktemp)"
trap 'rm -f "${CRATE}"' EXIT
curl --fail --location --proto '=https' --tlsv1.2 \
    --output "${CRATE}" \
    "https://static.crates.io/crates/actix-http/actix-http-${ACTIX_HTTP_VERSION}.crate"
printf '%s  %s\n' "${ACTIX_HTTP_SHA256}" "${CRATE}" | sha256sum --check --strict
install -d -m 0755 "${TRUSTEE_ROOT}/vendor"
tar -xzf "${CRATE}" -C "${TRUSTEE_ROOT}/vendor"

python3 - "${TRUSTEE_ROOT}" "${ACTIX_HTTP_VERSION}" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
version = sys.argv[2]
actix = root / "vendor" / f"actix-http-{version}" / "src" / "h1" / "mod.rs"
text = actix.read_text()
old = "const HW: usize = 32 * 1024;"
new = "const HW: usize = 128 * 1024;"
if new in text and old not in text:
    pass
elif text.count(old) == 1 and new not in text:
    actix.write_text(text.replace(old, new, 1))
else:
    raise SystemExit("unexpected Actix request-head buffer source")

manifest = root / "Cargo.toml"
text = manifest.read_text()
patch = f'actix-http = {{ path = "vendor/actix-http-{version}" }}'
if patch not in text:
    if text.count("[patch.crates-io]") != 1:
        raise SystemExit("unexpected Cargo patch layout")
    text = text.rstrip() + "\n" + patch + "\n"
manifest.write_text(text)

lock = root / "Cargo.lock"
text = lock.read_text()
parts = text.split("[[package]]")
matches = 0
for index in range(1, len(parts)):
    part = parts[index]
    if re.search(r'^name = "actix-http"$', part, re.MULTILINE) and \
       re.search(rf'^version = "{re.escape(version)}"$', part, re.MULTILINE):
        matches += 1
        part = re.sub(r'^source = .*\n', '', part, count=1, flags=re.MULTILINE)
        part = re.sub(r'^checksum = .*\n', '', part, count=1, flags=re.MULTILINE)
        parts[index] = part
if matches != 1:
    raise SystemExit(f"expected one actix-http lock entry, found {matches}")
lock.write_text("[[package]]".join(parts))

as_dockerfile = root / "attestation-service" / "docker" / "as-grpc" / "Dockerfile"
text = as_dockerfile.read_text()
install_old = (
    "    cargo install --path attestation-service --bin grpc-as "
    "--features grpc-bin,${VERIFIER} --locked ${TARGET_FLAG}"
)
install_new = (
    "    cargo install --root /opt/as-install --path attestation-service --bin grpc-as "
    "--features grpc-bin,${VERIFIER} --locked ${TARGET_FLAG} && \\\n"
    "    rm -rf /usr/local/cargo/git /usr/local/cargo/registry "
    "/usr/src/attestation-service/target"
)
if install_new not in text:
    if text.count(install_old) != 1:
        raise SystemExit("unexpected attestation-service cargo install layout")
    text = text.replace(install_old, install_new, 1)
copy_old = "COPY --from=builder /usr/local/cargo/bin/grpc-as /usr/local/bin/grpc-as"
copy_new = "COPY --from=builder /opt/as-install/bin/grpc-as /usr/local/bin/grpc-as"
if copy_new not in text:
    if text.count(copy_old) != 1:
        raise SystemExit("unexpected attestation-service binary copy layout")
    text = text.replace(copy_old, copy_new, 1)
as_dockerfile.write_text(text)

rvps = root / "rvps" / "docker" / "Dockerfile"
text = rvps.read_text()
text = text.replace(
    "FROM --platform=${BUILDPLATFORM:-linux/amd64} docker.io/library/rust:latest AS builder",
    "FROM --platform=${BUILDPLATFORM:-linux/amd64} "
    "docker.io/library/rust@sha256:f49565f188ee00bc2a18dd418183f2c5f23ef7d6e691890517ed341a598f67c3 AS builder",
)
text = text.replace(
    "FROM debian\n",
    "FROM ubuntu@sha256:7c06e91f61fa88c08cc74f7e1b7c69ae24910d745357e0dfe1d2c0322aaf20f9\n"
    "RUN apt-get update && apt-get install -y ca-certificates openssl && "
    "rm -rf /var/lib/apt/lists/*\n",
)
rvps.write_text(text)
PY

grep -Fxq 'const HW: usize = 128 * 1024;' \
    "${TRUSTEE_ROOT}/vendor/actix-http-${ACTIX_HTTP_VERSION}/src/h1/mod.rs"
[[ "$(git -C "${TRUSTEE_ROOT}" diff --name-only | sort | tr '\n' ' ')" == \
   'Cargo.lock Cargo.toml attestation-service/docker/as-grpc/Dockerfile rvps/docker/Dockerfile ' ]]

sudo docker build --pull=false --build-arg ARCH=x86_64 \
    --build-arg VAULT=false --build-arg EXTERNAL_PLUGIN=false \
    --file "${TRUSTEE_ROOT}/kbs/docker/coco-as-grpc/Dockerfile" \
    --tag "${KBS_IMAGE}" "${TRUSTEE_ROOT}"
sudo docker build --pull=false --build-arg ARCH=x86_64 --build-arg VERIFIER=all-verifier \
    --file "${TRUSTEE_ROOT}/attestation-service/docker/as-grpc/Dockerfile" \
    --tag "${AS_IMAGE}" "${TRUSTEE_ROOT}"
sudo docker build --pull=false --build-arg ARCH=x86_64 \
    --file "${TRUSTEE_ROOT}/rvps/docker/Dockerfile" \
    --tag "${RVPS_IMAGE}" "${TRUSTEE_ROOT}"

sudo docker run --rm --volume "${TRUSTEE_ROOT}:/src" --workdir /src \
    "${RUST_BUILDER}" cargo build --locked --release --package kbs-client \
    --features snp-attester,tdx-attester
sudo chown -R "$(id -u):$(id -g)" "${TRUSTEE_ROOT}/target"
install -m 0755 "${TRUSTEE_ROOT}/target/release/kbs-client" "${KBS_CLIENT}"

MANIFEST="${TRUSTEE_ROOT}/built-image-ids.txt"
for image in "${KBS_IMAGE}" "${AS_IMAGE}" "${RVPS_IMAGE}"; do
    printf '%s %s\n' "$(sudo docker image inspect --format '{{.Id}}' "${image}")" \
        "${image}"
done | tee "${MANIFEST}"
printf 'Built pinned post-v0.21 Trustee images and KBS client.\n'
