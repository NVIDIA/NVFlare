# Local coordinator: prepare and distribute a clean package

Start in `NVFlare/examples/devops/coco` and follow
[CONFIGURATION.md](../CONFIGURATION.md). Run the offline validator and checksum
check before transferring code. This machine does not execute deployment
installers locally merely to validate the package.

Authenticate each destination's SSH host key independently. Set role-specific
SSH targets in a private local record; no machine identities are included here.
Transfer only the relevant clean role directory, with executable modes retained.
For example, after setting `SERVICE_SSH` to the authenticated actual target:

```bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$SERVICE_SSH" \
  'test ! -e "$HOME/coco-service-admin" && install -d -m 0700 "$HOME/coco-service-admin"'
scp -o StrictHostKeyChecking=yes -rp service/. "$SERVICE_SSH:coco-service-admin/"
```

Compare received hashes against the authenticated package inventory before
execution. Use `admin/` for the provisioning node, `trusted_system/` for the
trusted platform system, and `coco/` for CoCo IT. If a guide uses package-root
paths, adjust to the received role directory; do not transfer private material
from another role just to make paths match.

Follow the dependency order and destination table in CONFIGURATION.md. The
provisioning node coordinates transfer of the trusted system's outputs. Secure
services receives only its five-value JSON; admin receives its separate launch
contract. CoCo receives only the public chart/pins, registry CA, and final Pod
YAML, with its hash authenticated separately.

Do not copy credentials into this source checkout. Transfer the publisher
credential directly from secure services to admin over an authenticated
confidential channel. Transfer each confidential workload handoff directly
from admin to secure services. KBS administration and TLS private keys remain
on secure services; the signing private key remains on admin.

Retain execution logs and approvals privately, not in the public package. Never
regenerate the publication allowlist from a directory containing live outputs.
