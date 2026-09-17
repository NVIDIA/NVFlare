# Upstream CoCo Trustee deployment

CVM Builder uses unmodified Trustee v0.22.0, paired with CoCo v0.23.0.
This directory contains deployment configuration and systemd templates only.
There are no Trustee or guest-components source patches.

Use [kbs.json](kbs.json) with the upstream binary/image and follow
[TRUSTEE_GUIDE.md](../TRUSTEE_GUIDE.md) for certificates, default CPU/GPU policies,
RVPS reference storage, administrative roles, and the create-only vault key adapter.

The KBS process reads resource, AS-policy and reference namespaces without write
access. Its publisher role can update the resource policy and query references.
The application key adapter handles atomic upload and revocation in the shared
upstream local_fs repository. Keep permanent tombstones outside resource backups
and reconcile before KBS starts after a restore.
