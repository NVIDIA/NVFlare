# Distributed Provisioning — Interactive Mode (Bash)

Single recommended flow. Use a **single-site** `site.yml` (one participant per file).

## 0) Create `site.yml`
Use `../site.template.yml` as your starting point (copy to `site.yml`).

## 1) Project Admin: root CA
```bash
./01_project_admin_init_ca.sh <project_name> <ca_dir>
```

## 2) Site Admin: CSR (reads site.yml)
```bash
./02_site_admin_csr.sh <site_yaml> <csr_dir>
```

## 3) Project Admin: sign CSR
```bash
./03_project_admin_sign.sh <csr_path> <ca_dir> <out_dir>
```

## 4) Site Admin: package startup kit (uses site.yml)
Place these files in a single folder (site bundle):
- `<name>.key`
- `<name>.crt` (or `.pem`)
- `rootCA.pem`

Then run:
```bash
./04_site_admin_package.sh <site_yaml> <server_endpoint> <site_bundle_dir>
```

Notes:
- `openssl` is required by `04_site_admin_package.sh` to verify the `rootCA.pem`
  fingerprint before packaging.
- Before packaging or startup, verify the returned `rootCA.pem` fingerprint with the
  Project Admin through a trusted out-of-band channel. Example:
  `openssl x509 -in rootCA.pem -noout -fingerprint -sha256`
- Root CA does **not** require org.
- Org **is required** for each participant to align with project.yml and must match `^[A-Za-z0-9_]+$` (no hyphens or spaces).
