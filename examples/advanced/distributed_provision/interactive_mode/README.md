# Distributed Provisioning — Interactive Mode (Bash)

Role-separated shell scripts for a step-by-step federated provisioning flow.
Each step is owned by either the **Project Admin** or a **Site Admin**.

## Example scenario

| Role | Name | Type | Org | Host |
|---|---|---|---|---|
| Project Admin | — | — | NvidiaOrg | (local machine) |
| Server | `server` | `server` | NvidiaOrg | `server.example.com:8002` |
| Client 1 | `site-1` | `client` | Org1 | (local machine) |
| Client 2 | `site-2` | `client` | Org2 | (local machine) |

Working directory used throughout: `./distprov_demo`

---

## Step 0 — Create `site.yml` for each participant

Start from the template:
```bash
cp ../site.template.yml server.yml
cp ../site.template.yml site-1.yml
cp ../site.template.yml site-2.yml
```

Edit each file so it looks like the examples below.

**`server.yml`**
```yaml
name: server
org: NvidiaOrg
type: server
```

**`site-1.yml`**
```yaml
name: site-1
org: Org1
type: client
```

**`site-2.yml`**
```yaml
name: site-2
org: Org2
type: client
```

> `org` must match `^[A-Za-z0-9_]+$` (no hyphens or spaces).

---

## Step 1 — Project Admin: initialise root CA

```bash
./01_project_admin_init_ca.sh fed-project ./distprov_demo/ca
```

Outputs: `./distprov_demo/ca/rootCA.pem` and the CA private key.

---

## Step 2 — Site Admin: generate CSR + private key

Each site admin runs this on their own machine, one call per site.

```bash
# Server site admin
./02_site_admin_csr.sh ./server.yml ./distprov_demo/server-csr

# Client site-1 admin
./02_site_admin_csr.sh ./site-1.yml ./distprov_demo/site-1-csr

# Client site-2 admin
./02_site_admin_csr.sh ./site-2.yml ./distprov_demo/site-2-csr
```

Each `<csr_dir>` receives `<name>.csr` and `<name>.key`.
The site admin **keeps the `.key` private** and sends only the `.csr` to the Project Admin.

---

## Step 3 — Project Admin: sign each CSR

The Project Admin receives the `.csr` files and signs them one at a time.

```bash
./03_project_admin_sign.sh \
  ./distprov_demo/server-csr/server.csr \
  ./distprov_demo/ca \
  ./distprov_demo/server-signed

./03_project_admin_sign.sh \
  ./distprov_demo/site-1-csr/site-1.csr \
  ./distprov_demo/ca \
  ./distprov_demo/site-1-signed

./03_project_admin_sign.sh \
  ./distprov_demo/site-2-csr/site-2.csr \
  ./distprov_demo/ca \
  ./distprov_demo/site-2-signed
```

Each `<out_dir>` receives `<name>.crt` and `rootCA.pem`.
The script prints the `rootCA.pem` SHA256 fingerprint — share it with each site admin
through a **trusted out-of-band channel** (e.g. a secure Slack DM or signed email)
so they can verify it before packaging.

---

## Step 4 — Site Admin: assemble bundle and package startup kit

Each site admin assembles a bundle directory containing:
- `<name>.key` (kept from Step 2)
- `<name>.crt` and `rootCA.pem` (received from the Project Admin in Step 3)

Then runs `04_site_admin_package.sh`. The script prints the `rootCA.pem` fingerprint
and asks for confirmation before building the kit.

```bash
# Server site admin
mkdir -p ./distprov_demo/server-bundle
cp ./distprov_demo/server-csr/server.key   ./distprov_demo/server-bundle/
cp ./distprov_demo/server-signed/server.crt ./distprov_demo/server-bundle/
cp ./distprov_demo/server-signed/rootCA.pem ./distprov_demo/server-bundle/

./04_site_admin_package.sh \
  ./server.yml \
  grpc://server.example.com:8002 \
  ./distprov_demo/server-bundle

# Client site-1 admin
mkdir -p ./distprov_demo/site-1-bundle
cp ./distprov_demo/site-1-csr/site-1.key    ./distprov_demo/site-1-bundle/
cp ./distprov_demo/site-1-signed/site-1.crt ./distprov_demo/site-1-bundle/
cp ./distprov_demo/site-1-signed/rootCA.pem ./distprov_demo/site-1-bundle/

./04_site_admin_package.sh \
  ./site-1.yml \
  grpc://server.example.com:8002 \
  ./distprov_demo/site-1-bundle

# Client site-2 admin
mkdir -p ./distprov_demo/site-2-bundle
cp ./distprov_demo/site-2-csr/site-2.key    ./distprov_demo/site-2-bundle/
cp ./distprov_demo/site-2-signed/site-2.crt ./distprov_demo/site-2-bundle/
cp ./distprov_demo/site-2-signed/rootCA.pem ./distprov_demo/site-2-bundle/

./04_site_admin_package.sh \
  ./site-2.yml \
  grpc://server.example.com:8002 \
  ./distprov_demo/site-2-bundle
```

The startup kits are written into the standard NVFlare production directory
(`~/nvflare/poc/<project>/prod_00/` by default).

---

## Notes

- `openssl` is required by `04_site_admin_package.sh` to display and verify the
  `rootCA.pem` fingerprint before packaging.
- Root CA initialisation does **not** require an `org` in `site.yml`.
- `org` **is required** for every participant and must match `^[A-Za-z0-9_]+$`.
- To override a participant's role at signing time, pass the type as a fourth
  argument to `03_project_admin_sign.sh`:
  ```bash
  ./03_project_admin_sign.sh <csr_path> <ca_dir> <out_dir> org_admin
  ```
