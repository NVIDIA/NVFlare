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
./01_project_admin_init_ca.sh fed-project NvidiaOrg ./distprov_demo/ca
```

Outputs: `./distprov_demo/ca/rootCA.pem` and the CA private key.

---

## Step 2 — Site Admin: create request zip + private key

Each site admin runs this on their own machine, one call per site.

```bash
# Server site admin
./02_site_admin_request.sh fed-project ./server.yml ./distprov_demo/server-request

# Client site-1 admin
./02_site_admin_request.sh fed-project ./site-1.yml ./distprov_demo/site-1-request

# Client site-2 admin
./02_site_admin_request.sh fed-project ./site-2.yml ./distprov_demo/site-2-request
```

Each `<request_dir>` receives `<name>.key`, `<name>.csr`, `site.yaml`,
`request.json`, and `<name>.request.zip`.
The site admin **keeps the `.key` private** and sends only the `.request.zip`
to the Project Admin.

---

## Step 3 — Project Admin: approve each request zip

The Project Admin receives the `.request.zip` files and approves them one at a time.

```bash
./03_project_admin_approve.sh \
  ./distprov_demo/server-request/server.request.zip \
  ./distprov_demo/ca \
  ./distprov_demo/server.signed.zip

./03_project_admin_approve.sh \
  ./distprov_demo/site-1-request/site-1.request.zip \
  ./distprov_demo/ca \
  ./distprov_demo/site-1.signed.zip

./03_project_admin_approve.sh \
  ./distprov_demo/site-2-request/site-2.request.zip \
  ./distprov_demo/ca \
  ./distprov_demo/site-2.signed.zip
```

Each approval command creates a `<name>.signed.zip`. Return only this signed zip
to the requesting site admin.

The approval command prints `rootca_fingerprint_sha256`. Share this value with
each site admin through a **trusted out-of-band channel** (e.g. a secure Slack
DM or signed email) so they can verify it before packaging.

---

## Step 4 — Site Admin: package startup kit

Each site admin runs `04_site_admin_package.sh` with the signed zip returned by
the Project Admin and the original local request directory from Step 2. The
package command prints the `rootCA.pem` fingerprint from the signed zip and asks
for confirmation before building the kit.

```bash
# Server site admin
./04_site_admin_package.sh \
  ./distprov_demo/server.signed.zip \
  grpc://server.example.com:8002 \
  ./distprov_demo/server-request

# Client site-1 admin
./04_site_admin_package.sh \
  ./distprov_demo/site-1.signed.zip \
  grpc://server.example.com:8002 \
  ./distprov_demo/site-1-request

# Client site-2 admin
./04_site_admin_package.sh \
  ./distprov_demo/site-2.signed.zip \
  grpc://server.example.com:8002 \
  ./distprov_demo/site-2-request
```

The startup kits are written under the package workspace:
`workspace/<project>/prod_NN/<name>/` by default. Since signed-zip packaging
builds one participant at a time, repeated package commands can create separate
`prod_NN` directories.

---

## Notes

- `python3` is required by the helper scripts to read `site.yml`.
- `org` is required for every participant and must match `^[A-Za-z0-9_]+$`.
- The package helper accepts an optional fourth `project_file` argument for
  custom builders or package configuration. The signed zip remains the source of
  truth for participant identity. If you pass a single-site YAML as the project
  file, include a matching `project` or `project_name`; otherwise omit it.
- Project Admin approval signs the identity in the request zip. To change the
  requested type or role, reject the request and ask the requester to generate a
  corrected request zip.
