# NVFlare Dashboard frontend

This directory contains the open-source MUI Dashboard frontend. It uses Next.js Pages Router and exports static HTML, CSS, and JavaScript for the Flask Dashboard at `/nvflare-dashboard`. Running the installed NVFlare Dashboard does not require Node.js.

The frontend source follows NVFlare's Apache-2.0 license. Its MUI Material, Data Grid, and Date Picker packages are the MIT-licensed community editions and require no commercial MUI license key. Third-party notices are generated from production browser source maps, with shared license text and preserved package attribution. Maps are removed from the export before publication.

## Prerequisites

- Node 24 LTS and npm 11; `.nvmrc` and `package.json` record the supported major versions.
- A supported Python environment with the NVFlare checkout's Dashboard dependencies: from the repository root, run `python -m pip install -e '.[dashboard]'`.
- A system `zip` command for the optional real-backend encrypted-kit tests.

`package-lock.json` is the authoritative frontend dependency lockfile. Package installation lifecycle scripts are disabled in `.npmrc`. Use `npm ci` for a clean install; update dependencies and the lockfile deliberately rather than independently installing arbitrary latest versions.

## Build and publish the packaged UI

Run from the NVFlare repository root:

```sh
python dashboard/build_ui.py --install
python dashboard/build_ui.py --check
```

The first command installs the locked npm dependencies, builds the static export, generates third-party notices, and publishes it to `nvflare/dashboard/application/static/nvflare-dashboard/`. Subsequent source edits can be built and published with `python dashboard/build_ui.py` after dependencies are installed. The check command detects packaged assets that no longer match the frontend source; it does not rebuild them.

For frontend-only iteration, run from this directory:

```sh
npm ci
npm run typecheck
npm run lint
npm test -- --runInBand
npm run test:notices
npm run build
```

`npm run build` writes `out/`, including `THIRD_PARTY_NOTICES.txt`. Next.js performs the static export through `output: 'export'`; there is no separate `next export` step. A direct npm build does not publish its output into the Python package; use the repository build helper for that step. Generated local output and installed npm packages are ignored by Git.

Before building a wheel or source distribution, build/publish the frontend and run the freshness check. Python packaging includes the published static UI, not `node_modules` or a production Node server. Use the existing NVFlare Dashboard CLI to run the backend after the package or editable installation is ready:

```sh
nvflare dashboard --start --local
```

The CLI's account setup, ports, TLS configuration, and production deployment options are documented in the Dashboard command guide.

## Develop with the backend

Start the Dashboard backend in a Python environment containing the current checkout, then run this directory's development server with the backend's origin:

```sh
NVFLARE_DASHBOARD_BACKEND_URL=http://127.0.0.1:8443 npm run dev
```

Open `http://127.0.0.1:3000/nvflare-dashboard`. Replace the example origin with the backend's actual protocol and port. If using a development HTTPS certificate, configure Node to trust its CA, for example through `NODE_EXTRA_CA_CERTS`.

The development server proxies `/nvflare-dashboard/api/v1/` to that backend. The proxy is absent from the production static export, where the UI and Flask API share the same origin and prefix. No Flask CORS changes are needed for this development loop.

## Browser and backend tests

The Jest tests cover registration, profile/site edits, reviewed bulk approvals, rejected drafts, project freeze, and download error handling. To exercise the exported production assets with actual MUI controls and mocked API responses:

```sh
npm run build
npx playwright install chromium
npm run test:browser
```

Use `NVFLARE_TEST_BROWSER_CHANNEL=chrome` to select an installed Google Chrome instead of Playwright's Chromium.

The optional backend browser suite uses the current checkout's real Flask Dashboard, an isolated temporary SQLite database, and encrypted startup kits. Activate a Python environment with `nvflare[dashboard]` installed from this checkout so that `python3` on `PATH` uses that environment, then run:

```sh
npm run test:backend
```

The fixture serves only on loopback, uses its own database and known test credentials, and cleans up on exit. It registers participants in frozen and unfrozen projects, approves users/sites through the UI, rejects a direct self-promotion attempt, and opens downloaded ZIP files with the displayed PIN to check encryption and certificate attributes. It does not use an existing Dashboard database. The frontend suite complements the Python Dashboard authorization and concurrency tests in `tests/unit_test/dashboard/` and `tests/integration_test/fast/`.

## Behavior contract

- Participants can request Member, Lead, or Org Admin. Pending Org Admin applicants can register their own pending client sites.
- User approval includes the role and organization displayed for review. A changed registration must be reloaded and reviewed before retrying approval.
- Role and organization fields already set by participants are locked. Approved legacy profiles with blank identity fields require administrator repair.
- Project Admins can edit user profiles and approve, deny, or delete users and sites. Site ownership and authorization remain backend-enforced.
- Rejected edits preserve entered values and display API guidance. Authentication failures clear the session; policy errors do not claim session expiration.
- Kit downloads require a frozen project and the applicable approvals. The PIN is displayed after a successful download. Participant forms do not send operator-controlled provisioning properties.

See `DEPENDENCIES.md` for version selection, update checks, and the scope of the notice inventory. The contributor guide in `docs/programming_guide/dashboard_frontend.rst` describes the repository integration.
