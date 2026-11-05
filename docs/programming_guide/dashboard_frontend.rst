.. _dashboard_frontend:

Building the Dashboard Frontend
==============================

The Dashboard frontend source lives in ``dashboard/ui/``. It uses the open-source
MUI community components, React, and Next.js Pages Router. Its build exports static
HTML, CSS, and JavaScript for the existing Flask Dashboard at
``/nvflare-dashboard``. Dashboard users who install NVFlare need only the
:ref:`Dashboard Python dependencies <dashboard_api>`; Node.js is a contributor
build dependency and is not needed to serve the packaged UI.

Requirements
------------

Use Node 24 LTS and npm 11, as recorded by ``dashboard/ui/.nvmrc`` and
``dashboard/ui/package.json``. Use a supported Python environment with the
Dashboard extra installed from this checkout:

.. code-block:: shell

   python -m pip install -e '.[dashboard]'

The frontend uses ``package-lock.json`` and ``npm ci``. Installation lifecycle
scripts are disabled in ``.npmrc``. MUI Material, Data Grid, and Date Pickers are
the MIT-licensed community editions; a commercial MUI license key is not needed.

Build and package
-----------------

From the repository root, install the locked dependencies, build the export,
and publish the result to NVFlare's packaged static directory:

.. code-block:: shell

   python dashboard/build_ui.py --install
   python dashboard/build_ui.py --check

After dependencies are installed, ``python dashboard/build_ui.py`` rebuilds and
publishes without reinstalling them. The helper writes the frontend assets to
``nvflare/dashboard/application/static/nvflare-dashboard/`` and records the source
fingerprint. ``--check`` detects packaged output that is stale relative to the
source; it does not rebuild it. Run this check before creating a release package.

For frontend-only work, run the following from ``dashboard/ui/``:

.. code-block:: shell

   npm ci
   npm run typecheck
   npm run lint
   npm test -- --runInBand
   npm run test:notices
   npm run build

The direct npm build exports into ``dashboard/ui/out/``. It does not update the
Python package's static files. Next.js uses ``output: 'export'``; there is no
separate ``next export`` command. The repository helper performs the publishing
step, including the generated third-party notices.

Build the frontend and verify freshness before building an NVFlare wheel or
source distribution. Python packaging includes the published static assets;
it does not include ``node_modules`` or require a production Node.js server.

.. code-block:: shell

   python -m pip install build
   python -m build
   python dashboard/smoke_wheel.py dist/*.whl --sdist dist/*.tar.gz

The smoke check requires the Dashboard Python dependencies in the active
environment. It installs the wheel without downloading dependencies into a
temporary directory, then checks Flask pages, every bundled asset, and the API.
The source archive also includes the frontend source and build helper.

Run and develop with Flask
--------------------------

Use the existing Dashboard command to run the backend:

.. code-block:: shell

   nvflare dashboard --start --local

See :doc:`../user_guide/nvflare_cli/dashboard_command` for account setup, ports,
TLS, Docker, and deployment options. ``--local`` is intended for development
and testing.

For interactive frontend development, start the backend, then run this from
``dashboard/ui/`` using its actual origin:

.. code-block:: shell

   NVFLARE_DASHBOARD_BACKEND_URL=http://127.0.0.1:8443 npm run dev

Open ``http://127.0.0.1:3000/nvflare-dashboard``. The Next.js development server
proxies the prefixed API requests to Flask. This proxy is development-only;
the static production UI and Flask API share their origin and prefix. Flask
CORS changes are unnecessary. If the backend uses HTTPS, configure Node to trust
its certificate authority, for example with ``NODE_EXTRA_CA_CERTS``.

Validation
----------

The Jest tests cover frontend state, validation, and error behavior. Browser tests
exercise the production static export with actual MUI controls and mocked APIs:

.. code-block:: shell

   npm run build
   npx playwright install chromium
   npm run test:browser

Set ``NVFLARE_TEST_BROWSER_CHANNEL=chrome`` to use an installed Google Chrome.

To validate against a real Dashboard backend, activate a Python environment with
this checkout's ``dashboard`` extra so its ``python3`` is on ``PATH``, then run:

.. code-block:: shell

   npm run test:backend

The fixture uses a loopback-only Flask server and temporary SQLite database. It
registers participants before and after project freeze, approves users and sites
through the UI, rejects a direct role-escalation attempt, and verifies encrypted
client and console kits using the displayed PIN. A system ``zip`` executable is
required for these real-kit tests. The fixture does not use an existing Dashboard
database. The frontend tests complement the Python Dashboard authorization and
concurrency tests, rather than replacing them.

Dependency and license updates
------------------------------

Review the dependency set as a whole, including peer requirements, when updating
versions. Run the validation commands above and both ``npm audit`` and
``npm audit --omit=dev``. The baseline and known development-tool findings are
recorded in ``dashboard/ui/DEPENDENCIES.md``.

The static build generates ``THIRD_PARTY_NOTICES.txt`` from installed dependency
licenses without importing their code. Required packages missing supplied or
reviewed license text fail notice generation. Keep the notices with the exported
assets. Native compiler and image-processing binaries are build dependencies and
are not included in the static UI or Python distribution.
