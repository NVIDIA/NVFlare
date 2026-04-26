.. _package_command:

###############
Package Command
###############

``nvflare package`` assembles a startup kit from a signed zip returned by the
Project Admin and the requester's local private key.

The public distributed provisioning form is:

.. code-block:: none

   usage: nvflare package [-h] [-e ENDPOINT] [-w WORKSPACE] [-p PROJECT_FILE]
                          [--request-dir REQUEST_DIR] [--admin-port ADMIN_PORT]
                          [--force] [--schema]
                          [input]

The ``input`` positional argument is the ``*.signed.zip`` file produced by
``nvflare cert approve``.

*******************
Basic Package Flow
*******************

For a site request created in ``./site-3``:

.. code-block:: shell

   nvflare package site-3.signed.zip -e grpc://server1:8002 --request-dir ./site-3

If the signed zip is already inside or next to the request folder, the
``--request-dir`` option is often not needed:

.. code-block:: shell

   nvflare package site-3/site-3.signed.zip -e grpc://server1:8002

The command validates that:

- the signed zip contains ``signed.json``, ``site.yaml``, one signed
  certificate, and ``rootCA.pem``;
- the signed zip does not contain private keys;
- the local private key matches the signed certificate;
- the certificate chains to ``rootCA.pem``;
- request metadata and signed metadata match when local request state is found.

The output goes under:

.. code-block:: text

   <workspace>/<project-name>/prod_NN/<identity>/

For example:

.. code-block:: text

   workspace/example_project/prod_00/site-3/

**************************
Package a User Startup Kit
**************************

For a lead user:

.. code-block:: shell

   nvflare cert request user lead lead@nvidia.com --org nvidia --project example_project
   nvflare cert approve lead@nvidia.com.request.zip --ca-dir ./ca
   nvflare package lead@nvidia.com.signed.zip -e grpc://server1:8002 --request-dir ./lead@nvidia.com

The generated startup kit contains:

.. code-block:: text

   startup/fl_admin.sh

Run it with:

.. code-block:: shell

   cd workspace/example_project/prod_00/lead@nvidia.com
   ./startup/fl_admin.sh

**************************
Use Project Builders
**************************

Use ``--project-file`` when packaging needs custom builders or project-level
package configuration:

.. code-block:: shell

   nvflare package site-3.signed.zip \
       -e grpc://server1:8002 \
       --project-file ./project.yml \
       --request-dir ./site-3

In signed-zip mode, the signed zip remains the source of truth for the
participant identity. The project file supplies builders and non-identity
package configuration. Only the signed participant is built, even if
``project.yml`` lists other sites or users.

If the signed participant is listed in the project file, identity fields must
match the signed zip. If they do not match, packaging fails.

If the signed participant is not listed in the project file, packaging continues
with the signed identity and project builders, and prints a warning.

****************
Main Arguments
****************

- ``input``: approved ``*.signed.zip`` returned by ``nvflare cert approve``.
- ``-e, --endpoint``: server endpoint URI. Supported schemes are ``grpc://``,
  ``tcp://``, and ``http://``. Required.
- ``-w, --workspace``: workspace root directory. Default: ``workspace``.
- ``-p, --project-file``: project YAML for custom builders or package
  configuration. In signed-zip mode, only the signed participant is built.
- ``--request-dir``: local request directory containing the private key. Use it
  when the signed zip is not next to the request folder.
- ``--admin-port``: server admin port. Default: same as the service port.
- ``--force``: allow packaging when the participant name already appears in the
  latest ``prod_NN`` directory. A new ``prod_NN`` is created alongside.
- ``--schema``: print JSON schema for this command.

*********************
JSON Output and Help
*********************

Use ``--schema`` for machine-readable command discovery:

.. code-block:: shell

   nvflare package --schema

The top-level CLI also supports JSON output mode:

.. code-block:: shell

   nvflare package site-3.signed.zip -e grpc://server1:8002 --format json

For the end-to-end distributed provisioning workflow, see
:ref:`distributed_provisioning`.
