.. _multi_study_guide:

Multi-Study Support
*******************

Overview
========

Studies provide logical separation within a shared-trust NVFlare deployment. Each study defines which
sites participate and which admins are members. Study-aware job and client-targeted operations
are scoped to the active study. The default study (``"default"``) is the fallback session context:
it uses the certificate-based role and scopes visibility to jobs in the default study.

If ``studies:`` is absent from ``project.yml``, the deployment starts single-tenant: only the
``default`` study can be used at login until studies are registered at runtime (see
:ref:`updating_studies`).

Configuring Studies in project.yml
==================================

Multi-study requires ``api_version: 4`` in your ``project.yml``. Studies are defined in a top-level
``studies`` section that maps study names to their site and admin configurations:

.. code-block:: yaml

    api_version: 4
    name: my_project

    participants:
      - name: server1
        type: server
        org: nvidia
        fed_learn_port: 8002
        admin_port: 8003
      - name: site-1
        type: client
        org: nvidia
      - name: site-2
        type: client
        org: nvidia
      - name: site-3
        type: client
        org: nvidia
      - name: admin@nvidia.com
        type: admin
        org: nvidia
        role: project_admin
      - name: lead@nvidia.com
        type: admin
        org: nvidia
        role: lead

    studies:
      cancer-research:
        site_orgs:
          nvidia: [site-1, site-2]
        admins:
          - admin@nvidia.com
          - lead@nvidia.com
      drug-discovery:
        site_orgs:
          nvidia: [site-2, site-3]
        admins:
          - admin@nvidia.com

Validation rules:

- Sites listed under ``site_orgs`` must reference existing client participants in the specified organization.
- Admins listed in a study must reference existing admin participants.
- Study names use lowercase alphanumeric characters plus hyphens or underscores, 1-63 characters, and must start and end with an alphanumeric character.
- ``"default"`` is reserved and cannot be used as a study name.
- Provisioning generates ``study_registry.json`` in the server's ``local/`` folder, which seeds the
  runtime registry on first server start (see :ref:`updating_studies`).

Study Membership and Roles
==========================

When a user logs in to a named study, the server requires the study to exist in its
registry. The user must either appear in that study's ``admins`` list or have a
validated certificate that authorizes the study. If neither source authorizes
membership, login is rejected. The registry remains authoritative for study
existence and site enrollment.

The user's role always comes from the certificate's ``unstructuredName`` field.
Study membership does not assign or override that role. For example, a ``lead``
certificate uses the ``lead`` role in every study the user can access.

The ``default`` study requires successful certificate authentication but no named-study
membership. Default sessions still only see default-study jobs.

.. _certificate_study_entitlements:

Certificate-Derived Study Membership
----------------------------------------

An admin certificate can name up to 64 unique studies using one URI SAN per study:

.. code-block:: text

    https://nvidia.com/nvflare/v1/project/demo/study/cancer-research

The project label is one UTF-8 percent-encoded path segment, leaving RFC 3986
unreserved characters unescaped. It is informational: FLARE does not compare it
with the running project name. Any federation trusting the issuing CA can honor
the certificate's matching study names.

Names must follow the study naming rules above; ``default`` and wildcards are
not allowed. Unrelated URI SANs are ignored, but malformed or unsupported URIs
under ``https://nvidia.com/nvflare/`` reject all certificate logins, including
``default``. Without study SANs, existing registry-based access is unchanged.

See :ref:`step_ca_study_entitlements` for mapping issuer-validated claims to
these SANs and a downloadable template.

Membership is visible in the certificate and server-signed session token,
which preserves study discovery after session reconstruction.

Removing IdP entitlements or registry membership does not revoke existing
certificate-derived access; it lasts until certificate expiry. There is no
local denylist. Do not reuse a study name for a different study.

Certificate-authorized administrators can make role-permitted registry changes,
including membership grants that persist until explicitly removed, independently
of certificate expiry or IdP entitlement removal.

Using Studies
=============

FLARE Console
-------------

Pass the ``--study`` flag when launching the admin console:

.. code-block:: bash

    fl_admin.sh --study cancer-research

If ``--study`` is omitted, the session uses the ``default`` study.

FLARE API
---------

Specify the ``study`` parameter when creating a secure session:

.. code-block:: python

    new_secure_session("admin@nvidia.com", "/path/to/admin", study="cancer-research")

ProdEnv (Recipes)
-----------------

Pass the ``study`` parameter to :class:`ProdEnv`:

.. code-block:: python

    ProdEnv(startup_kit_location="/path/to/admin", study="cancer-research")

PocEnv (Recipes)
----------------

Pass the ``study`` parameter to :class:`PocEnv` when your POC deployment is provisioned from a custom
``project.yml`` that defines ``studies:``:

.. code-block:: python

    PocEnv(num_clients=2, project_conf_path="/path/to/project.yml", study="cancer-research")

If the POC deployment uses the default generated project with no ``studies:``, only the ``default`` study is valid.

Study-Scoped Behavior
=====================

When a session is bound to a study, the following scoping rules apply to study-aware operations:

- ``list_jobs`` shows only jobs belonging to the active study.
- ``get_job_meta`` and ``clone_job`` return "not found" for jobs in other studies.
- ``check_status client`` shows only sites enrolled in the active study.
- ``submit_job`` tags the job with the active study; ``@ALL`` is narrowed to study-enrolled sites.
- ``deploy_map`` validation rejects sites not enrolled in the study.
- Jobs without a ``study`` field (legacy) are normalized to ``"default"``.

Server-only/global operations continue to use the certificate-based role.

When to Use Multi-Study vs. Separate Deployments
=================================================

Multi-study is suited for scenarios where organizations share trust (same PKI, same server) but
want logical isolation of experiments. For stronger isolation — separate PKI, separate blast
radius — use separate NVFlare deployments.

.. _updating_studies:

Updating Studies
================

The ``nvflare study`` command family manages the server-side study registry at runtime, without
reprovisioning or restarting the server: ``register`` creates or merges a study, ``add-site`` /
``remove-site`` change site enrollment, ``add-user`` / ``remove-user`` manage study-user
membership, and ``remove`` deletes a study. See :ref:`study_command` for the full command
reference.

These commands change only the server's view of a study — enrollment, login membership, and job
scoping. They do not configure the participating sites: sites must already be provisioned and
connected to be enrolled (adding a new site or admin identity to the deployment still requires
provisioning, since certificates must be issued), and per-site runtime resources for a study —
data mounts, job images, and related settings in each site's ``local/study_runtime.yaml`` — are
managed by each site's operator, not by these commands. Until a site defines runtime resources for
a study, jobs for that study run at that site without any study-specific data mounts or settings.

Runtime study mutations are persisted to ``study_registry.json`` in the server workspace root, not
in the ``local/`` folder. The provisioned copy under ``local/`` is only a first-start seed: once a
runtime mutation has been persisted, the workspace-root copy is authoritative, shadows the seed,
and survives server restarts. Because writes never target ``local/``, runtime study management also
works in deployments where ``local/`` is mounted read-only, such as Kubernetes deployments that
stage ``local/`` as a ConfigMap.

Study Environment in Process Mode
=================================

Client and server process launchers read ``local/study_runtime.yaml`` from their own workspace
before each job launch. Use ``env`` for literal settings and ``secret_env`` to map variables from
the parent service's environment into the job's environment, following the same pattern as Docker:

.. code-block:: yaml

   format_version: 2
   studies:
     lung-cancer:
       env:
         DB_HOST: db.internal
         TRAINING_DATA: /srv/studies/lung-cancer/training
         DB_CA_FILE: /srv/studies/lung-cancer/secrets/ca.crt
       secret_env:
         DB_PASSWORD:
           source: LUNG_DB_PASSWORD
     default:
       env:
         TRAINING_DATA: /srv/studies/default/training

Set ``LUNG_DB_PASSWORD`` in the parent service's environment before starting it. A job in
``lung-cancer`` receives its value as ``DB_PASSWORD``; the value itself stays out of the YAML file.
The optional ``key`` field is accepted for compatibility with Docker configuration and is ignored.
A missing source variable fails launch. An empty source value is allowed.

Study values override inherited values in the child only. ``env`` accepts literal scalar values,
including empty strings, with no shell expansion. Names must be valid POSIX environment variable
names; launcher-owned names such as ``PYTHONPATH`` cannot be configured or used as secret sources.
A destination cannot appear in both ``env`` and ``secret_env``.

Applications must read path variables such as ``TRAINING_DATA`` and ``DB_CA_FILE`` to use them.
Files must already exist on the host. Process mode supports only ``env`` and ``secret_env``;
container and mount settings are rejected, and hard-coded container paths are not translated.

Process mode inherits the complete parent environment, including secret source variables and
variables for other studies. This feature provides configuration convenience, not isolation between
studies. Use a suitably configured Docker deployment when jobs require isolation.

If the file or selected study entry is absent, the job uses its usual environment. Jobs without a
study use ``default``. Invalid configuration anywhere in the file fails launch. Process mode always
ignores the legacy ``study_data.yaml``, including when both files exist.

Replace the file atomically to update subsequent launches. Running jobs retain their environment;
changing secret source values normally requires restarting the parent service.

.. note::

   On upgrading to process-mode study environment support, an existing ``study_runtime.yaml``
   becomes active. Review it first: environment settings will start applying, and unsupported
   container or mount settings will fail launch.
