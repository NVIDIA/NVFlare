.. _multi_study_guide:

Multi-Study Support
*******************

Overview
========

Studies provide multi-tenant isolation within a single NVFlare deployment. Each study defines which
sites participate and what role each admin user has. Study-aware job and client-targeted operations
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
        role: org_admin

    studies:
      cancer-research:
        sites: [site-1, site-2]
        admins:
          admin@nvidia.com: project_admin
          lead@nvidia.com: lead
      drug-discovery:
        sites: [site-2, site-3]
        admins:
          admin@nvidia.com: project_admin

Validation rules:

- Sites listed in a study must reference existing client participants.
- Admins listed in a study must reference existing admin participants.
- Study names use lowercase alphanumeric characters plus hyphens or underscores, 1-63 characters, and must start and end with an alphanumeric character.
- ``"default"`` is reserved and cannot be used as a study name.
- Provisioning generates ``study_registry.json`` in the server's ``local/`` folder, which seeds the
  runtime registry on first server start (see :ref:`updating_studies`).

Per-Study Role Resolution
=========================

When a user logs in to a named study, their role is looked up from that study's ``admins`` mapping
instead of using the certificate-based role. This resolved role is used for study-scoped
authorization decisions during that session. If the user is not listed in the study's ``admins``
mapping, login is rejected.

.. note::

   If a study maps a user to ``project_admin``, that means the user has full authority for
   study-scoped operations in that study. It does **not** make the user a deployment-wide project
   admin for server-only or other global operations. Those continue to use the certificate-based
   role from the admin participant definition in ``project.yml``.

The ``default`` study always uses the certificate-based role. In a multi-study deployment, default
sessions still only see default-study jobs.

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
