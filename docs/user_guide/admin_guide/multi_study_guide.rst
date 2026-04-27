.. _multi_study_guide:

Multi-Study Support
*******************

Overview
========

Studies provide multi-tenant isolation within a single NVFlare deployment. Each study defines which
sites participate and what role each admin user has. Study-aware job and client-targeted operations
are scoped to the active study. The default study (``"default"``) is the fallback session context:
it uses the certificate-based role and scopes visibility to jobs in the default study.

If ``studies:`` is absent from ``project.yml``, the deployment is single-tenant and only the
``default`` study can be used at login.

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
- Provisioning generates ``study_registry.json`` in the server's ``local/`` folder.

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

Updating Studies
================

Study site enrollment and user membership can be updated at runtime using the ``nvflare study``
command family without reprovisioning or restarting the server. See :ref:`study_command` for the
full command reference.

Changing the core study definition (adding new studies or altering the base provisioned
configuration) requires reprovisioning and a server restart.
