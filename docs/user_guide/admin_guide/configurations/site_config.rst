.. _site_config:

Site Configuration Metadata
===========================

A client site can advertise local metadata to the server during registration
and control a small set of site-level policies through its
``local/resources.json``. The metadata is delivered automatically — no extra
communication channel is required.

This page covers two related controls:

- :ref:`site_config_metadata` — advertise custom site metadata (labels,
  capabilities, resource hints, …) to the server.
- :ref:`allow_log_streaming` — opt out of live log streaming.

.. _site_config_metadata:

Advertising Site Metadata
-------------------------

What it is
~~~~~~~~~~

When a client registers with the server, FLARE forwards a curated dict of
site metadata (the *site_config*) on the existing registration message. The
server validates and stores it on the registered ``Client`` object, where it
becomes available to controllers and is included in job metadata.

Use cases include:

- Tagging sites with environment / region labels (``"region": "us-east"``).
- Advertising site capabilities (``"capabilities": ["he", "psi"]``).
- Reporting hardware hints to the controller (``"site_resources":
  {"memory_gb": 128, "gpu_count": 4}``).

How a site contributes metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add custom top-level keys to the site's ``local/resources.json``:

.. code-block:: json

    {
      "format_version": 2,
      "client": { "retry_timeout": 30 },
      "components": [ ... ],

      "labels": { "region": "us-east", "tier": "research" },
      "capabilities": ["he", "psi"],
      "site_resources": { "memory_gb": 128, "cpu_cores": 32 }
    }

FLARE automatically projects the file into a site_config: it deep-copies all
top-level keys and drops the structural / local-only ones that wouldn't make
sense on another machine. The following keys are always **excluded**:

- ``format_version``
- ``client``
- ``servers``
- ``components``
- ``handlers``
- ``snapshot_persistor``
- ``admin``
- ``relay_config``
- ``overseer_agent``

If a site explicitly sets ``client.site_config`` in its config, that value is
respected as-is (the auto-projection is skipped).

The same projection runs in both POC / production startup
(``FLClientStarterConfiger``) and the simulator
(``SimulatorDeployer``), so the two modes produce identical payloads.

Server-side validation
~~~~~~~~~~~~~~~~~~~~~~

The server applies three checks when it receives a site_config during
registration:

1. The value must be a ``dict`` (otherwise dropped with a warning).
2. The value must be JSON-serializable.
3. The serialized payload must not exceed 64 KB.

Site_config is also dropped from non-regular clients (relays, etc.). Failures
are soft-dropped — registration succeeds, just without the metadata.

Accessing site_config server-side
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inside server-side code (workflows, widgets), look up a registered client and
read its metadata:

.. code-block:: python

    client = engine.get_client_from_name(client_name)
    site_config = client.get_site_config() or {}
    region = site_config.get("labels", {}).get("region")

The same dict appears in the client's serialized form
(``Client.to_dict()``) and is therefore visible in job metadata.

.. note::

   Anything a site adds to ``site_config`` is observable to the server and any
   controller that inspects job metadata. **Do not place secrets here.**

.. _allow_log_streaming:

Controlling Live Log Streaming
------------------------------

A client may not want its job logs streamed to the server in real time. The
``allow_log_streaming`` boolean in ``resources.json`` is the site-level kill
switch for live log streaming.

The default is ``true`` — streaming is enabled. To disable streaming at a
site, opt out by setting:

.. code-block:: json

    {
      "format_version": 2,
      "client": { ... },
      "components": [ ... ],

      "allow_log_streaming": false
    }

When ``allow_log_streaming`` is set to ``false``:

- :class:`~nvflare.app_common.logging.job_log_streamer.JobLogStreamer` logs a
  warning at job start and does nothing — no live stream is opened.
- :class:`~nvflare.app_common.logging.system_log_streamer.SystemLogStreamer`
  strips any pre-declared ``JobLogStreamer`` from the deployed job
  configuration, skips its own automatic injection, and skips the
  post-completion error-log upload.
- The server-side
  :class:`~nvflare.app_common.logging.job_log_receiver.JobLogReceiver`
  logs an error (once per ``(client, job_id)``) if a chunk somehow arrives
  from a site that has explicitly disabled streaming. The error message
  identifies the offending client and job.

Because the value is forwarded as part of the site_config (it is not on the
exclusion list), the server can independently verify each client's policy at
chunk-receive time.

Default behavior
~~~~~~~~~~~~~~~~

If the field is missing from ``resources.json``, FLARE treats the site as
allowing streaming. Likewise, if ``resources.json`` cannot be read, or no
``Client`` is registered yet for the sender, the receiver falls back to
allow rather than deny — an explicit ``false`` is the only thing that
disables the stream.
