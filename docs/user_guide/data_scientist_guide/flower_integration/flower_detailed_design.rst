.. _flower_http_design:

***************
Detailed Design
***************

The integration uses Flower's unary protobuf-over-HTTP Fleet API and requires
``flwr>=1.43``.
The Recipe, TIE Controller/Executor, app deployment, and analytics relay retain
the existing architecture. The local transport adapters and Flower process
launchers use HTTP::

    Flower SuperNode
      <-> loopback HTTP endpoint in NVFlare client
      <-> NVFlare ReliableMessage
      <-> NVFlare server connector
      <-> loopback HTTP Flower SuperLink

Each job has its own SuperLink, and each participating site has its own SuperNode
and local Fleet endpoint. Flower ports bind only to ``127.0.0.1``. HTTP on these
local connections does not replace NVFlare's transport or TLS between sites.
As with the previous integration, the host and its local processes are trusted.

Protocol boundary
=================

The bridge carries the HTTP method, exact relative route, required headers, and
binary protobuf body in a Shareable. Responses preserve the status, content type,
retry metadata, and binary body, including Flower error responses. Models and
Flower message contents are not decoded or reserialized by the bridge.

Both connectors validate the request. Only POST is accepted, and only these
``/v1/fleet/`` routes are forwarded:

* ``register-node``, ``activate-node``, ``deactivate-node``, ``unregister-node``
* ``send-node-heartbeat``, ``pull-messages``, ``push-messages``
* ``get-run``, ``get-fab``, ``push-object``, ``pull-object``
* ``confirm-message-received``

Absolute URLs, query strings, redirects, encoded request bodies, and arbitrary
routes are not supported. The upstream destination is fixed by the connector.
Control and Runtime APIs share Flower's listener but cannot be reached through
this bridge. The server runs ``flwr run/list/stop --format json`` locally and saves the
ServerApp log with ``flwr log --show`` before stopping SuperLink. This is a
best-effort shutdown snapshot in ``server_app_log.txt``, not a live log. It can
be absent if SuperLink crashes before the snapshot.

The bridge preserves Flower's public-key, timestamp, signature, and runtime-version
headers. A key is associated with the registering authenticated NVFlare site for
the lifetime of that job. SuperLink still verifies the signature and the node
identity inside the protobuf request. NVFlare job authorization is unchanged.

Limits, concurrency, and failure
================================

``max_message_bytes`` defaults to 512 MiB per request or response and is passed by
FlowerRecipe to both connectors. Flower transfers complete object protobufs;
this bridge buffers them in memory. Size the host for simultaneous transfers
and training. The local server admits at most 32 connections, with four bulk
request workers and separate capacity for heartbeat and metadata requests.
Excess requests receive a busy response or connection close.

ReliableMessage owns retransmission and transaction deduplication between sites.
The HTTP forwarding layer sends each accepted operation once; it does not follow
redirects or add retries to SuperLink. Flower retains its own HTTP error/retry
policy. An uncertain state-changing NVFlare transaction fails the client workflow.
Transient heartbeat communication failures return HTTP 503 without marking the
client failed, allowing Flower to send subsequent heartbeats. An explicit server
shutdown still ends the client cleanly.

``per_msg_timeout`` and ``tx_timeout`` default to 2 and 25 seconds. Larger values
are capped, with a warning showing the effective limits, to fit Flower's 30-second HTTP timeout. Heartbeats
use caps of 0.5 and 3.5 seconds to fit Flower's 5-second heartbeat timeout.
Deployments must be able to transfer a payload within these limits. Shutdown
cancels forwarding, closes local sockets, joins workers, and terminates owned
Flower processes, including detached SuperExec descendants. An independent
process guard retains descendant identities while the job is alive. Closing its
control pipe, including when the job process is killed, triggers cleanup. The
control pipe is not inherited by Flower workers. Descendant monitoring uses one
process-table snapshot per scan.

Compatibility and migration
===========================

Install ``flwr>=1.43`` with Python 3.11 or newer on every site. The Recipe and deployed applets check this
minimum version before starting Flower. The integration was validated with
Flower 1.43.0. Flower itself requires matching minor versions for peer runtime
metadata.

The legacy ``GrpcClientConnector``, ``GrpcServerConnector``, gRPC helper modules,
and their demo mocks are removed. Remove ``int_client_grpc_options`` from old
FlowerController configurations. Regenerate exported jobs with the current
FlowerRecipe. There is no automatic legacy transport fallback.

Runtime dependency installation remains opt-in through
``allow_runtime_dependency_installation``. The launcher explicitly disables the
new SuperLink default. The server and each client have their own job-scoped
``FLWR_HOME``; the caller's global Flower configuration is not used. With runtime
installation enabled, Flower manages app environments and package-index access
must be available. NVFlare must be importable from any app that uses its analytics
API.

Metrics continue through the separate analytics bootstrap and MetricRelay path.
They are not embedded in Fleet request headers or routed through the Control API.


Known limitations
=================

* **Unexpected process-inspection errors:** orphan discovery reads process
  environments. Normal process-exit and access-denied races are handled, but an
  unexpected OS error can interrupt a scan and leave discovery incomplete.
  Cleanup reports failure rather than claiming that all workers exited. This
  OS-specific case has not been reproduced; targeted error handling requires
  the failing operation and error code. Operators should check for surviving
  job-owned workers when cleanup reports failure.
* **Flower key lifecycle:** ownership is enforced from key to authenticated
  NVFlare site. There is no per-site key limit, and a successful registration's
  ownership is retained for the job lifetime, including after unregistering.
  This permits a restarted SuperNode to register a new key and prevents
  reassignment of an old key while requests might remain in flight. It does not
  enforce one Flower identity per site, and repeated registrations can accumulate
  ownership entries. Per-site quotas and safe key replacement/release require a
  defined reconnect and unregister policy. SuperLink still verifies signatures.
