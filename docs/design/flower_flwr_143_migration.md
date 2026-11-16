# Flower 1.43 migration for the NVFlare Flower integration

Status: implemented on branch `flower-143-migration`. Section 12 lists what the spike and the implementation changed relative to this text.
Scope: `nvflare/app_opt/flower/`, the hello-flower example, Flower docs, Flower tests, and the CI Flower install.

## 1. Summary

The NVFlare Flower integration does not work with Flower 1.36 or later. The examples pin `flwr>=1.26,<1.40` as a workaround, but from 1.36 the SuperLink rejects the `--serverappio-api-address` flag that NVFlare passes, so the working range is effectively `<1.36`. The breakage is in the **runner** layer (how NVFlare launches Flower's SuperLink and SuperNode and carries Flower messages), not in user Flower apps, which are largely portable across these versions.

Flower 1.43 removes the gRPC-adapter transport NVFlare uses and serves the Fleet API as protobuf over HTTP. The migration replaces NVFlare's gRPC-adapter bridge with an HTTP tunnel over FLARE ReliableMessage. The tunnel keeps today's message pattern, one ReliableMessage transaction per Fleet RPC, so no change in call count is expected (section 4, "Call pattern").

The migration goes directly to `flwr>=1.43`. There is no interim pin change.

## 2. Version compatibility

Each row says how the version was checked. "Run" means an end-to-end run of `examples/hello-world/hello-flower` (`flwr-pt`, simulator, 2 clients) with the matching `flwr` installed. "Wheel" means the Flower wheel was inspected without running it. "CLI" means the `--help` output of the installed superlink was checked.

| Flower | Superlink launch with NVFlare's flags | Evidence | Result |
|---|---|---|---|
| 1.26.0 | Flags accepted | CLI | Not run. Expected to work, same family as 1.27. |
| 1.27.0 | Launches; `flwr run` submits; clients download CIFAR-10 and train | Run | **Works through training round 1 on both clients.** Round 2 started; the run then ended with no error logged and no final result. Completion is not verified. |
| 1.28–1.32 | `--serverappio-api-address` defined in the superlink entry point (`flwr/server/app.py`) | Wheel (text match) | Not run. Likely works; not verified. |
| 1.33 | `--serverappio-api-address` accepted by superlink CLI | Wheel (source read) | Not run. Likely works; not verified. |
| 1.34–1.35 | Flag text present in superlink CLI | Wheel (text match) | Not run. Likely works; not verified. |
| 1.36–1.38 | `--serverappio-api-address` is a stub that raises "no longer supported; use `--host` and `--port`" | Wheel (source read) | **Broken** (same failure as 1.39). Not run. |
| 1.39.0 | Superlink exits at argument parsing | Run | **Broken.** `flower-superlink: error: argument --serverappio-api-address: this option is no longer supported`. NVFlare then reports "cannot connect to server after 10.0 seconds". |
| 1.40–1.42 | Superlink rejects `--serverappio-api-address` | CLI | **Broken.** `grpcadapter` module still present. Not run. |
| 1.43.0 | Superlink rejects `--serverappio-api-address`; SuperNode rejects `--clientappio-api-address` and has no `--grpc-adapter`; `flwr.proto.grpcadapter_pb2*` removed | CLI + import | **Broken.** Not runnable without the migration. |

Install-time findings:

- `pip install nvflare flwr==1.39.0` does not resolve: NVFlare requires `uv>=0.13.6,<0.14.0` (`setup.cfg:23`) and Flower 1.39 requires `uv<0.12`.
- `pip install -e . flwr==1.43.0` resolves (pip dry run, base install; dev extras not checked). Flower 1.43 caps `protobuf<7`, so adding it moves protobuf from 7.36 (NVFlare alone) to 6.33. NVFlare's generated code is 4.25.1 and 5.29.0 gencode, which works with either runtime.

The applet's install hint at `applet.py:80` (`pip install 'flwr>=1.26'`) resolves to 1.43 today, so it sends new users to a version the current code cannot run. The migration fixes this by raising the minimum to 1.43.

## 3. Current architecture (as-is)

Message path in the gRPC-adapter design (1.26 to 1.35):

```
FLARE client host                                   FLARE server host
-----------------                                   -----------------
flower-supernode --grpc-adapter                     flower-superlink --fleet-api-type grpc-adapter
  --superlink <superlink_addr>                        --serverappio-api-address <serverapp_api_addr>
      |                                               --fleet-api-address <fleet_api_addr>
      | gRPC SendReceive(MessageContainer)            --control-api-address <exec_api_addr>
      v                                                     ^
GrpcClientConnector                                         | gRPC SendReceive(MessageContainer)
  (GrpcAdapterServicer on superlink_addr)                   |
      |                                               GrpcServerConnector
      |  FLARE ReliableMessage (Shareable)              (internal GrpcClient -> fleet_api_addr)
      +-------------------------------------------------->  ^
                                                            |
                                                      flwr run . nvflare  (submits the ServerApp
                                                      through the control API at exec_api_addr)
```

Every Fleet RPC the SuperNode makes, including per-object `PushObject` and `PullObject`, is wrapped in one `SendReceive` (`flwr/client/grpc_rere_client/grpc_adapter.py:81-100, :183` in Flower 1.27) and therefore becomes one FLARE ReliableMessage transaction.

Key code, with file and line:

- `nvflare/app_opt/flower/applet.py`
  - `:37-42` executable names, `MIN_FLWR_VERSION_FOR_RUNTIME_DEPS`.
  - `:80` install hint `pip install 'flwr>=1.26'`.
  - `:189-205` `FlowerClientApplet.get_command`: builds `flower-supernode --insecure --grpc-adapter --superlink ... --clientappio-api-address ...`.
  - `:315-317` reads the three server addresses from the app context.
  - `:372-381` `FlowerServerApplet.start`: builds `flower-superlink --insecure --fleet-api-type grpc-adapter ... --serverappio-api-address ... --fleet-api-address ... --control-api-address ...`.
  - `:404-413` readiness check: `create_channel(server_addr=fleet_api_addr, test_only=True)` (gRPC channel probe).
  - `:416` `flwr run . nvflare --format json`.
  - `:437-451` `_build_flower_config` writes `config.toml` with `[superlink.nvflare] address = <exec_api_addr>`, `insecure = true`.
  - `:462-492` `_flower_command` builds `flwr run`, `flwr stop`, `flwr list`.
  - `:532` `stop`; `:583-625` `_query_for_run_status` parses `flwr list` JSON (`runs[0].status`, `startswith("finished")`, `endswith("completed")`).
- `nvflare/app_opt/flower/connectors/grpc_server_connector.py:72` `get_local_addresses(3)` (serverapp, fleet, exec); `:84` `GrpcClient(fleet_api_addr)`; `:95` `send_request_to_flower` converts the Shareable to `pb2.MessageContainer` and calls `SendReceive`.
- `nvflare/app_opt/flower/connectors/grpc_client_connector.py:104` `get_local_addresses(2)`; `:110` hosts `GrpcServer` with `self` as servicer; `:135` `SendReceive` forwards to the FLARE server via `_send_flower_request`. When the FLARE side has stopped, or the server returns a non-OK return code, it sets `_stopping` and `_exit_waiter` and returns `reply_should_exit()`. `_stop_client` waits up to 2 s for that should-exit to be delivered, then sleeps 1 s before stopping the applet.
- `nvflare/app_opt/flower/grpc_client.py`, `grpc_server.py` (whole files): `GrpcAdapterStub`/`GrpcAdapterServicer`, `create_channel`, `add_insecure_port`. Import `flwr.proto.grpcadapter_pb2*`.
- `nvflare/app_opt/flower/utils.py` (whole file): `msg_container_to_shareable`, `shareable_to_msg_container`, `reply_should_exit` on `pb2.MessageContainer`. Shareable keys are `flower.content`, `flower.headers`, `flower.name` (`PARAM_KEY_*` in `defs.py`).
- `nvflare/app_opt/flower/defs.py:48-50` the three server address keys; `:60-62` `GRPC_DEFAULT_OPTIONS`, using `MAX_FRAME_SIZE`.
- `nvflare/app_opt/flower/controller.py:42, 62, 88, 95-101`: public `int_client_grpc_options` parameter and the `GrpcServerConnector` factory.
- `nvflare/app_opt/flower/executor.py:53, 62-67, 85`: `int_server_grpc_options` and the `GrpcClientConnector` factory.
- `nvflare/app_opt/flower/recipe.py:26-27, 74`: `SUPPORTED_FLWR_MIN_VERSION = 1.26`, `SUPPORTED_FLWR_SPEC = "flwr>=1.26"`, docstring.
- `nvflare/app_opt/flower/mock/`: a gRPC echo harness (`echo_servicer.py`, `flower_client.py`, `flower_server.py`) built on `GrpcServer`/`GrpcClient` and `pb2`; `applet.py`, `controller.py`, `executor.py` subclass the real controller/executor.
- `nvflare/fuel/utils/component_classes.json`: check whether it lists the gRPC connector classes.
- `.github/workflows/premerge.yml:196, 327, 435`: CI installs `uv pip install --system --no-deps "flwr>=1.16,<1.26"`. That is below the 1.26 minimum the recipe enforces. `--no-deps` was added in #4267 to keep Flower from changing NVFlare's grpc/protobuf versions.

Flower-agnostic parts that stay: `Connector`, `TieController`, `TieExecutor`, `CLIApplet`, `ProcessManager`, the job/recipe API (`FlowerJob`, `FlowerRecipe`), the job meta handling in `nvflare/private/fed/server/job_meta_validator.py` and `app_authz.py`, and `nvflare/app_common/tie/*`.

Tests touching the integration:

- `tests/unit_test/app_opt/flower/applet_test.py` (8 test functions), including `[superlink.nvflare]` config assertions (`:103-128`).
- `tests/unit_test/app_opt/flower/test_flower_applet.py` (5).
- `tests/unit_test/recipe/flower_recipe_test.py` (13): version gating at 1.26, parametrized over versions from `1.15.9` to `1.27.5`.
- `tests/unit_test/recipe/flower_job_test.py` (16).
- No test imports `flwr`, or any NVFlare module that imports it (`utils.py`, `grpc_*`, connectors). No test exercises the transport, and there are no Flower integration tests. This is why the breakage went unnoticed.

## 4. What Flower 1.43 provides

Verified by reading the 1.43.0 wheel (`flwr/`). Confirmed by a run in the spike (section 12.1), except where section 12 says otherwise.

- **One HTTP server for the SuperLink.** `flwr/superlink/main.py:213-219` mounts Health, Control (`/v1/control`), Fleet (`/v1/fleet`), and Runtime routers on one FastAPI app, served by a single `uvicorn.run` (`flwr/superlink/cli/flower_superlink.py:371`). `--host` and `--port` set its address. `--fleet-api-type`, `--fleet-api-address`, and `--control-api-address` are accepted but deprecated; `--serverappio-api-address` is rejected. The three NVFlare server addresses collapse into one.
- **Defaults to override.** `--host` defaults to `127.0.0.1`; SuperLink `--port` defaults to a fixed 8000 and SuperNode `--port` to 9094 (`flwr/supercore/constant.py:76-78`). Concurrent jobs, or two simulated clients on one host, collide unless NVFlare passes a free port explicitly.
- **ServerApp path (was open question O1).** In subprocess isolation the SuperLink starts its superexec with `--runtime-api-address <host>:<port>`, the same host and port as the Fleet API (`flower_superlink.py:139-141, :412`). One port serves the SuperNode tunnel, the ServerApp, and `flwr` CLI control calls.
- **Fleet API is protobuf over HTTP.** `flwr/supernode/fleet_http_client.py` posts `application/protobuf` bodies (`flwr/supercore/protobuf/constants.py:18`) to fixed paths: `/v1/fleet/register-node`, `activate-node`, `deactivate-node`, `unregister-node`, `send-node-heartbeat`, `pull-messages`, `push-messages`, `get-run`, `get-fab`, `push-object`, `pull-object`, `confirm-message-received`. All are unary; no Fleet call uses the streaming helper. The SuperNode makes no calls outside `/v1/fleet/` on the SuperLink address.
- **Request headers matter.** The Fleet router runs a `VersionDependency` on every request (`flwr/superlink/routers/fleet/router.py:56-68`), which reads the request headers (`flwr/supercore/dependencies/version.py:42`). A tunnel must forward the SuperNode's request headers.
- **SuperNode address is `host:port`** with no scheme (`flwr/supernode/start_client_internal.py:587`). `--insecure` selects plain HTTP.
- **SuperNode retries are unlimited by default.** `--max-retries` and `--max-wait-time` default to `None`, "no limit" (`flwr/supernode/cli/flower_supernode.py:409-424`). A SuperNode whose SuperLink becomes unreachable keeps retrying until it is killed.
- **Heartbeats.** Interval 30 s, call timeout 5 s, patience 2 (`flwr/common/constant.py:33-40`). These are the same values as in 1.27, so the tunnel's latency budget for heartbeats does not change.
- **Health check:** `GET /health` (and `HEAD`) (`flwr/supercore/routers/health/router.py:23-25`).
- **`flwr` CLI (was open question O2).** `flwr run <app> <connection>` (`flwr/cli/run/run.py:55`), `flwr stop <run_id> <connection>` (`flwr/cli/stop.py`), and `flwr list` (`flwr/cli/app.py:83`) keep the argument order the applet uses. `flwr list --format json` returns `{"success": true, "runs": [{"run-id": ..., "status": ...}]}` with `status` as `status:sub_status`, for example `finished:completed` (`flwr/cli/ls.py:296-324`, `flwr/cli/run_utils.py:110-112`). The applet's parsing at `applet.py:583-625` matches.
- **Connection config** still uses `[superlink.<name>] address = ...` and `insecure = true` (`flwr/cli/constant.py:116-118`). The `config.toml` generated at `applet.py:437-451` keeps its shape.
- **Dependencies:** `grpcio>=1.70,<2`, `protobuf>=5.28,<7`, `uv>=0.13.5,<0.14`, `uvicorn[standard]>=0.49,<0.50`, plus `httpx`, `fastapi`, `starlette`, `sqlalchemy`, `alembic`. Resolves with NVFlare's base install (section 2).

### Call pattern (affects performance)

1.27 already sends models as an object tree: the inflatable object store and the Fleet `PushObject`/`PullObject` RPCs exist in 1.27 (`flwr/supercore/inflatable/`, `flwr/proto/fleet_pb2_grpc.py`), and the gRPC adapter tunnels each of them as one `SendReceive` (section 3). The HTTP tunnel keeps that one-to-one mapping. So the number of FLARE ReliableMessage transactions per round should stay about the same. The extra cost is only the local HTTP hop on each side, which is small compared with a ReliableMessage round trip. This replaces an earlier draft's claim that 1.43 multiplies transactions; that claim was wrong.

## 5. Change list

### 5.1 Runner: launch commands (`applet.py`)

| Location | Now | 1.43 |
|---|---|---|
| `:189-205` SuperNode | `flower-supernode --insecure --grpc-adapter --superlink <superlink_addr> --clientappio-api-address <clientapp_api_addr>` | `flower-supernode --insecure --superlink <proxy_addr> --host 127.0.0.1 --port <clientapp_api_port>`. Always pass `--port` (default 9094 collides between simulated clients). |
| `:372-381` SuperLink | `flower-superlink --insecure --fleet-api-type grpc-adapter --serverappio-api-address ... --fleet-api-address ... --control-api-address ...` | `flower-superlink --insecure --host 127.0.0.1 --port <port> [--database ...] [--allow-runtime-dependency-installation]`. Always pass both `--host` (keep it loopback-only) and `--port` (default 8000 collides between jobs). |
| `:404-413` readiness | `create_channel(fleet_api_addr, test_only=True)` | Poll `GET http://<addr>/health` until 200, within `superlink_ready_timeout`. Drop the `grpc_utils` import (`:31`). |
| `:416`, `:462-492`, `:583-625` | `flwr run`, `flwr stop`, `flwr list` | No change (section 4). |
| `:437-451` `config.toml` | `address = <exec_api_addr>` | Same keys; address is the SuperLink's single address. |
| `:315-317`, `defs.py:48-50` | `APP_CTX_SERVERAPP_API_ADDR`, `APP_CTX_FLEET_API_ADDR`, `APP_CTX_EXEC_API_ADDR` | One address key (name TBD). |
| `:172-173` | `APP_CTX_SUPERLINK_ADDR`, `APP_CTX_CLIENTAPP_API_ADDR` | Keep; `SUPERLINK_ADDR` now holds the local proxy address. |
| `:42`, `_check_runtime_dependency_installation_support` | Version check for runtime dependency installation (`>=1.29`) | Remove the check. **Corrected (section 12.1):** the SuperLink needs `--disable-runtime-dependency-installation` when the option is False; the SuperNode needs `--allow-...` only when True. |
| `:80`, docstrings | `pip install 'flwr>=1.26'` | `pip install 'flwr>=1.43'`. |

### 5.2 Connectors and transport

| File | Change |
|---|---|
| `connectors/grpc_server_connector.py` | Replaced by `connectors/http_server_connector.py` (`HttpServerConnector`). `start()`: allocate 1 address (was 3). `send_request_to_flower`: `POST http://<superlink><path>` with the tunneled headers and body; return status, headers, and body in the reply Shareable. |
| `connectors/grpc_client_connector.py` | Replaced by `connectors/http_client_connector.py` (`HttpClientConnector`): the gRPC servicer becomes a local HTTP proxy (section 6.2) that accepts `POST /v1/fleet/*` from the SuperNode and forwards through `_send_flower_request`. Still 2 addresses (proxy, ClientApp API). Keep the `_stopping` detection; change the stop path per section 6.3. |
| `connectors/flower_connector.py` | Base classes unchanged. |
| `grpc_client.py`, `grpc_server.py` | Delete. New code goes into the connectors or small helper modules. |
| `utils.py` | Remove `pb2`. Replace the converters with functions over `{path, headers, body, status}`. |
| `defs.py:60-62` | Remove `GRPC_DEFAULT_OPTIONS` and the `MAX_FRAME_SIZE` import if unused. |

Constraint: the bridge must not import `flwr` or `httpx`. It forwards opaque bytes, so it needs no Flower types. Use the standard library (`http.server`, `http.client`) or `requests`, which NVFlare already depends on. This keeps `nvflare.app_opt.flower` importable without Flower's dependency set, and lets CI keep its `--no-deps` install (section 5.8).

### 5.3 Public API (`controller.py`, `executor.py`, `flower_job.py`, `recipe.py`)

- `FlowerController(int_client_grpc_options=...)` and `FlowerExecutor.int_server_grpc_options` are public. Deprecate them (accept, ignore, warn) for one release, then remove; or remove directly if O7 finds no users. `FlowerJob` and `FlowerRecipe` do not expose them.
- `recipe.py:26-27, 74`: `SUPPORTED_FLWR_MIN_VERSION = Version("1.43")`, `SUPPORTED_FLWR_SPEC = "flwr>=1.43"`. Whether to also cap is O5.
- Remove the `flwr>=1.29` runtime-dependency notes in `flower_job.py`, `controller.py`, `executor.py`, and `applet.py` docstrings.
- Wire compatibility: FLARE server and client must run the same NVFlare version, because the Shareable contents change (5.4). State this in the release notes; do not support mixed versions.

### 5.4 Shareable contract

Keep the three existing keys so `PARAM_KEY_*` stays stable, with new meanings:

- `flower.name`: the HTTP path, for example `/v1/fleet/pull-messages`.
- `flower.headers`: forwarded HTTP headers (dict of str to str), with hop-by-hop headers removed (6.2).
- `flower.content`: request or response body bytes.

Add one key for the HTTP status code on replies: `flower.status` (`PARAM_KEY_STATUS`).

### 5.5 Mock harness (`mock/`)

Deleted (section 12.3): the harness was already dead code, and `tests/integration_test/fast/flower_http_bridge_test.py` replaces it. `applet.py`, `controller.py`, `executor.py` need import changes only if connector class names change.

### 5.6 Examples and docs

- `examples/hello-world/hello-flower/requirements.txt`, `flwr-pt/pyproject.toml`, `flwr-pt-tb/pyproject.toml`: `flwr>=1.43`; remove the `<1.40` pins and gRPC-adapter comments; update the "Tested with" headers.
- `examples/hello-world/hello-flower/flwr-pt-tb/flwr_pt_tb/client.py:20`: `from flwr.common.record import MetricRecord, RecordDict` fails on 1.43. Change it to `from flwr.common import MetricRecord, RecordDict`; `flwr.common` re-exports both in 1.43 (`flwr/common/__init__.py:69-70`).
- `examples/hello-world/hello-flower/README.md` lines 41-47 and `docs/hello-world/hello-flower/index.rst` lines 48-56 (the `<1.40` note).
- `docs/user_guide/data_scientist_guide/flower_integration/flower_run_as_flare_job.rst` lines 7-23 and `flower_job_structure.rst` lines 41, 53, 83-85.
- `docs/programming_guide/fed_job_api.rst`, `docs/programming_guide/timeouts.rst`: check for gRPC-adapter or transport-specific text.
- `nvflare/tool/examples/catalog.json`: check the hello-flower entry.

### 5.7 CI (`.github/workflows/premerge.yml`)

Lines 196, 327, 435: change `"flwr>=1.16,<1.26"` to `"flwr>=1.43,<1.44"` (or the O5 choice). Keep `--no-deps` if the bridge follows the 5.2 constraint; the unit tests then only need Flower's metadata for the recipe version check. A test that runs the real `flwr` CLI needs Flower's full dependency set and belongs in the integration layer (section 7).

### 5.8 Remove

- `grpc_client.py`, `grpc_server.py`, and `mock/echo_servicer.py` (unless rewritten per 5.5).
- All imports of `flwr.proto.grpcadapter_*`.

## 6. Bridge design

### 6.1 Principle

FLARE server and client talk only through FLARE ReliableMessage. The Fleet API is unary request/response over HTTP with opaque protobuf bodies, so the bridge is a transparent HTTP-over-FLARE tunnel:

1. The SuperNode sends `POST /v1/fleet/<op>` to the local proxy.
2. The proxy puts `{path, headers, body}` in a Shareable and sends it with `_send_flower_request`, as today.
3. The server connector sends the same request to `http://<superlink><path>`.
4. `{status, headers, body}` travels back in the reply; the proxy writes it out as the HTTP response.

The bridge never parses protobuf, so Flower schema changes do not affect it.

### 6.2 Proxy

- `http.server.ThreadingHTTPServer` on `127.0.0.1` and a free port. Concurrent requests are expected (heartbeats run beside pulls and object transfers). Set `protocol_version = "HTTP/1.1"` so the SuperNode's HTTP client can keep connections alive.
- Accept only `POST` under `/v1/fleet/`; return 404 otherwise.
- Forward request and response headers except hop-by-hop and length headers (`Connection`, `Keep-Alive`, `Transfer-Encoding`, `Content-Length`, `Host`), which each hop sets itself.
- Preserve the SuperLink's status code and body, including errors.
- No body-size limit beyond FLARE's own (frame limit about 2 GB, `nvflare/fuel/f3/drivers/net_utils.py:40`), the same bound as today.

### 6.3 Stopping the SuperNode

The gRPC design told the SuperNode to exit with a `should-exit` metadata reply. 1.43 has no such channel, and the SuperNode retries forever by default (section 4). So:

- When the FLARE side has stopped or the server returns a non-OK return code, the proxy returns HTTP 503 and sets `_stopping`, as `SendReceive` does today. The existing `_is_stopped` logic then ends the connector.
- `_stop_client` stops waiting for a should-exit delivery (the 2 s `_exit_waiter` wait and the 1 s sleep) and terminates the SuperNode through `stop_applet`. `_exit_waiter` can be removed.
- Pass `--max-wait-time` to the SuperNode as a safety net (default 600 s, `FlowerClientApplet(supernode_max_wait_time=...)`; see section 12.1 for what it bounds). During testing for this spec, SuperLink, SuperNode, and ClientApp processes stayed alive after the parent simulator process was killed (1.27, abnormal termination; cause not investigated). A bounded retry time limits how long an orphaned SuperNode survives.

### 6.4 Timeouts

Call volume and heartbeat timing are unchanged from 1.27 (section 4), so the current `per_msg_timeout` (10 s) and `tx_timeout` (100 s) defaults are a reasonable starting point. One mismatch to watch: the SuperNode's heartbeat call times out after 5 s, below `per_msg_timeout`. A slow ReliableMessage round trip fails the heartbeat before FLARE gives up. This is also true today.

## 7. Tests

Follow `tests/README.md#choosing-the-test-layer` before adding or moving tests (per `CLAUDE.md`).

1. **Unit, no Flower needed:** SuperLink and SuperNode command builders (including explicit `--host`/`--port`); `config.toml` contents; version gating (`1.42.9` rejected, `1.43.0` accepted); proxy round trip with a fake upstream (path, headers, body, status preserved; hop-by-hop headers dropped; non-POST and non-`/v1/fleet/` paths get 404); 503 and `_stopping` when the server side has ended.
2. **Integration, real Flower CLI:** with Flower's full dependency set, run `flower-superlink --help` and `flower-supernode --help` and assert that every flag the applet passes exists. This catches the next flag removal of the kind that broke 1.36 and 1.43.
3. **Integration, real runner:** run `examples/hello-world/hello-flower/job.py` with `flwr-pt` on the simulator, 2 clients, 1 round, and assert the job completes. CIFAR-10 is about 170 MB; CI needs a cached copy or a smaller dataset (O6).

Acceptance: `flwr-pt` and `flwr-pt-tb` complete all rounds on 1.43 in the simulator; existing unit tests pass with updated expectations; `./runtest.sh -u` and `./runtest.sh -s` are clean.

## 8. Rollout

- Release notes: "The Flower integration requires `flwr>=1.43`. The gRPC-adapter transport is removed. FLARE server and client must run the same NVFlare version."
- Users who must stay on `flwr<1.36` should stay on an NVFlare release from before this change.

## 9. Phased plan

1. **Spike (about 1 day):** manual run of a 1.43 SuperLink and SuperNode through a stub proxy, to confirm section 4 in practice (one port, headers, `/health`, `flwr run`/`list`/`stop`). Optionally time one round on 1.27 and 1.43 with a larger model as a performance baseline (O4).
2. **Runner and bridge (one PR):** sections 5.1, 5.2, 5.4, 6.
3. **API cleanup, mocks, CI:** sections 5.3, 5.5, 5.7, 5.8.
4. **Tests and examples:** sections 5.6 and 7.
5. **Docs and release notes:** sections 5.6 and 8.

Step 2 carries the risk; the rest is mechanical once it works.

## 10. Open questions

Resolved from 1.43 source (confirm in the spike run):

- **O1. ServerApp port:** same host and port as the Fleet API (section 4).
- **O2. `flwr` CLI shape:** unchanged; the applet's parsing matches (section 4).
- **O3. Should-exit:** terminate the SuperNode from the applet (section 6.3). Relying on SuperNode retry limits does not work; they are unlimited by default.

Still open:

- **O4. Performance baseline. NOT DONE.** (see section 12.4)
  Original text: The call pattern is unchanged (section 4), so no regression in transaction count is expected. Time one round on 1.27 and 1.43 with the same larger model (a few hundred MB) to confirm. Optimizations such as batching objects or using FLARE streaming are out of scope unless this shows a regression.
- **O5. Upper bound. DECIDED, pending maintainer confirmation:** CI is capped at `<1.44`; the recipe check and install hints are not (no cap in user installs). Original text: Cap at `<1.44` or not. Flower changed the runner at 1.26, 1.36, and 1.43, so an uncapped `>=1.43` may break again. A cap trades silent breakage for an explicit upgrade step.
- **O6. CI dataset. PARTLY ANSWERED:** the CI integration test uses a numpy-only app, so it needs no dataset (section 12.4). CIFAR-10 for hello-flower itself is still open. Original text:  for the integration test, and whether CI can fetch CIFAR-10.
- **O7. External users. UNKNOWN:** cannot be determined from the repo. Chosen: deprecate (accept, ignore, warn). Original text:  of `int_client_grpc_options`/`int_server_grpc_options`; decides deprecate versus remove.
- **O8. Dev extras. ANSWERED:** `pip install -e ".[dev]" "flwr==1.43.0"` resolves (protobuf 6.30.2, click 8.1.7), and the Flower CLI and runner work with click 8.1.7. Original text: The resolver check covered NVFlare's base install with `flwr==1.43.0`; check `.[dev]` too (it pins `click==8.1.7` on Python below 3.14).

## 11. Verification evidence (for the PR description)

- End-to-end runs, `examples/hello-world/hello-flower`, `flwr-pt`, simulator, 2 clients:
  - Flower 1.27.0: training round 1 completed on both clients; round 2 started; the run ended with no logged error and no final result (not verified to completion).
  - Flower 1.39.0: failed at superlink startup, `--serverappio-api-address` no longer supported.
- Install checks: `nvflare` with `flwr==1.39.0` does not resolve (`uv` pin); with `flwr==1.43.0` it resolves (base install, dry run).
- CLI and wheel inspection: 1.26 accepts the flags; 1.33 accepts `--serverappio-api-address`; 1.36–1.38 reject it with a stub; 1.40–1.42 reject it at `--help`; 1.43 removes `flwr.proto.grpcadapter_pb2*`, has no SuperNode `--grpc-adapter`, and rejects `--clientappio-api-address`.
- Not run: 1.26, 1.28–1.38, 1.40–1.42.
- Flower 1.43.0, simulator, 2 clients, NVFlare from this branch (CPU torch 2.15.0, click 8.1.7, protobuf 6.30.2):
  - `flwr-pt`, 3 rounds (the pyproject default): `Flower Run ... finished: status='finished:completed'`; about 12 to 14 min per round on CPU.
  - `flwr-pt-tb` with `--stream_metrics`, **1 round** (a scratch copy of the app with `num-server-rounds = 1`): job `FINISHED:COMPLETED`, 1259 s. Not run for 3 rounds.
  - numpy-only app, 2 rounds, through the bridge: `FINISHED:COMPLETED` (`flower_runner_test.py`).
  - No Flower processes were left after the completed runs.

## 12. Corrections and additions from the spike and implementation

### 12.1 Spike results (Flower 1.43.0, SuperLink and two SuperNodes through a stub stdlib proxy)

Confirmed as written in section 4:

- One SuperLink port serves Fleet (via the proxy), the ServerApp (`flower-superexec --runtime-api-address` is the same host:port), and `flwr` CLI control calls.
- `GET /health` and `HEAD /health` return 200.
- Fleet calls are all unary `POST /v1/fleet/<op>`; ops seen: register-node, activate-node, send-node-heartbeat, pull-messages, push-messages, get-run, get-fab, push-object, pull-object, confirm-message-received. The SuperNode sends `flwr-component-name`, `flwr-package-name`, `flwr-package-version`, `flwr-public-key`, `flwr-signature`, `flwr-timestamp`, and `content-type`/`accept` headers. A request without them gets `401 {"detail":"Invalid SuperNode authentication"}`. (That is an authentication check; the version check in `VersionDependency` was not isolated.)
- `flwr run . <conn> --format json` returns `run-id`; `flwr list <conn> --format json` returns `status` `running`, then `finished:completed`; `flwr stop <run-id> <conn>` leads to `finished:stopped`.

New or more precise:

- `flower-superlink` starts `flower-superexec` by bare name, which starts `flwr-serverapp`. Without the venv `bin` directory on `PATH`, the SuperLink fails with `FileNotFoundError: 'flower-superexec'`. The applet already adds it (`_build_flower_env`); this must be kept.
- `flwr list` is `flwr list [SUPERLINK] [--run-id N]`. The run id is an option, not a positional. The applet never passes one, so it is unaffected. `flwr list . <conn>` (the old form with an app path) fails with a migration message.
- A 503 from the tunnel does not stop a SuperNode: 502/503/504 are retryable, and the SuperNode retries with growing backoff (about 16 s apart after a minute). The same holds when the proxy dies. A 401 on heartbeat makes the SuperNode send itself SIGINT. Section 6.3's decision (terminate the SuperNode from the applet) is therefore required, not optional.
- `--max-wait-time` bounds the retrying of a single call, not the SuperNode's lifetime. It is a safety net only.
- SuperLink became ready about 1 s after start (`superlink_ready_timeout=10` is ample).
- Observed once: after a failed run, the SuperLink was terminated but its `flwr-serverapp` child stayed alive with parent pid 1. Cause not investigated. Not seen again in later clean runs.

- **Runtime dependency installation is on by default in the 1.43 SuperLink.** `--allow-runtime-dependency-installation` is a deprecated no-op there, and `--disable-runtime-dependency-installation` (mutually exclusive with it) switches it off. The SuperNode default is still off. Without the disable flag every run starts a `uv sync` of the app's `pyproject.toml` dependencies in the SuperLink's environment: in the first hello-flower attempt it tried to download CUDA torch wheels, filled the disk, and the run ended `finished:failed` with the cause only visible in `flwr list` `status-details`. The applet now passes `--disable-runtime-dependency-installation` unless `allow_runtime_dependency_installation=True`. The e2e test uses an unresolvable app dependency to guard this.
- Orphans on abnormal exit: when a simulator process ended on a failed run, one SuperNode and its SuperExec survived with parent pid 1 (bounded only by `--max-wait-time`, default 600 s here). Not fixed; same class of problem as the 1.27 observation above.

### 12.2 Security: the tunnel must not expose the other SuperLink APIs

The SuperLink serves the Control API (`/v1/control`, which starts and stops runs) and the Runtime API on the same port as the Fleet API. A server connector that replays any path from a FLARE client would let a client call them. `HttpServerConnector` therefore replays only paths matching `^/v1/fleet/[A-Za-z0-9_-]+(/[A-Za-z0-9_-]+)*$` and answers `BAD_REQUEST_DATA` otherwise (`utils.is_fleet_path`). The client proxy also returns 404 for anything outside `/v1/fleet/`. Section 6.2 only required the client-side check.

### 12.3 Deviations from the change list

- 5.2: classes are `HttpServerConnector` and `HttpClientConnector`; `component_classes.json` was updated. The client connector has a `_start_proxy` method so tests can run the real proxy without the applet. The server connector no longer takes `flower_server_ready_timeout`; readiness is polled by the applet.
- 5.3: `int_server_grpc_options` on `FlowerExecutor` was an attribute, not a constructor argument, and was removed. `FlowerController(int_client_grpc_options=...)` is deprecated (accepted, ignored, warns). No upper cap in the recipe check (O5).
- 5.4: response headers `date` and `server` are also dropped, because the proxy's HTTP server writes its own. Duplicate response headers collapse to one value (Fleet responses carry none).
- 5.5: `mock/` was deleted, and its `component_classes.json` entries removed. It referenced `Constant.APP_CTX_SERVER_ADDR`, which does not exist, so it could not have run, and nothing used it.
- 5.1: the readiness poll also fails immediately if the SuperLink process has exited, instead of waiting for the timeout.
- 5.7: besides the pin change, a new job `flower-integration-tests` installs Flower with its dependencies and runs the two integration files in section 12.4.

### 12.4 Tests as implemented

- Unit (`tests/unit_test/app_opt/flower/`): `applet_test.py` (command builders, config, readiness), `utils_test.py`, `http_client_connector_test.py` (proxy driven through fake socket files, no sockets), `http_server_connector_test.py` (HTTP connection faked), `controller_executor_test.py`; `tests/unit_test/recipe/flower_recipe_test.py` (1.42.9 rejected, 1.43.0 accepted).
- Integration, no Flower needed: `tests/integration_test/fast/flower_http_bridge_test.py` (real sockets, fake SuperLink, FLARE hop replaced by a direct call).
- Integration, needs Flower with dependencies (skips otherwise): `tests/integration_test/fast/flower_runner_test.py`: flags passed by the applet exist in the installed Flower's `--help`; a numpy-only Flower app completes 2 rounds in the simulator with 2 clients, with no dataset.
- `./runtest.sh -u`: 18335 passed, 2 failed (`slurm_batch_test.py::test_renderer_delivers_credentials_through_environment[apptainer|pyxis]`), run with `umask 022`. They assert `'-d' not in <command>`, and the command contains the pytest temp path, which here includes the user name `dvanderwal`; they pass with `--basetemp` set to a neutral path. Unrelated to this change. Without `umask 022`, 6 more tests that check file modes fail (package and deploy tests).
- `./runtest.sh -s` and `-l`: pass.
- Not done: O4 (performance baseline, 1.27 versus 1.43). The 1.27 side would need the old code and a larger model run.
