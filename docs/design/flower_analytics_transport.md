# Analytics transport over a direct Cell

## Scope

This change refactors the existing `MetricsSender` and `MetricRelay` analytics
transport from Pipe/PipeHandler to direct Cell messaging. Flower is the first
standalone consumer; the transport itself is not Flower-specific.

```text
launched process -> MetricsSender -> normal child Cell
                                  -> existing CJ Cell -> MetricRelay
                                  -> existing federated analytics event
                                  -> server tracking receiver
```

Flower model exchange continues through its TIE connectors. The analytics path
has no task backend, task/model protocol, Pipe, or Client API selector.

## Operation

At `ABOUT_TO_START_RUN`, `MetricRelay` uses the existing CJ Cell's internal
listener, allocates a launch-specific child FQCN, and registers one `LOG`
callback. It atomically writes an owner-only bootstrap containing only the
internal listener URL and the receiver/client Cell FQCNs.

For a Flower job, `FlowerJob` installs the relay and the Flower applet passes the
bootstrap path in `NVFLARE_ANALYTICS_BOOTSTRAP`. `nvflare.client.tracking.init()`
creates `MetricsSender`; rank zero starts a credential-free child Cell and sends
each analytic DXO directly to the relay. `FlowerExecutor` has no analytics
responsibilities.

For temporary compatibility, `ExternalConfigurator` reads the relay's direct
Cell configuration through `get_external_config()`, allowing legacy
`flare.log()` to use `MetricsSender` without making `MetricRelay` inherit the
old `AttributesExportable` abstraction. This bridge is not used by Flower and
can disappear with the remaining task-exchange stack.

The maintained Flower example aliases the explicit tracking module as `flare`:
`import nvflare.client.tracking as flare`. This preserves the familiar
`flare.init()`/`flare.shutdown()` lifecycle spelling without initializing the
task/model Client API. Existing `SummaryWriter` calls are unchanged.

## Trust and lifecycle

This local leg follows the existing trusted launched-process model: NVFlare
launches the external process for the site job and supplies its Cell identity.
The relay accepts `LOG` only from that launch-specific Cell origin. This is a
routing/association check, not a separate authentication or delegation
protocol.

The sender Cell uses `root_url=None`, `secure=False`, and empty credentials, so
the analytics transport does not give the launched process a site/server bearer
credential. Flower receives only the dedicated analytics bootstrap. The secure
CP-to-server connection is unchanged.

Cell owns connection liveness and the runtime launcher owns process shutdown.
At `ABOUT_TO_END_RUN`, `MetricRelay` removes the bootstrap and clears the active
origin. There is no second heartbeat, token/session, or shutdown protocol layered
over Cell.

## Colossus validation

1. Provision a secure server and three clients and run three rounds of
   `hello-flower/flwr-pt-tb`.
2. Verify 36 TensorBoard scalar points: four tags, three steps, and three sites.
3. Inspect the Flower environment, bootstrap, open files, and captured messages;
   assert the site bearer token and signature are absent and the bootstrap
   contains only the internal URL and Cell FQCNs.
4. Repeat with the site internal scheme set to FileDriver/shared-file. The
   bootstrap should contain that internal URL without any F3 customization.
5. Test normal completion and abort. Confirm no launched processes, bootstrap,
   internal endpoint, or TensorBoard handles remain.

This PR must land before #5044. After rebasing, #5044 can remove
`ExternalConfigurator`, the task-side legacy Client API stack, and the Pipe
hierarchy without changing Flower analytics again.
