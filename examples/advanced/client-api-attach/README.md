# Client API Attach Mode

This example runs the ordinary Client API loop in an externally owned process.
The trainer may start before or after the NVFlare job; both sides rendezvous on
`attach_id=numpy_trainer`.

`attach_profile.json` is a template. Before running the trainer, replace its
`connect_url` with the Cell child endpoint exported by `site-1`. The endpoint is a
deployment value: the site operator must make the site's internal listener
reachable by the trainer and provision its URL outside the submitted job. The
template uses port `0` deliberately so it cannot silently connect to an unrelated
service.

For a provisioned secure site, use the secure endpoint and add:

```json
{
  "connection_security": "mtls",
  "ca_cert": "/absolute/path/to/site-1/startup/rootCA.pem"
}
```

Keep `client.crt` and `client.key` beside `rootCA.pem`; Cell discovers them using
the same credential path as `IPCAgent`. Attach validates that all three files are
present and readable before constructing the Cell. Bare-CA one-way TLS
(`connection_security=tls`) is not supported. A cleartext non-loopback endpoint
is rejected unless the job explicitly sets `allow_insecure_attach=True`.

The attach protocol itself is driver-neutral. When the replacement
shared-filesystem driver lands, its profile can use a provisioned endpoint such as
`shared-file://0/absolute/shared/directory` and omit the TLS/CA fields. That driver
will enforce its own path and permission policy; it is not included in this change.

## Local POC

The stock one-client POC selects its client Cell listener port dynamically, so
configure a fixed listener before starting the site. The following walkthrough
uses port `8101`.

Prepare a one-client workspace:

```bash
nvflare poc config --pw /tmp/nvflare-attach-poc
nvflare poc prepare -n 1 --force
```

Create
`/tmp/nvflare-attach-poc/example_project/prod_00/site-1/local/comm_config.json`:

```json
{
  "allow_adhoc_conns": false,
  "backbone_conn_gen": 2,
  "internal": {
    "scheme": "grpc",
    "resources": {
      "host": "127.0.0.1",
      "port": 8101,
      "connection_security": "clear"
    }
  }
}
```

The component policy must authorize the exact job components. In each of these
directories, copy `resources.json.default` to `resources.json` if the latter does
not exist, preserve its existing `class_allow_list`, and ensure the indicated
class is present:

- `site-1/local/resources.json`:
  `nvflare.app_common.executors.client_api_executor.ClientAPIExecutor`
- `server/local/resources.json`:
  `nvflare.app_common.widgets.metrics_artifact_writer.MetricsArtifactWriter`

The second class is the metrics writer included by `BaseFedJob`; it is not specific
to Attach. Do not replace either allow-list with a broad package prefix. These
files and `comm_config.json` are startup configuration, so restart an already
running POC after changing them.

Change `attach_profile.json` to use:

```json
{
  "site_name": "site-1",
  "connect_url": "grpc://127.0.0.1:8101",
  "connection_security": "clear"
}
```

Port `8002` is the POC server's federation endpoint, not the `site-1` Cell
endpoint, and must not be used as `connect_url`.

Start the POC and install the example dependency:

```bash
nvflare poc start -ex admin@nvidia.com
python -m pip install -r examples/advanced/client-api-attach/requirements.txt
```

In one terminal, submit and monitor the job:

```bash
python examples/advanced/client-api-attach/job.py \
  --startup_kit_location \
  /tmp/nvflare-attach-poc/example_project/prod_00/admin@nvidia.com
```

`job.py` uses `ProdEnv` to submit the recipe and waits until the run completes.
Pass `--username` if the startup kit belongs to a user other than
`admin@nvidia.com`.

In another terminal, start the independently managed trainer:

```bash
cd examples/advanced/client-api-attach
python trainer.py --config attach_profile.json
```

The trainer and job commands may be started in either order, but the site and its
provisioned Cell endpoint must already be available.

Successful output ends with `Status: FINISHED:COMPLETED` and prints the downloaded
result directory. For this three-round example, verify the final model with:

```bash
python -c 'import numpy as np, sys; print(np.load(sys.argv[1]))' \
  <RESULT_DIR>/workspace/models/server.npy
```

The expected value is:

```text
[[4 5 6]
 [7 8 9]]
```

The server job log is `<RESULT_DIR>/workspace/log.txt`. The client job log is
`/tmp/nvflare-attach-poc/example_project/prod_00/site-1/<JOB_ID>/log.txt`, and the
trainer writes its log to its terminal.

`flare.send()` does not return until any lazy result payload has reached
receiver-confirmed terminal success. Keep the trainer process alive through that
confirmation; job shutdown closes only its attach Cell session and never terminates
the trainer process.
