# Client API Attach Mode

This example runs the ordinary Client API loop in an externally owned process.
The trainer may start before or after the NVFlare job; both sides rendezvous on
`attach_id=numpy_trainer`.

`attach_profile.json` is a template, not a ready-made POC profile. Before running
the trainer, replace its `connect_url` with the existing Cell child endpoint
exported by `site-1`. The endpoint is a deployment value: the site operator must
make the site's internal listener reachable by the trainer and provision its URL
outside the submitted job. The template uses port `0` deliberately so it cannot
silently connect to an unrelated service.

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

The stock one-client POC is not a complete attach deployment: its client Cell
listener uses a dynamically selected internal port and that port is not exported
as a trainer profile. In particular, `grpc://127.0.0.1:8002` is the POC server's
federation endpoint, not the `site-1` Cell endpoint. Do not use it as
`connect_url`.

After the site operator has provisioned a reachable site Cell endpoint and updated
`attach_profile.json`, install this repository and export the job:

```bash
python -m pip install -e .
python -m pip install -r examples/advanced/client-api-attach/requirements.txt
python examples/advanced/client-api-attach/job.py --job_dir /tmp/nvflare/jobs
```

Submit the exported job through that deployment's normal admin connection:

```bash
nvflare job submit -j /tmp/nvflare/jobs/client-api-attach
```

Start the independently managed trainer:

```bash
cd examples/advanced/client-api-attach
python trainer.py --config attach_profile.json
```

The last two steps may be reversed: the trainer may start before or after job
submission, but the site and its provisioned Cell endpoint must already be
available.

`flare.send()` does not return until any lazy result payload has reached
receiver-confirmed terminal success. Keep the trainer process alive through that
confirmation; job shutdown closes only its attach Cell session and never terminates
the trainer process.
