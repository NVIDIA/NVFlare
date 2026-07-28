# Client API Attach Mode

This example runs the ordinary Client API loop in an externally owned process.
The trainer may start before or after the NVFlare job; both sides rendezvous on
`attach_id=numpy_trainer`.

The included profile is a same-host, cleartext example. Replace `connect_url` with
the existing Cell endpoint exported by `site-1`. For a provisioned secure site, use
the secure endpoint and add:

```json
{
  "connection_security": "mtls",
  "ca_cert": "/absolute/path/to/site-1/startup/rootCA.pem"
}
```

Keep `client.crt` and `client.key` beside `rootCA.pem`; Cell discovers them using
the same credential path as `IPCAgent`. Bare-CA one-way TLS
(`connection_security=tls`) is not supported. A cleartext non-loopback endpoint
is rejected unless the job explicitly sets `allow_insecure_attach=True`.

Attach itself is driver-neutral. To use the shared-filesystem driver, set
`connect_url` to the provisioned site endpoint such as
`shared-file://0/absolute/shared/directory` and omit the TLS/CA fields. The
shared-file driver enforces its own path and permission policy.

The commands below use a one-client local POC system. The bundled cleartext profile
expects the trainer and `site-1` to run on the same host and the site's Cell endpoint
to be `grpc://127.0.0.1:8002`.

Install this repository, prepare the POC, and export the job:

```bash
python -m pip install -e .
python -m pip install -r examples/advanced/client-api-attach/requirements.txt
nvflare poc prepare -n 1
nvflare poc start
python examples/advanced/client-api-attach/job.py --job_dir /tmp/nvflare/jobs
```

Start the trainer in a second terminal:

```bash
cd examples/advanced/client-api-attach
python trainer.py --config attach_profile.json
```

Submit the exported job in a third terminal:

```bash
nvflare job submit -j /tmp/nvflare/jobs/client-api-attach
```

The trainer may start before or after job submission, but the site must already be
running so the trainer can connect to its Cell endpoint.

`flare.send()` does not return until any lazy result payload has reached
receiver-confirmed terminal success. Keep the trainer process alive through that
confirmation; job shutdown closes only its attach Cell session and never terminates
the trainer process.
