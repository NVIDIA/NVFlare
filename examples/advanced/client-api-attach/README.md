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
the same credential path as `IPCAgent`. A cleartext non-loopback endpoint is
rejected unless the job explicitly sets `allow_insecure_attach=True`.

Attach itself is driver-neutral. To use the shared-filesystem driver, set
`connect_url` to the provisioned site endpoint such as
`shared-file://0/absolute/shared/directory` and omit the TLS/CA fields. The
shared-file driver enforces its own path and permission policy.

Install this repository and the example dependency, then export and submit the job:

```bash
python -m pip install -e .
python -m pip install -r examples/advanced/client-api-attach/requirements.txt
python examples/advanced/client-api-attach/job.py --job_dir /tmp/nvflare/jobs
```

Start the trainer independently, either before or after job submission:

```bash
cd examples/advanced/client-api-attach
python trainer.py --config attach_profile.json
```

`flare.send()` does not return until any lazy result payload has reached
receiver-confirmed terminal success. Keep the trainer process alive through that
confirmation; job shutdown closes only its attach Cell session and never terminates
the trainer process.
