# Client API Attach Mode

This example runs the ordinary Client API loop in a trainer process that is
started and owned independently of NVFlare. NVFlare owns the Cell protocol
session but never starts, signals, or reaps the trainer process.

Use Attach only for this ownership model. If NVFlare should launch and own the
trainer process, use `external_process`; if training runs in the Client Job, use
`in_process`. See the
[Client API Attach guide](../../../docs/programming_guide/execution_api_type/client_api_attach.rst)
for the mode decision, configuration responsibilities, and legacy migration
mapping.

Attach keeps the 2.8 `IPCExchanger`/`IPCAgent` trust boundary: the Client Job
(CJ) materializes trainer results, and the trainer receives no site
`AUTH_TOKEN` or signature.

## Network Attach through the CP

Network Attach reuses the site's existing Client Parent (CP) listener:

```text
federation network <-> CP <-> CJ
                       ^
                       |
              externally owned trainer
```

The CP routes Cell messages between the trainer and dynamic CJ without decoding
their payloads. The CJ remains the task/result endpoint and materialization
boundary. No dynamic CJ listener, server certificate, server key, fixed Attach
port, or `listening_host` provisioning is needed.

Start from `attach_profile_network.json`:

```json
{
  "schema_version": 1,
  "execution_mode": "attach",
  "attach_id": "numpy_trainer",
  "site_name": "site-1",
  "connect_url": "tcp://site-1.example.com:8004",
  "connection_security": "clear",
  "secure_mode": true,
  "ca_cert": "/absolute/path/to/site/startup/rootCA.pem",
  "job_wait_timeout": null
}
```

`connect_url` must be the provisioned CP internal listener, not a CJ URL. Keep
`client.crt` and `client.key` beside `rootCA.pem`. `secure_mode=true` uses those
site credentials for Cell authentication even when the CP transport itself is
`connection_security="clear"`, matching the old IPCAgent model.

The trainer identity is stable:

```text
site-1.-client_api_numpy_trainer
```

It can therefore start before the job and bind the dynamic CJ from authenticated
`SESSION_OPEN`. For a relayed site, add its stable `cp_fqcn`; add
`auth_identity` as well when the provisioned site certificate CN differs from
`site_name`.

The site's normal `comm_config.json.internal` controls this route. Do not add a
network `client_api_attach` listener; the backend rejects one.

## Shared-Filesystem Attach

The existing F3 `FileDriver` supports a network-isolated trainer. Only this
CJ-to-trainer connection uses shared-file; CP-to-CJ stays on the normal network.

Add this section to the site-local `comm_config.json`:

```json
{
  "client_api_attach": {
    "scheme": "shared-file",
    "resources": {
      "root_dir": "/absolute/shared/nvflare-client-api-attach",
      "connection_security": "clear"
    }
  }
}
```

The trainer profile names the stable rendezvous directory:

```json
{
  "schema_version": 1,
  "execution_mode": "attach",
  "attach_id": "numpy_trainer",
  "site_name": "site-1",
  "rendezvous_dir": "/absolute/shared/nvflare-client-api-attach",
  "job_wait_timeout": null
}
```

The site/CJ and trainer must see the same absolute path. Keep the path owned by
the dedicated site account and restrict its group to the intended trainer
principals. The backend validates the FileDriver ownership marker, directory and
file permissions, and cross-node claim lock. A world-accessible route is always
rejected.

The trainer may start before or after the job. When the CJ starts, it publishes
the dynamic FileDriver endpoint under `(site_name, attach_id)`. The trainer
needs no network, DNS, CA, or certificate.

FileDriver creates directories and files with group access for the intended
site/trainer principals. The shared filesystem must support coherent atomic
rename and cross-node POSIX advisory locks. Shared-file Attach does not require
`--allow_insecure_attach`; an unsafe path is rejected regardless of that
deprecated compatibility argument.

## Local POC

Prepare a one-client workspace:

```bash
nvflare poc config --pw /tmp/nvflare-attach-poc
nvflare poc prepare -n 1 --force
```

For the simplest POC, configure the shared-file section above in:

```text
/tmp/nvflare-attach-poc/example_project/prod_00/site-1/local/comm_config.json
```

Use the same existing absolute directory in a private copy of
`attach_profile_shared_file.json`, and create it with mode `0770` under a
non-world-writable parent.

The site component policy must allow:

```text
nvflare.app_common.executors.client_api_executor.ClientAPIExecutor
```

Start the POC and install the example dependency:

```bash
nvflare poc start -ex admin@nvidia.com
python -m pip install -r examples/advanced/client-api-attach/requirements.txt
```

Submit the job:

```bash
python examples/advanced/client-api-attach/job.py \
  --startup_kit_location \
  /tmp/nvflare-attach-poc/example_project/prod_00/admin@nvidia.com
```

Start the externally managed trainer in either order:

```bash
cd examples/advanced/client-api-attach
python trainer.py --config attach_profile_shared_file.json
```

For network Attach, replace the placeholders in
`attach_profile_network.json` with the CP route and trainer credential paths,
then run the same command with that profile.

Successful output ends with `Status: FINISHED:COMPLETED`. For this three-round
example, the final model is:

```text
[[4 5 6]
 [7 8 9]]
```

Large trainer results use ordinary FOBS `ViaDownloader` transfer to the CJ.
The CJ materializes the result. Forwarding that concrete result may create a
new CJ-owned `DownloadService` transaction; the trainer reference does not
cross the CJ boundary.
