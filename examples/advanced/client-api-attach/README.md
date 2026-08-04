# Client API Attach Mode

This example runs the ordinary Client API loop in an externally owned process.
With the default shared-filesystem profile, the trainer may start before or after
the NVFlare job; both sides rendezvous on `attach_id=numpy_trainer`.

Attach uses a listener owned by the Client Job (CJ). It does not reuse or change
the site's ordinary CP-to-CJ `internal` connection:

```text
federation network <-> CP <-> CJ <-> dedicated Attach listener <-> trainer
```

The site operator configures the dedicated listener in the site-local
`comm_config.json` under `client_api_attach`. The trainer independently receives
either a direct network URL or a shared-filesystem rendezvous directory.

## Shared-Filesystem Attach

The existing F3 `FileDriver` supports a network-isolated trainer. Tasks,
heartbeats, results, lazy-result transfers, and shutdown messages all use the
shared filesystem. The trainer needs no network access, DNS, CA, or certificate.

Add this section to the site's `local/comm_config.json`:

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

This section is independent of `internal`; CP-to-CJ communication may continue
using its default TCP driver, gRPC, mTLS, or another configured driver.

The site/CJ runtime and trainer must see `root_dir` at the same absolute path.
Keep it owned by the dedicated site account and restrict its group to the intended
trainer principals. `FileDriver` creates directories with mode `0770` and files
with mode `0660`, explicitly restoring those group permissions after creation so
a restrictive process umask cannot block a different user in the shared group;
filesystem access is the peer-access boundary.

The checked-in `attach_profile_shared_file.json` names the stable rendezvous
directory, not the driver's dynamic `shared-file://.../lst_<id>` URL:

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

When the CJ starts, it creates its Attach listener and atomically publishes the
dynamic URL under the site/attach-ID rendezvous claim. A trainer started first
waits for that record. The CJ holds a cross-node POSIX advisory lock for its
lifetime, which rejects a concurrent live job using the same `attach_id` and
makes an unlocked crash artifact ineligible for discovery. The shared filesystem
must provide working cross-node advisory locks and coherent atomic rename.

Shared-file Attach does not require `--allow_insecure_attach`. The backend checks
the actual CJ-owned listener, FileDriver ownership marker, connection directory,
root permissions, and rendezvous permissions. An unsafe shared-file path is
rejected even when the insecure network opt-in is set.

## Network Attach

A registered network driver can instead be configured on the dedicated listener:

```json
{
  "client_api_attach": {
    "scheme": "grpcs",
    "resources": {
      "host": "site-1.example.com",
      "port": 8102,
      "connection_security": "mtls"
    }
  }
}
```

The checked-in `attach_profile_network.json` is the corresponding direct-profile
template:

```json
{
  "schema_version": 1,
  "execution_mode": "attach",
  "attach_id": "numpy_trainer",
  "site_name": "site-1",
  "cj_fqcn": "site-1.<job_id>",
  "connect_url": "grpcs://site-1.example.com:8102",
  "connection_security": "mtls",
  "ca_cert": "/absolute/path/to/trainer-credentials/rootCA.pem",
  "job_wait_timeout": null
}
```

Keep `client.crt` and `client.key` beside `rootCA.pem`; Cell discovers them using
the same credential-folder convention as `IPCAgent`. The CJ listener requires its
site-local CA, server certificate, and server key. A default provisioned client
kit does not contain `server.crt`/`server.key`; provision the site with
`listening_host` set (or otherwise install a site-local listener certificate and
key) before configuring the `grpcs` listener. Never distribute the server key to
the trainer.

The direct profile is job-specific: replace `<job_id>` with the submitted job ID.
The trainer FQCN is derived as
`<cj_fqcn>.-client_api_<attach_id>`, so a direct network trainer needs that CJ
identity before constructing its Cell. Unlike shared-file rendezvous, a static
direct profile cannot discover a later job automatically.

Bare-CA one-way TLS (`connection_security=tls`) is rejected. Clear network
listeners—including loopback listeners—require both an explicit
`connection_security=clear` in the profile and
`--allow_insecure_attach` on `job.py`. The flag only acknowledges an unprotected
CJ-to-trainer network route; it does not affect CP-to-CJ communication and must
not be used on an untrusted network.

Changing site-local `comm_config.json` requires restarting the site. A fixed
network port must also be reserved so another concurrent job cannot bind it.
Shared-file rendezvous avoids a fixed port and is preferred for local,
network-isolated trainers.

## Local POC

Prepare a one-client workspace:

```bash
nvflare poc config --pw /tmp/nvflare-attach-poc
nvflare poc prepare -n 1 --force
```

Create
`/tmp/nvflare-attach-poc/example_project/prod_00/site-1/local/comm_config.json`
with a dedicated shared-file Attach listener. Replace `/absolute/shared/...`
below and in a private copy of `attach_profile_shared_file.json` with the same
existing, absolute directory; create it with mode `0770` under a
non-world-writable parent:

```json
{
  "client_api_attach": {
    "scheme": "shared-file",
    "resources": {
      "root_dir": "/absolute/shared/nvflare-client-api-attach",
      "connection_security": "clear",
      "poll_interval": 0.01,
      "max_poll_interval": 0.25
    }
  }
}
```

The checked-in `attach_profile_shared_file.json` uses the same explicit
placeholder so the example never silently trusts a shared `/tmp` location.

The component policy must authorize the exact job components. In each directory,
copy `resources.json.default` to `resources.json` if needed, preserve the existing
`class_allow_list`, and add:

- `site-1/local/resources.json`:
  `nvflare.app_common.executors.client_api_executor.ClientAPIExecutor`

`MetricsArtifactWriter` is already in the default allow-list. Do not replace the
site allow-list with a broad package prefix.

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

In another terminal, start the independently managed trainer:

```bash
cd examples/advanced/client-api-attach
python trainer.py --config attach_profile_shared_file.json
```

The trainer and job commands may be started in either order. `job.py` uses
`ProdEnv`, waits for completion, and prints the downloaded result directory.
Pass `--username` if the startup kit belongs to another administrator.

For network Attach, configure the `client_api_attach` network listener described
above, replace every placeholder in `attach_profile_network.json`, and run:

```bash
python trainer.py --config attach_profile_network.json
```

The direct profile contains the job-specific CJ FQCN, so normally submit the job
and generate the final profile before starting the network trainer. The training
code and job recipe are otherwise identical to the shared-file example.

Successful output ends with `Status: FINISHED:COMPLETED`. For this three-round
example, verify the final model with:

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
`/tmp/nvflare-attach-poc/example_project/prod_00/site-1/<JOB_ID>/log.txt`.

`flare.send()` returns only after lazy result payloads reach receiver-confirmed
terminal success. Keep the trainer alive through that confirmation. Job teardown
closes the Attach Cell session but never terminates the externally owned process.
