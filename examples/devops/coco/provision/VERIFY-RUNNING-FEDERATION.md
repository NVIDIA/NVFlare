# Verify the running federation from the trusted side

CoCo IT's stage 70 checks Pod readiness and the absence of host-readable
container output. Neither a Kubernetes `Ready` condition nor a silent container
proves that NVFlare registered, ran a job, or passed CC validation. Do not enable
stdout, `kubectl exec`, SSH, or log copying from protected Pods to diagnose this.

The workload owner and FL project administrator perform the checks below from
trusted machines. The administrator's ordinary mTLS startup kit is private: it
must not be delivered to CoCo IT. Use the same NVFlare version as provisioning.
The examples use the repository's `.venv/bin/python`; substitute the equivalent
trusted NVFlare environment when using an installed package.

## 1. Review the required attestation configuration

Before releasing the signed images/kits, review each participant's generated
`local/cc_manager__p_resources.json` and `local/coco_authorizer__p_resources.json`
in the **private provisioning workspace**. Never
edit a kit after signing or use configuration returned by CoCo IT as approval.
For an ordinary trusted server, also confirm that its deployed kit is the
reviewed kit and that no local override disables its CC components.

For example, check a generated configuration for one protected client:

```bash
export REVIEWED_LOCAL=/private/review/server/local
export REQUIRED_CC_SITES=site-1
# With a protected server as well, use REQUIRED_CC_SITES=server,site-1.
.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

required = set(os.environ["REQUIRED_CC_SITES"].split(","))
local = Path(os.environ["REVIEWED_LOCAL"])
components = {}
for name in ("cc_manager", "coco_authorizer"):
    config = json.loads((local / (name + "__p_resources.json")).read_text())
    for item in config["components"]:
        if item["id"] in components:
            raise SystemExit("Duplicate component in reviewed files")
        components[item["id"]] = item
manager = components["cc_manager"]
if manager["path"] != "nvflare.app_opt.confidential_computing.cc_manager.CCManager":
    raise SystemExit("Unexpected CCManager implementation")
args = manager["args"]
expected_map = {site: ["coco_authorizer"] for site in required}
if (
    not required
    or "" in required
    or set(args["cc_enabled_sites"]) != required
    or args["required_site_verifier_ids"] != expected_map
    or args["cc_verifier_ids"] != ["coco_authorizer"]
    or args["require_site_binding"] is not True
):
    raise SystemExit("Required attestation coverage does not match the approved project")
authorizer = components["coco_authorizer"]
if authorizer["path"] != "nvflare.app_opt.confidential_computing.coco_authorizer.CoCoAuthorizer":
    raise SystemExit("Unexpected CoCoAuthorizer implementation")
print("Required CC participants:", ", ".join(sorted(required)))
print("Configured periodic interval:", args["verify_frequency"], "seconds")
PY
```

This checks the generated structure, not its authenticity. Independently review
the pinned AS public key, project audience, issuer settings, image digest,
InitData/security policy, signature, and secure-services release authorization
as described in [CCMANAGER.md](CCMANAGER.md). The server's CC identity is
`server`, not its certificate DNS name. Repeat the check for every participant
that must enforce these requirements. Ordinary non-CC participants intentionally
do not belong in `REQUIRED_CC_SITES`.

## 2. Check registration and run a reviewed validation job

The FL project administrator runs this on a trusted administrator machine, not
on a CoCo cluster host. The kit path is the admin directory **containing**
`startup/`, not the `startup/` directory itself. Replace the example names with
the provisioned identities.

Prepare a harmless, finite NVFlare validation job **before building images**.
Its reviewed executor/controller code must already be available in the relevant
images and permitted by their class allowlists. CCManager rejects BYOC jobs;
do not upload new executable code or weaken that restriction for this check.
This package does not supply a universal validation job: use the application's
reviewed validation workflow, which must fail unless every expected client
returns the required result. Set its `meta.json` `mandatory_clients` to the
expected client list, `min_clients` to its length, and review its `deploy_map`
and controller target/response requirements. Scheduling every client alone
does not prove the controller actually exercised every client.

```bash
export FL_ADMIN=admin@example.com
export FL_ADMIN_KIT=/private/kits/admin@example.com
export EXPECTED_CLIENTS=site-1
export VALIDATION_JOB=/private/jobs/reviewed-validation
export CHECK_TIMEOUT_SECONDS=600
.venv/bin/python - <<'PY'
import json
import os
import time
from pathlib import Path

from nvflare.fuel.flare_api.flare_api import new_secure_session

expected_list = os.environ["EXPECTED_CLIENTS"].split(",")
expected = set(expected_list)
if not expected or "" in expected or len(expected) != len(expected_list):
    raise SystemExit("EXPECTED_CLIENTS must contain distinct provisioned client names")
job = Path(os.environ["VALIDATION_JOB"]).resolve(strict=True)
job_meta = json.loads((job / "meta.json").read_text())
if (
    set(job_meta.get("mandatory_clients", [])) != expected
    or job_meta.get("min_clients") != len(expected)
):
    raise SystemExit("Validation job must require every expected client")
timeout = int(os.environ["CHECK_TIMEOUT_SECONDS"])
if timeout <= 0:
    raise SystemExit("CHECK_TIMEOUT_SECONDS must be positive")
session = new_secure_session(
    username=os.environ["FL_ADMIN"],
    startup_kit_location=os.environ["FL_ADMIN_KIT"],
    timeout=30.0,
    command_timeout=30.0,
    auto_login_max_tries=3,
)
try:
    deadline = time.monotonic() + timeout
    while True:
        info = session.get_system_info()
        registered = {client.name for client in info.client_info}
        if registered == expected:
            print("Registered clients:", ", ".join(sorted(registered)))
            print("Server status:", info.server_info.status)
            break
        if registered - expected:
            raise SystemExit("Unexpected registered clients: " + ", ".join(sorted(registered - expected)))
        if time.monotonic() >= deadline:
            raise SystemExit("Timed out waiting for: " + ", ".join(sorted(expected - registered)))
        time.sleep(5)

    job_id = session.submit_job(str(job))
    print("Validation job ID:", job_id, flush=True)
    deadline = time.monotonic() + timeout
    while True:
        meta = session.get_job_meta(job_id)
        status = meta.get("status", "")
        if status == "FINISHED:COMPLETED":
            print("Reviewed application validation completed:", job_id)
            break
        if status.startswith("FINISHED:"):
            raise SystemExit("Validation job failed: " + status)
        if time.monotonic() >= deadline:
            raise SystemExit("Timed out; inspect this job through the trusted admin API: " + job_id)
        time.sleep(5)
finally:
    session.close()
PY
```

API errors, missing clients, unexpected clients, terminal failures, and timeout
are not success. Timeout does not abort an already submitted job; retain its
printed ID and manage it through the authenticated FL administration interface.
Polling is bounded, with an additional bounded API request/session-close time.

`get_system_info()` reports registration, not cryptographic attestation evidence.
`FINISHED:COMPLETED` is useful only with the reviewed workflow's success
semantics. Together these checks establish authenticated application operation;
they are not an independent substitute for the enforced CC and KBS policies.

## 3. Confirm CC validation without exporting protected logs

CCManager validates required proofs during registration and performs periodic
cross-site validation. It also performs validation before job scheduling **if
its `cross_validation_run_once` flag is still false**. Do not claim that each
submitted job forces a new attestation round: a previous periodic or pre-job
round can already have set that flag.

With an **ordinary trusted server**, its owner can observe the next validation
round in that server's local `log.txt`. This is not a command for CoCo IT or a
protected server. Run it on the trusted server using that server's NVFlare
environment. Start after the required clients have registered. Allow the
configured interval, initial jitter, and token collection/retry time.

```bash
export TRUSTED_SERVER_LOG=/private/server-workspace/log.txt
export CC_WAIT_SECONDS=600
.venv/bin/python - <<'PY'
import os
import time
from pathlib import Path

path = Path(os.environ["TRUSTED_SERVER_LOG"])
timeout = int(os.environ["CC_WAIT_SECONDS"])
if timeout <= 0:
    raise SystemExit("CC_WAIT_SECONDS must be positive")
deadline = time.monotonic() + timeout
with path.open() as stream:
    original = os.fstat(stream.fileno())
    stream.seek(0, 2)
    while time.monotonic() < deadline:
        current = path.stat()
        if (current.st_dev, current.st_ino) != (original.st_dev, original.st_ino) or current.st_size < stream.tell():
            raise SystemExit("Log rotated/truncated; restart this observation rather than accept stale evidence")
        line = stream.readline()
        if not line:
            time.sleep(1)
            continue
        if "CCManager" in line and "Cross-site validation passed" in line:
            print("Observed a new CCManager validation success on the trusted server")
            break
    else:
        raise SystemExit("No new successful validation observed before timeout")
PY
```

The command starts at the current file end; an old success line does not pass.
The observation is meaningful only because this server, its software/configuration,
and its log are trusted. A log string supplied by an adversary is not an
attestation proof. Retain only the minimal result needed for operations; do not
print or distribute JWTs, private keys, or application logs to CoCo IT.

With a **protected server**, leave its stdout/stderr suppression and guest
access restrictions intact. Use the authenticated administrator session and
reviewed application validation workflow above; a trusted participating client
can also verify the server's proof through its provisioned CCManager. There is
currently no dedicated public FLARE API endpoint that returns CCManager's last
successful validation time/proofs. If explicit externally observable fresh CC
status is required, this example does not provide that API: record it as
unverified until an authenticated application-level mechanism is designed and
reviewed. Do not call Pod readiness or a generic job result direct proof of a
fresh attestation round.

Finally, the secure-services owner independently verifies the installed RVPS/AS
references and KBS resource policy through the [secure-services
guide](../service/README.md). CCManager checks peer proofs locally; a CCManager
success does not itself show that a particular image decryption key was newly
released. Keep platform appraisal, resource authorization, and application
readiness as separate verification results.
