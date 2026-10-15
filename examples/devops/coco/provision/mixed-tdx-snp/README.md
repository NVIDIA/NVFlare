# Ordinary server, TDX CPU-only client and SNP+GPU client

This is a complete provisioning starting point for the mixed functional workflow:

| Participant | Deployment | Required attestation |
| --- | --- | --- |
| `server.example.com` | Ordinary trusted server, beside trusted provisioning/admin | Verifies both clients; issues no CC proof |
| `site-1` | TDX CPU-only Kata guest | TDX CPU, no GPU appraisal |
| `site-2` | SNP+NVIDIA GPU Kata guest | SNP CPU and NVIDIA GPU appraisal |
| `admin@example.com` | Ordinary trusted administration kit | FL administration over authenticated transport |

There is no observation client. The ordinary server's generated CCManager and
`trustee_authorizer` verify the two workload clients. Both clients also receive
the same complete required-participant mapping and verifier constraints.

A real mixed functional run passed on 2026-10-04: both encrypted clients
registered, periodic peer verification succeeded, and the finite job returned
nonce-bound values 3 and 7, aggregate 10, with zero errors. The job performs
CPU arithmetic. GPU appraisal is required for `site-2`; GPU training/performance,
TDX+GPU, protected-server, renewal and full security qualification are outside
that recorded pass. See [runtime validation status](../../RUNTIME-VARIANTS.md#support-status-and-hardware-validation).

## Prepare the trusted inputs

Follow the [two-owner deployment guide](../../FL-DEPLOYMENT.md) and
[configuration procedure](../../CONFIGURATION.md). First approve and install
independent platform references, authenticated TLS and the persistent AS
signing public key. Rehearse the actual guest-local AA token API. Prepare a
TDX CPU-only admin profile (`kata-qemu-tdx`) and a separate SNP+GPU profile
(`kata-qemu-nvidia-gpu-snp`, one `nvidia.com/pgpu`). A RuntimeClass name alone
is not evidence of successful attestation.

Keep builds, plaintext signed kits and image keys on trusted storage. Copy
this source example into a new private deployment directory; do not add real
configuration or generated artifacts to this public directory. Its relative
paths assume:

```text
/private/deployment/
  admin/                       # reviewed runner and common admin tools
  admin-tdx-cpu/platform.env    # independently approved TDX profile/admin kit
  admin-snp-gpu/platform.env    # independently approved SNP+GPU profile/admin kit
  provision/mixed-tdx-snp/      # private copy of this example
```

Each `platform.env` lives with that kit's scripts and authenticated v4 launch
contract, public certificates and private publisher inputs. Assembled kits
run independently; see [role-kit assembly](../../README.md#assemble-self-contained-role-kits).
Absolute runner/platform paths also work when arranging kits differently.

Before provisioning:

1. Replace the project name with a unique deployment audience and set actual
   server/admin identities in [project.yaml](project.yaml). The server's DNS
   endpoint must be directly reachable from both guests; the default learning
   and admin ports are 8002 and 8003.
2. Configure the shared Trustee, registry, credentials, and trusted build
   command once in [cc_project.yml](cc_project.yml). Select distinct, fresh
   `release_name` values and repositories in
   [cc_site-1.yml](cc_site-1.yml) and [cc_site-2.yml](cc_site-2.yml). Authenticate
   the AS public PEM and put it at `trustee-as-public.pem` beside these files.
   It is different from the Trustee TLS certificate and administration key.
3. Keep one complete `workload_constraints` map in `cc_project.yml`. `site-1`
   selects `gpu_tee: none`; `site-2` selects `gpu_tee: nvidia_cc`.
   Provisioning derives the verifier's internal `gpu_required` claim from those
   values, so do not repeat it in the constraints. Add approved MRTD/RTMR or SNP
   measurement constraints when your policy requires them. KBS separately
   enforces approved complete platform tuples and final release InitData.
   Do not place final InitData inside its own image's CC config: that creates
   a circular image/policy dependency.
4. Replace both deliberately non-runnable Dockerfile bases with reviewed
   digest-pinned application bases containing the same NVFlare revision as
   provisioning. CPU arithmetic requires no CUDA application dependency;
   applications using the GPU must bake their own reviewed GPU libraries.
5. Bake the finite application's reviewed code before image approval. From
   the checkout, copy [tdx_acceptance.py](../../acceptance/application/tdx_acceptance.py)
   into both private `site-1/` and `site-2/` build contexts. For example, from
   the private example directory with `CHECKOUT` set to the reviewed checkout:

   ```bash
   cp "$CHECKOUT/examples/devops/coco/acceptance/application/tdx_acceptance.py" site-1/
   cp "$CHECKOUT/examples/devops/coco/acceptance/application/tdx_acceptance.py" site-2/
   ```

   Install the identical module in a private directory on the ordinary server,
   and include that directory in its `PYTHONPATH` before starting NVFlare.
   The module's historical filename is shared by both TEE clients. The two
   Dockerfiles use `COPY --chown=65532:65532` so private source retained with
   mode 0600 is readable by the approved guest UID/GID. Do not copy another
   participant's credentials into either application context.

Keep the default 120-second verification interval, 300-second maximum EAR age,
300-second registration budget, 30-second refresh budget and 45-second peer
request timeout consistent across the project. Periodic validation allows a
fixed 600-second initial grace for missing registrations only; invalid proofs
and job submission still fail closed, and peer/proof collection has its own
budgets. AS token lifetime is configured
on secure services; the participant setting does not change it. Coordinate
launch so both clients can register and provide proofs before the first
required validation round. A missing required participant must not be treated
as an optional member.

## Provision, authorize and launch

On the trusted provisioning node, activate the matching NVFlare environment,
then run from the private example directory:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
nvflare provision -p project.yaml
```

The public [ValidationBuilder](validation_builder.py) grants only the two
reviewed application classes on the ordinary server during trusted provisioning.
The normal CC builder grants those classes on the protected clients and
installs the pinned AS key, shared audience, required sites and per-site
constraints. Keep the supplied Workspace → StaticFile → Cert → CC →
Validation → Signature builder order. Review every generated kit/configuration
privately. Protected kits are signed; the ordinary server kit uses the normal
mTLS trust model and must remain on trusted storage. The server has no issuer;
`cc_enabled_sites` and
`required_site_verifier_ids` cover exactly `site-1` and `site-2`, each mapped to
`trustee_authorizer`.

The packager builds each client separately, includes only its own signed kit,
generates its own encryption key, encrypts/signs its image, and publishes
ciphertext and signatures to the authenticated secure-services registry.
The [image runner](../../admin/build_coco_image.sh) requires plaintext-image
review before continuing. The ordinary server gets a startup kit, not a
confidential-container image.

Deliver the two private `trusted-service/` release bundles directly to the
secure-services administrator. They contain the image keys, policies and
signature-verification public keys. Install each with
[service stage 12](../../service/TRUSTED-HANDOFF-RUNBOOK.md), using independently
authenticated manifest hashes and merging into the complete default-deny
resource policy. The registry never receives the image decryption keys.

After both releases are authorized, start the ordinary server from its private
kit, with the reviewed application module on its `PYTHONPATH`. Deliver only
its own generated Pod YAML and authenticated hash to each compute operator.
Run [cluster stages 50 and 70](../../coco/README.md) for the corresponding
handoff: stage 50 applies the exact Pod using `kubectl apply -f`; stage 70
checks the actual workload/profile and zero application-log output. Do not
modify the Pod command, InitData, image, resources or security context.

The guests pull directly from the TLS registry and contact Trustee over TLS;
they connect directly to the NVFlare server over authenticated FL transport.
For a registry on port 5000 and Trustee on port 8443, configure those endpoints
in the trusted service/admin inputs before generating InitData. The control
workstation and build node are not relays between guests and secure services.

## Submit the baked finite job

Generate a fresh nonce and a **new private** JSON-only job directory on the
trusted admin node. With `CHECKOUT` set to the same reviewed checkout:

```bash
python3 "$CHECKOUT/examples/devops/coco/acceptance/application/generate_job.py" \
  --output /private/jobs/mixed-validation-run \
  --nonce "$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
```

The generator references the already installed `AcceptanceController` and
`AcceptanceExecutor`. It uploads no Python module or other executable BYOC
code. The job requires `mandatory_clients: [site-1, site-2]` and `min_clients: 2`;
its controller targets both clients and checks authenticated names, nonce,
distinct expected values and aggregate. Do not reuse a previous run's nonce
or accept only `FINISHED:COMPLETED` without checking the result artifact.

Follow [trusted federation verification](../VERIFY-RUNNING-FEDERATION.md):
set `EXPECTED_CLIENTS=site-1,site-2`, `REQUIRED_CC_SITES=site-1,site-2`, and
`VALIDATION_JOB` to the new job directory. Confirm exact registration, the
ordinary server's newly observed successful periodic CCManager rounds, and
submit through the private authenticated admin kit. Allow at most 600 seconds
for registration and at most 600 seconds for job completion.

Download results through that same authenticated admin API and inspect
`tdx_acceptance_result.json`: require `status: passed`, the submitted nonce and
job ID, `values: {site-1: 3, site-2: 7}`, `aggregate: 10`, and an empty `errors`
list. Keep private evidence, kit signatures and exact installed-policy/release
hashes with the run's revisions. Secure-services log observations are useful
diagnostics but do not cryptographically correlate separate requests.

Stop participants through normal authenticated NVFlare administration, then
remove only this run's reviewed Pods through their owning clusters. The finite
job does not qualify long-running renewal, guest-policy confidentiality or
hardware denial cases; run those separately before full security acceptance.
