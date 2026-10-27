# 2. Set values in the configured Trustee

KBS, AS, RVPS, TLS, and appraisal policies already exist. These helpers update
reference values and workload resources; they do not deploy services or replace
the configured CPU/GPU appraisal policies.

The policies must consume the reference IDs exported by capture: SNP launch
measurement and four minimum reported-TCB fields, or complete TDX profile tuples.
GPU policy uses `allowed_driver_versions` and `allowed_vbios_versions`.
The signed EAR must contain the required successful CPU/GPU trust vectors.
Project participants select the single Trustee object in `cc_project.yml`,
as described in the [common project configuration](https://nvflare.readthedocs.io/en/2.9/user_guide/confidential_computing/deployment.html#common-project-configuration).
`service/platform.env` contains separate administration credentials/tool inputs.
The existing service is managed externally; see [upstream KBS configuration](https://github.com/confidential-containers/trustee/blob/v0.22.0/kbs/docs/config.md).
The [CVM Trustee guide](../../../nvflare/lighter/cc/image_builder/TRUSTEE_GUIDE.md)
covers CVM-specific vault policies and is not an extra CoCo installation step.

The optional Rego files in `service/policies/` describe that compatibility contract.

## Configure existing service access

From the example directory, create the administration configuration:

~~~bash
cp service/platform.env.example service/platform.env
chmod 0600 service/platform.env
editor service/platform.env
~~~

Set the following values in that file:

~~~bash
KBS_URL=https://trustee.example.com
TRUSTEE_PUBLIC_CERT="$HOME/coco-private/trustee-ca.crt"
ADMIN_TOKEN="$HOME/coco-private/kbs-admin-token"
KBS_CLIENT=/usr/local/bin/kbs-client
~~~

Keep the token private. The helpers use the received CA/key/configuration files
directly and verify HTTPS with the configured CA.

## Install and read back CPU references

~~~bash
VALUES="$HOME/coco-private/new-capture/platform-reference-values.json"
bash service/02-install-platform-reference-values.sh "$VALUES"
~~~

The installer validates, writes, and reads back the supplied references.
Use `--verify-only` with either reference helper to check without writing.
SNP and TDX sets remain independent. A replacement measurement allowlist omits
previous values unless they are included in the supplied complete list.
Reference changes do not recover image keys already released.

## Set GPU references when required

Use the configured expected driver/VBIOS versions in the private GPU JSON:

~~~json
{
  "schema": "coco-gpu-reference-values/v1",
  "allowed_driver_versions": ["expected-driver-version"],
  "allowed_vbios_versions": ["expected-vbios-version"]
}
~~~

~~~bash
GPU_VALUES="$HOME/coco-private/gpu-reference-values.json"
bash service/02g-install-gpu-reference-values.sh "$GPU_VALUES"
~~~

Skip GPU values for CPU-only releases. A failed required GPU appraisal must
not authorize a GPU release through CPU-only fallback.

## After provisioning: register each workload release

[Provisioning](provision/README.md) generates a private `trusted-service/` handoff
containing the image key, signing public key, image policy, release authorization,
resource-policy fragment, and checksum manifest.

For release registration, also set `CURRENT_RESOURCE_POLICY_FILE` in
`service/platform.env` to a current complete export of the existing KBS resource
policy. The API cannot retrieve its bytes. Use the same canonical export path
for successive installations; the installer updates it after confirmed writes.

**Reusing a release name with a stale or recreated policy export is unsupported.**
Duplicate-release checks use the local export; the installer does not check
Trustee for existing release resources. If that export omits an existing release,
registration can overwrite its image key, signing public key, and image policy,
disrupting workloads that use that release. Installing an incomplete policy can
also remove authorization for unrelated releases. Use a unique name for each new
release and retain the complete current export and any pending recovery files.
If these files are lost or stale, reconcile them with the active Trustee
configuration before registering another release.

The export must include the full current policy and all unrelated existing rules,
and must already use Rego v1 syntax with an explicit `import rego.v1` line.
**Rego v0 is unsupported and must not be used.** The installer rejects a missing
v1 import before uploading resources; this is not a complete Rego syntax check.
If the existing service uses Rego v0, migrate and validate that policy externally
before exporting it and registering a handoff. The installer merges Rego v1
release rules; it does not translate legacy policy syntax.

~~~bash
HANDOFF="$HOME/coco-private/releases/site-1/trusted-service"
bash service/12-install-trusted-service-handoff.sh "$HANDOFF"
~~~

The installer consumes the directory directly, validates the release, installs
its resources, and merges its rule into the complete policy while retaining
unrelated releases. Success records that the resource uploads and policy update
were accepted by the API; it does not read back release-resource bytes or the
active policy. Reference readback above is a separate check. Verify actual key
release and NVFlare operation after launch. Deliver image keys only to Trustee
administration.

An uncertain policy write leaves pending recovery files beside the canonical
export. Retry the same handoff after resolving the cause; another release is
blocked until the pending update is reconciled. Launch only after registration
succeeds, following [Pod launch](LAUNCH.md).

## Optional handoff digest check

If you already have an independently known SHA-256 for the intended
`SHA256SUMS` file, pass it as the second argument:

~~~bash
bash service/12-install-trusted-service-handoff.sh "$HANDOFF" "$EXPECTED_MANIFEST_SHA256"
~~~

This check is optional. Bundled checksums detect inconsistent payloads, not
sender identity. Replacing the entire handoff and its checksum manifest can
authorize a different workload; supplying the intended handoff remains the
trusted administrator's responsibility.
