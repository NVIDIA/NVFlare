# `coco` IT runbook

IT receives one file named `<release>-pod.yaml`. It contains no registry
credential, image key, signing private key, KBS admin credential, or application
secret. Obtain the expected SHA-256 through a separate authenticated channel.

Do not edit the manifest. If anything must change, return the request to the
workload owner; they must generate a new immutable release.

This handoff supports SNP or TDX, with or without one confidential NVIDIA GPU.
CoCo IT first installs the matching runtime using the
[cluster guide](../coco/README.md). CPU-only releases use `kata-qemu-snp` or
`kata-qemu-tdx`, allocate no GPU, and do not require NVIDIA hardware or GPU
Operator. GPU releases use the corresponding `kata-qemu-nvidia-gpu-*` runtime
and require one confidential pGPU. Never change a received manifest's runtime
or GPU allocation; those settings are bound to its approved profile and policy.

## Inspect, launch, and verify

Run these from the `coco` kit on the CoCo cluster machine, not from `admin`:

```bash
sha256sum <release>-pod.yaml
./50-launch-handoff.sh <release>-pod.yaml EXPECTED_SHA256
./70-verify-running-workload.sh <release>-pod.yaml EXPECTED_SHA256
```

Compare the first command's digest exactly with the independently authenticated
value before running `50-launch-handoff.sh`. `50-launch-handoff.sh` rejects a
mutable image tag, wrong registry/runtime, host namespaces, volumes,
interactive I/O, service-account token, privilege, capabilities, missing
digest, or an embedded policy that permits exec/streaming or policy
replacement by default; it performs a dry run and requires explicit
confirmation before applying. KBS independently checks the attested InitData
digest, selected CPU claims (plus GPU claims for a GPU release), and requested
resource path against the release authorization. Its image-digest and command
checks compare policy-derived metadata bound into InitData, not a direct
observation of the image actually pulled/decrypted or the running process.

Digest-pinned pulls, signature verification, encrypted layers, and guest-policy
checks still apply, but exact-image authorization has an unresolved coverage
limitation: another image signed by the accepted key in the permitted repository
has not been shown to be rejected solely because its digest differs while
InitData remains unchanged. Do not assume that every image substitution is
denied. See the architecture's
[image authorization analysis](https://nvflare.readthedocs.io/en/2.9/user_guide/confidential_computing/coco_security_architecture.html#image-changes-and-the-current-authorization-limitation).

`70-verify-running-workload.sh` re-checks the live Pod against the same
authenticated manifest: runtime class, image digest, command, and init-data
annotation must match; the Pod must be `Ready`/`Running` with zero restarts;
the intentionally silent workload must expose no log bytes to its bounded probe
(or log access must be denied by the guest's `ReadStreamRequest` policy); and
`kubectl exec` must be denied by the guest policy. Other log-access errors do not
count as a pass. The log check samples at most one byte and never prints
application output. Do not launch with a bare `kubectl apply` — it skips these
checks.

Passing stage 70 is a cluster-side prerequisite, not proof of application success
or attestation: CoCo IT controls Kubernetes and its reported status. Before
accepting the deployment, the trusted federation operator must separately follow
[NVFlare registration, attestation and readiness verification](../provision/VERIFY-RUNNING-FEDERATION.md)
over authenticated owner-controlled channels. CoCo IT receives no admin kit.

Allowed operational commands are deliberately limited:

```bash
kubectl get pod -n default <release> -o wide
kubectl describe pod -n default <release>
kubectl delete pod -n default <release>
```

Use stage 70's bounded, non-displaying probe for log verification rather than
printing or collecting application logs. The demo is silent by construction;
NVFlare provisioning redirects startup stdout/stderr to `/dev/null` before kit
signing, and packaging enforces that setting. A noisy image requires a newly
built, signed, encrypted and authorized release, not a skipped check or an
edited Pod command. Guest-local file logs may still exist; IT must not retrieve
them. Arbitrary images without this silent logging contract are unsupported by
the verifier.

If any application output is visible, stop verification and notify the workload
owner without copying its contents into tickets or support bundles. A zero-byte
sample is not proof that no earlier or future leak exists; guest stream-policy
rules do not retroactively erase emitted data. Secrets must never be logged.

The application reports success to a workload-owner endpoint using its own mTLS
protocol. IT does not need SSH, `kubectl exec`, attach, copy, or an interactive
shell. Those operations should fail under the embedded Kata agent policy.

IT must not receive or request the trusted-service handoff, registry publisher
credential, build context, plaintext image, image key, Cosign private key, KBS
admin token, Trustee private key, or RVPS enrollment access.

The cluster remains able to inspect public Pod and image metadata and to stop or
deny service to the Pod. These are explicit limitations of the threat model.
