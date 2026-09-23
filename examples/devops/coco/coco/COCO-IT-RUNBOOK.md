# Manual release handoff and launch

For each immutable release, CoCo IT receives exactly one file named
`RELEASE-pod.yaml`. Receive its expected SHA-256 independently from the workload
owner; a checksum sent beside the file does not authenticate it.

Do not edit the manifest. If anything must change, return the request to the
workload owner, who creates a new release and new authorization.

## Inspect, launch, and verify

```bash
sha256sum RELEASE-pod.yaml
./50-launch-handoff.sh RELEASE-pod.yaml EXPECTED_SHA256
./70-verify-running-workload.sh RELEASE-pod.yaml EXPECTED_SHA256
```

The launcher rejects a mutable image tag, wrong registry/runtime, host
namespaces, volumes, interactive I/O, service-account token, privilege,
capabilities, missing digest, or an embedded policy that permits exec/streaming
or policy replacement by default. The service independently refuses the image
key unless attestation and release policy match the authorized image digest,
command vector, init-data digest, CPU/GPU claims, and resource paths.

`70-verify-running-workload.sh` re-checks the live Pod against the same
authenticated manifest: runtime class, image digest, command, and init-data
annotation must match; the Pod must be `Ready`/`Running` with zero restarts;
the intentionally silent workload must expose no log bytes (or log access must
be denied by the guest's `ReadStreamRequest` policy); and `kubectl exec` must be
denied by the guest policy. Other log-access errors do not count as a pass.
The log check samples at most one byte and never prints application output.

The demo is silent by construction. For NVFlare, the provisioning builder
redirects startup stdout/stderr to `/dev/null` **before kit signing**, and the
packager enforces this setting before publication. Standard startup messages and
the inherited console logger therefore remain off the host-visible stream.
Guest-local file logs may still exist in the confidential VM; IT must not retrieve
them. An older noisy image needs a newly built, signed, encrypted and authorized
release, not a skipped check or an edited Pod command. Arbitrary images not
prepared with this silent logging contract are unsupported by this verifier.

Passing stage 70 is a cluster-side prerequisite, not proof of application success
or attestation: CoCo IT controls Kubernetes and its reported status. Before
accepting the deployment, the trusted federation operator must separately follow
[NVFlare registration, attestation and readiness verification](../provision/VERIFY-RUNNING-FEDERATION.md)
over authenticated owner-controlled channels. CoCo IT receives no admin kit.

Allowed operational commands are deliberately limited:

```bash
kubectl get pod -n default RELEASE -o wide
kubectl describe pod -n default RELEASE
kubectl delete pod -n default RELEASE
```

Use stage 70's bounded, non-displaying probe for log verification rather than
printing or collecting application logs. If any application output is visible,
stop verification and notify the workload
owner without copying its contents into tickets or support bundles. A zero-byte
sample is not proof that no earlier or future leak exists; guest stream-policy
rules do not retroactively erase emitted data. Secrets must never be logged.
Except for stage 70's expected-denial test, do not use `kubectl exec`, `attach`, `cp`, debug containers,
`port-forward`, node shell tooling, memory inspection, or guest-agent policy
replacement. The generated Kata policy is designed to reject interactive guest
operations, but the operational prohibition remains part of the release
contract.

IT must not receive the `trusted-service/` handoff, registry publisher
credential, build context, plaintext image, image key, Cosign private key, KBS
admin token, service private keys, or RVPS administration.
