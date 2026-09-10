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
no log bytes may have leaked; and `kubectl exec` must be denied by the guest
policy. Do not consider the release launched until this step passes.

Allowed operational commands are deliberately limited:

```bash
kubectl get pod -n default RELEASE -o wide
kubectl describe pod -n default RELEASE
kubectl logs -n default RELEASE
kubectl delete pod -n default RELEASE
```

Logs are visible only if the application writes them; secrets must never be
logged. Do not use `kubectl exec`, `attach`, `cp`, debug containers,
`port-forward`, node shell tooling, memory inspection, or guest-agent policy
replacement. The generated Kata policy is designed to reject interactive guest
operations, but the operational prohibition remains part of the release
contract.

IT must not receive the `trusted-service/` handoff, registry publisher
credential, build context, plaintext image, image key, Cosign private key, KBS
admin token, service private keys, or RVPS administration.
