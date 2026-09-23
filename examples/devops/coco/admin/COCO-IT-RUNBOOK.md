# `coco` IT runbook

IT receives one file named `<release>-pod.yaml`. It contains no registry
credential, image key, signing private key, KBS admin credential, or application
secret. Obtain the expected SHA-256 through a separate authenticated channel.

Do not edit the manifest. If anything must change, return the request to the
workload owner; they must generate a new immutable release.

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
confirmation before applying. The service independently refuses the image key
unless attestation and release policy match the authorized image digest,
command vector, init-data digest, CPU/GPU claims, and resource paths.

`70-verify-running-workload.sh` re-checks the live Pod against the same
authenticated manifest: runtime class, image digest, command, and init-data
annotation must match; the Pod must be `Ready`/`Running` with zero restarts;
no log bytes may have leaked; and `kubectl exec` must be denied by the guest
policy. Do not consider the release launched until this step passes. Do not
launch with a bare `kubectl apply` — it skips these checks.

Allowed operational commands are deliberately limited:

```bash
kubectl get pod -n default <release> -o wide
kubectl describe pod -n default <release>
kubectl logs -n default <release>
kubectl delete pod -n default <release>
```

The application reports success to a workload-owner endpoint using its own mTLS
protocol. IT does not need SSH, `kubectl exec`, attach, copy, or an interactive
shell. Those operations should fail under the embedded Kata agent policy.

IT must not receive or request the trusted-service handoff, registry publisher
credential, build context, plaintext image, image key, Cosign private key, KBS
admin token, Trustee private key, or RVPS enrollment access.

The cluster remains able to inspect public Pod and image metadata and to stop or
deny service to the Pod. These are explicit limitations of the threat model.
