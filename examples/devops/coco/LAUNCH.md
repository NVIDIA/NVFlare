# 4. Launch the confidential Pod

Start after the Trustee administrator has registered the image key and release
policy using the generated [private handoff](TRUSTEE.md#after-provisioning-register-each-workload-release).
Ordinary server/admin startup kits stay on their trusted machines.

Use the current Kubernetes context, or set standard `KUBECONFIG` to the existing
cluster configuration. The manifest must select an available compatible
RuntimeClass and, for GPU profiles, one allocatable confidential GPU.

## Apply the received manifest

From the example directory:

~~~bash
POD=/path/to/site-1-release-pod.yaml
kubectl config current-context
bash coco/50-launch-handoff.sh "$POD"
kubectl get -f "$POD" -o wide
~~~

Invoking the helper authorizes applying the Pod. It validates the manifest,
performs a server dry run, applies the received file, and waits for readiness.

Keep the generated image digest, command, InitData, resources, and security
context together. Do not change them to work around an attestation failure.

## Check NVFlare startup

Use the normal trusted FL administration connection to confirm the expected
participants registered and their required peer-proof/CC validation succeeds.
Pod readiness alone does not establish successful NVFlare startup.

The guest obtains image resources from Trustee after the configured CPU/GPU
appraisal and exact release authorization pass. Resolve failures through the
existing application and Trustee diagnostics. Keep credentials, image keys,
and sensitive application output away from cluster IT.

## Troubleshoot a failed launch

Start with Kubernetes metadata. For a created Pod, use the same `POD` path as above.
Run these checks in another terminal while the launcher waits, setting `POD`
there to that same path:

~~~bash
kubectl get runtimeclass
kubectl get -f "$POD" -o wide
kubectl describe -f "$POD"
NAMESPACE="$(kubectl get -f "$POD" -o jsonpath='{.metadata.namespace}')"
POD_NAME="$(kubectl get -f "$POD" -o jsonpath='{.metadata.name}')"
kubectl get events -n "$NAMESPACE" \
  --field-selector "involvedObject.kind=Pod,involvedObject.name=$POD_NAME" \
  --sort-by=.metadata.creationTimestamp
~~~

A RuntimeClass object and successful server dry run do not prove that its handler
works on the assigned worker. Use the reported reason to choose the next check:

| Symptom | Next check |
| --- | --- |
| Capture or build fails before producing a Pod | Follow [measurements](MEASUREMENTS.md) or [provisioning](provision/README.md). On the trusted capture/build machine, inspect the command error and any retained `$OUTPUT/` capture records or `$WORK_ROOT/releases/<release>/review/` image-build records. |
| Missing RuntimeClass or `FailedScheduling` | Check `kubectl get runtimeclass` and the Pod events above. Use `kubectl describe nodes` to compare GPU requests with allocatable `nvidia.com/pgpu`, taints, and node readiness. |
| `FailedCreatePodSandBox` or runtime-handler error | Check the assigned worker's existing kubelet/containerd logs. On a systemd worker, use `sudo journalctl -u kubelet -u containerd --since "15 minutes ago" --no-pager`. |
| `ErrImagePull`, `ImagePullBackOff`, or image decryption/signature failure | Check the event's registry/image error and the Pod's digest against the generated release output; check the endpoint and CA/bundle in `provision/cc_project.yml`. For denied image-key access, use the Trustee checks below. |
| Trustee appraisal or resource-policy denial | On the Trustee administration machine, run the reference helpers from [Trustee](TRUSTEE.md) with `--verify-only`. Inspect the existing deployment's KBS/AS/RVPS logging destination for the release's resource paths, CPU/GPU appraisal, and policy decision; compare `release-authorization.json` with `CURRENT_RESOURCE_POLICY_FILE`. |
| Pod is Ready but NVFlare registration or peer proofs fail | On the trusted FL administrator machine, use the [normal admin session](https://nvflare.readthedocs.io/en/2.9/user_guide/data_scientist_guide/flare_api.html) and `sess.get_system_info()`. Check the ordinary trusted server's workspace `log.txt` beside `startup/` for connection and CC validation failures. |

Keep capture/build records and Trustee/NVFlare logs on trusted machines.
Do not enable guest exec or print raw AA token responses, keys, or sensitive
application logs for cluster IT.
