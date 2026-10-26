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
