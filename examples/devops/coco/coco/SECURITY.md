# CoCo adversary model

Assume the CoCo owner controls the host OS, Kubernetes API, containerd, storage,
network, scheduling, and all cluster-level metadata. Therefore:

- the owner can read and replace a delivered Pod YAML, but a replacement changes
  its authenticated SHA-256 and, for measured fields, its init-data;
- the owner can fetch only the encrypted OCI artifact from the registry and can
  corrupt or substitute it, but immutable digest selection, signature policy,
  and KBS authorization fail closed;
- the owner can attempt `kubectl exec` or guest-agent requests, but the measured
  Kata agent policy default-denies exec, streaming, and policy replacement;
- the owner cannot obtain the image key unless Trustee accepts the TEE evidence,
  RVPS/reference values, AS policies, GPU/CPU claims, exact image digest,
  process arguments, init-data digest, and requested KBS resource path;
- the owner can always deny service, kill or never schedule the Pod, block
  Trustee, replay public metadata, and observe resource usage and timing.

The owner may still learn information intentionally emitted by the application,
including logs and network traffic not protected by application TLS. Do not put
secrets in the Pod YAML, command line, environment, logs, Kubernetes Secrets,
host mounts, or container layers before encryption.

No ordinary mechanism can make a hostile cluster owner unable to modify bytes.
The guarantee is detection and refusal: modified security-relevant inputs do not
receive a decryption key or an allowed guest execution path. Availability and
traffic-analysis resistance are out of scope.
