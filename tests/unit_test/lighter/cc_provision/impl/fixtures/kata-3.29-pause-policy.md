# Pinned genpolicy pause fixture

`kata-3.29-pause-policy.json` contains an unmodified sandbox container entry
and the relevant `common.default_caps` and `cluster_config.pause_container_image`
values extracted from real Kata 3.29.0 genpolicy output on Linux amd64.
Captured on 2026-09-21; no container or Pod was launched.

Provenance:

- [Official tools archive](https://github.com/kata-containers/kata-containers/releases/download/3.29.0/kata-tools-static-3.29.0-amd64.tar.zst), SHA-256
  `d7b2b5d846fce3cd8136a28b7626d7b543bfcab5edea2bf8b37591af122f63fb`
  (verified against `admin/platform.env.example` before extraction).
- Extracted `opt/kata/bin/genpolicy` SHA-256:
  `35f3539409ff4dfcbf14a114d182acb15dc5b5ed59d4c3ed880f2d19cc5b371f`.
- Both rules and settings were the archive's unmodified defaults.
- Public pause image: `mcr.microsoft.com/oss/kubernetes/pause:3.6`.
  Its manifest-list digest at capture was
  `sha256:b4b669f27933146227c9180398f99d8b3100637e4a0a1ccf804f8b12f4b9b8df`.
  This records provenance; it does not introduce a new deployment image pin.

## Reproduce

In a new temporary directory, extract the verified archive and save this as
`pod.yaml` (genpolicy replaces it with annotated YAML):

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pause-profile-fixture
spec:
  runtimeClassName: kata-qemu-nvidia-gpu-snp
  containers:
    - name: app
      image: mcr.microsoft.com/oss/kubernetes/pause:3.6
      command: ["/pause"]
      securityContext:
        privileged: false
        allowPrivilegeEscalation: false
        runAsUser: 65532
        runAsGroup: 65532
        readOnlyRootFilesystem: false
        capabilities:
          drop: ["ALL"]
        seccompProfile:
          type: RuntimeDefault
```

Run without private Docker credentials:

```bash
DOCKER_CONFIG="$PWD/empty-docker-config" timeout 90 ./opt/kata/bin/genpolicy \
  --rego-rules-path ./opt/kata/share/defaults/kata-containers/rules.rego \
  --json-settings-path ./opt/kata/share/defaults/kata-containers/genpolicy-settings.json \
  --yaml-file ./pod.yaml
```

Decode the generated `io.katacontainers.config.hypervisor.cc_init_data` using
`workload_security.decode_initdata()`, read `data["policy.rego"]`, and JSON-decode
the suffix after `policy_data := `. Select the container whose OCI annotation
`io.kubernetes.cri.container-type` is `sandbox`, plus the two settings above.
No field in that container entry is normalized or removed for this fixture.

The captured sandbox user is `UID: 65535`, `GID: 65535`,
`AdditionalGids: [65535]`, `Username: ""`. Before the fix this exact output failed
with `ValueError: unexpected pause supplementary groups`.

Tests combine this sandbox entry with a synthetic application and the existing
reviewed hardened-rules fixture. The captured OCI version is the stock `1.1.0`;
stage 30 sets `1.3.0` before generation, which does not change the pause user.
This is generator/validator regression evidence, not hardware-attestation or
full signed/encrypted release-chain validation. Test runs consume the local
JSON fixture offline and do not download tools or contact a registry.
