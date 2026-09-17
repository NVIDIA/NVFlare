# NRAS claim-shape fixture

`nras_gpu_v3.json` is synthetic test data, authored independently of the generated
policy. It follows the detached GPU claim layout documented in NVIDIA's
[Hopper NRAS example](https://docs.nvidia.com/attestation/quick-start-guide/latest/attestation-examples/hopper_single_gpu.html#decoded-nras-token)
and the current [version 3.0 claims guide](https://docs.nvidia.com/attestation/advanced-documentation/latest/claims-guide/gpu_claims.html#version-3-0).

The version belongs to the signed overall token; neither `x-nvidia-ver` nor
`x-nvidia-device-type` is required in the detached token. The fixture includes
the current OCSP freshness/response-validity fields required by this profile.
Older illustrative responses lacking those security fields intentionally fail
the strict policy. Tests inject fresh timestamps, issuer and nonce, then sign
both tokens using disposable keys. This fixture is not hardware evidence.
