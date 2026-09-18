# NRAS claim-shape fixture

`nras_gpu_v3.json` is synthetic test data, authored independently of the generated
policy. It follows the detached GPU claim layout documented in NVIDIA's
[Hopper NRAS example](https://docs.nvidia.com/attestation/quick-start-guide/latest/attestation-examples/hopper_single_gpu.html#decoded-nras-token)
and the current [version 3.0 claims guide](https://docs.nvidia.com/attestation/advanced-documentation/latest/claims-guide/gpu_claims.html#version-3-0).

The version belongs to the signed overall token; neither `x-nvidia-ver` nor
`x-nvidia-device-type` is required in the detached token. The fixture includes
the current OCSP freshness/response-validity fields required by this profile.
Older illustrative responses lacking those security fields intentionally fail
the strict policy. Policy tests model the claims emitted by upstream Trustee v0.22.0, including
the overall result copied from the signed top-level token. HTTPS tests sign EARs
with disposable AS keys. NVIDIA signature verification is upstream’s responsibility;
this fixture does not exercise NRAS cryptography and is not hardware evidence.

## Genuine SNP / Trustee v0.22.0 EAR

`snp_trustee_v022.json` contains an unmodified, ES256-signed EAR issued by a clean
Trustee v0.22.0 `restful-as`, its disposable public signing key, and provenance.
The input is upstream's public, signed
[SNP evidence fixture](https://github.com/confidential-containers/trustee/blob/512fed65642015b849f38fb13bfdec7806639987/attestation-service/tests/e2e/evidence.json),
whose SHA-256 is `3c7d1e6a3575beb6b7632e2a08123d45de9c9128a357a0bd3f97630ef4b1ac3d`.
Upstream verified the AMD report signature, endorsement chain and TCB and applied
this repository's CPU appraisal policy with isolated test-only RVPS approvals.
The resulting affirming token carries hexadecimal `snp.measurement` and
`init_data`. No production key, raw lab quote, or private machine identity is
included; the token's report data comes from the public upstream fixture.

This is archived evidence, not a fresh challenge response from a live guest:
the capture deliberately omits expected runtime data, as upstream's own fixture
request does. It tests cryptographic parsing and the AS/guest/policy encoding
boundary; it does not replace live KBS freshness or hardware acceptance tests.
Guest tests validate the original token signature at its recorded issuance time.
Rego tests refresh only `iat`/`exp` to exercise current-time policy decisions.

Regenerate from the repository root in an isolated Linux environment:

```sh
git clone --branch v0.22.0 https://github.com/confidential-containers/trustee.git /tmp/trustee
cargo build --locked --release --manifest-path /tmp/trustee/Cargo.toml \
  -p attestation-service --bin restful-as --no-default-features --features restful-bin,snp-verifier
PYTHONPATH=nvflare/lighter/cc/image_builder python3 \
  tests/integration_test/lighter/cc/image_builder/capture_snp_fixture.py /tmp/trustee \
  tests/unit_test/lighter/cc/image_builder/fixtures/snp_trustee_v022.json
```

The capture tool requires a clean pinned checkout, starts an AS bound only to
localhost with temporary storage/keys, verifies the returned EAR, and stops it.
It never alters an existing Trustee deployment. Issuance times, ephemeral public
keys and signatures change on regeneration; the evidence-derived claims do not.
