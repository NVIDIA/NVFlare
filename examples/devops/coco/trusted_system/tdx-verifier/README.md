# Pinned TDX evidence verifier

`build.sh OUTPUT_DIR` builds the trusted-host verifier and installs
`OUTPUT_DIR/tdx-evidence-verify`. Docker must be usable directly or via
passwordless `sudo`. The resulting wrapper embeds the immutable Docker image
ID; stage 06 records the wrapper's SHA256. Do not replace that wrapper or its
container image after approval.

```bash
./tdx-verifier/build.sh /absolute/path/to/platform-tools/tdx-verifier
/absolute/path/to/platform-tools/tdx-verifier/tdx-evidence-verify \
  evidence.json request-data.bin rehearsal-initdata.toml > verified-claims.json
```

If the trusted build host reaches Intel's package server but Docker's default
build network cannot resolve it, use `TDX_VERIFIER_BUILD_NETWORK=host` for
`build.sh`. This changes build-time networking only; the verifier wrapper still
uses an unprivileged, read-only container with Docker's default runtime network.
The build accepts only `default` or `host`, and retains all package, signing-key,
source revision, and Cargo dependency checks.

The verifier is built from Trustee revision
`338610fbfed57b66c61a8a3a60e0e4386bdce793`, with the `tdx-verifier` feature only.
Its Rust dependencies are locked in `Cargo.lock`, and its compiler container is
digest pinned. The manifest explicitly enables `reqwest-middleware` on the
already locked `http-cache-reqwest` dependency because the pinned Trustee
disables its default features. The native Intel QVL packages are pinned to DCAP 1.27
(`1.27.100.1-noble1`); package authentication uses the **full** Intel signing-key
fingerprint `150434D1488BF80308B69398E5C7F0FA1C6C6C3C` published in
[Intel's SGX README](https://github.com/intel/confidential-computing.sgx#downloads).
The package version follows the pinned DCAP tag's
[`se_version.h`](https://github.com/intel/confidential-computing.tee.dcap/blob/DCAP_1.27/QuoteGeneration/common/inc/internal/se_version.h)
and [Debian package build](https://github.com/intel/confidential-computing.tee.dcap/blob/DCAP_1.27/QuoteGeneration/installer/linux/deb/libsgx-dcap-quote-verify/build.sh).
An unavailable pinned package or signing-key change is a build failure, not a
reason to install an unreviewed version or disable repository authentication.

Verification requires outbound HTTPS to Intel PCS and an accurate trusted host
clock. It does **not** require SGX/TDX devices or privileged containers. Inputs
are mounted read-only; no Docker socket is exposed to the verifier.

The collector receives a newly generated nonce as exactly 64 lowercase ASCII
hexadecimal bytes (`secrets.token_hex(32).encode("ascii")`). Kata's guest-local
`/aa/evidence?runtime_data=...` endpoint copies these UTF8 bytes directly to
REPORT_DATA. It does not base64-decode or hash the parameter. The verifier
compares all 64 bytes, verifies SHA256 of the exact InitData TOML against the
zero-padded 48-byte MRCONFIGID, and requires nonempty CCEL.

The pinned Trustee verifier validates the Intel quote signature/collateral and
replays the combined firmware/AA event log against **all four RTMRs**. This
wrapper additionally requires non-debug mode, `UpToDate` TCB (including current
TCB when provided), and unexpired collateral. It emits only the verified claims
JSON on stdout. Parsing an unsigned report or decoding an EAR is not a fallback.

In Kata 3.29.0, guest-components revision
`de3f6ff62aa736619b80d99dfca5bc3d2c9a799d` returns exactly `quote` and
`cc_eventlog`; the latter includes AA events in TCG2 format. Unknown evidence
formats fail closed and require a reviewed collector/verifier update.
