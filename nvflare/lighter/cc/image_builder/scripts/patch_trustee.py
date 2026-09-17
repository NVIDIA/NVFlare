#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Apply the audited CVM boundary patch to the exact design compatibility pin.

This is a deployment patch, not a production approval. Published AS policies
must additionally be mounted read-only; resource storage must be read-only to
KBS and writable only by the mTLS key service.
"""

import argparse
import hashlib
import re
import subprocess
from pathlib import Path

PIN = "a2570329cc33daf9ca16370a1948b5379bb17fbe"


def patch(root, guest_source="https://github.com/confidential-containers/guest-components.git"):
    root = Path(root).resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != PIN:
        raise SystemExit("Refusing to patch a different Trustee revision")
    changed = subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--name-only"], text=True)
    if changed:
        raise SystemExit("Refusing to patch modified Trustee sources")
    builtin = root / "kbs/src/attestation/coco/builtin.rs"
    text = builtin.read_text()
    text = text.replace(
        "inner: RwLock<AttestationService>,", "inner: RwLock<AttestationService>,\n    policy_id: String,"
    )
    text = text.replace(
        'let policy_ids = vec!["default".to_string()];', "let policy_ids = vec![self.policy_id.clone()];"
    )
    text = text.replace(
        "Ok(Self { inner })",
        """let policy_id = std::env::var("CVM_AS_POLICY_ID")
            .context("CVM_AS_POLICY_ID must select an immutable versioned policy")?;
        if policy_id.is_empty() || policy_id == "default" || policy_id.len() > 64
            || !policy_id.bytes().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'_' || c == b'-') {
            bail!("invalid CVM_AS_POLICY_ID");
        }
        Ok(Self { inner, policy_id })""",
    )
    builtin.write_text(text)
    api = root / "kbs/src/api_server.rs"
    text, replacements = re.subn(
        r"core\.attestation_service\.set_policy\(&body\)\.await\?;\s*Ok\(HttpResponse::Ok\(\)\.finish\(\)\)",
        """core.admin_auth.validate_auth(&request)?;
            // CVM AS policies are deployment-managed immutable files.
            Ok(HttpResponse::Forbidden().finish())""",
        api.read_text(),
    )
    if replacements != 1:
        raise SystemExit("AS administrative boundary patch did not match exactly once")
    text = text.replace(
        """    match base_path {""",
        """    if base_path == "resource" && request.method() != Method::GET {
        // Only the create-only key service can write resource storage.
        return Ok(HttpResponse::Forbidden().finish());
    }
    match base_path {""",
        1,
    )
    api.write_text(text)
    token = root / "kbs/src/token/jwk.rs"
    text = token.read_text()
    old = """        let token_data = decode::<Value>(&token, &dkey, &Validation::new(alg))
            .context("Failed to decode attestation token")?;

        Ok(token_data.claims)"""
    new = """        let mut validation = Validation::new(alg);
        validation.leeway = 0;
        let token_data = decode::<Value>(&token, &dkey, &validation)
            .context("Failed to decode attestation token")?;
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs();
        let issued = token_data.claims.get("iat").and_then(Value::as_u64)
            .context("attestation token needs iat")?;
        let expires = token_data.claims.get("exp").and_then(Value::as_u64)
            .context("attestation token needs exp")?;
        if issued > now + 5 || now.saturating_sub(issued) > 300
            || expires <= now || expires <= issued || expires - issued > 300 {
            bail!("attestation token is not fresh");
        }
        Ok(token_data.claims)"""
    if text.count(old) != 1:
        raise SystemExit("Token freshness patch did not match exactly once")
    token.write_text(text.replace(old, new))
    # This deployment supports only bare-metal SNP and TDX, excluding sample,
    # vTPM and unrelated verifier fallbacks from its binary.
    cargo = root / "kbs/Cargo.toml"
    cargo.write_text(cargo.read_text().replace('"all-verifier",', '"snp-verifier", "tdx-verifier",'))
    # DCAP 1.25 increased the bounded advisory buffer from 320 to 450 bytes.
    # Accept the header-defined slice length instead of a stale array dimension.
    dcap = root / "deps/verifier/src/intel_dcap/mod.rs"
    dcap.write_text(
        dcap.read_text().replace("fn get_sa_list(sa_list: &[c_char; 320])", "fn get_sa_list(sa_list: &[c_char])")
    )
    # Backport the bounded in-memory SNP VCEK cache behavior used by newer
    # Trustee releases. The URL is a chip-id + reported-TCB cache key. A miss is
    # serialized, fetched with explicit network deadlines, cryptographically
    # checked against the AMD chain/report, and only then inserted.
    snp = root / "deps/verifier/src/snp/mod.rs"
    text = snp.read_text()
    text = text.replace(
        "use reqwest::{get, Response as ReqwestResponse, StatusCode};",
        "use reqwest::{Response as ReqwestResponse, StatusCode};",
    )
    text = text.replace(
        "use std::{collections::HashMap, hash::Hash, result::Result::Ok, sync::LazyLock};",
        "use std::{collections::HashMap, hash::Hash, result::Result::Ok, sync::LazyLock, time::Duration};\n"
        "use tokio::sync::Mutex;",
    )
    anchor = "#[derive(Default, Debug)]\npub struct Snp {}"
    replacement = (
        "static VCEK_CACHE: LazyLock<Mutex<HashMap<String, Vec<u8>>>> =\n"
        "    LazyLock::new(|| Mutex::new(HashMap::new()));\n"
        "const VCEK_CACHE_CAPACITY: usize = 1024;\n\n" + anchor
    )
    if text.count(anchor) != 1:
        raise SystemExit("SNP cache declaration patch did not match exactly once")
    text = text.replace(anchor, replacement)
    text = text.replace(
        "let vcek_buf = fetch_vcek_from_kds(report, proc_gen.clone())",
        "let vcek_buf = fetch_vcek_from_kds(report, proc_gen.clone(), vendor_certs)",
    )
    text = text.replace(
        "    proc_gen: ProcessorGeneration,\n) -> Result<Vec<u8>> {",
        "    proc_gen: ProcessorGeneration,\n    vendor_certs: &VendorCertificates,\n) -> Result<Vec<u8>> {",
        1,
    )
    request = """    // VCEK in DER format
    let vcek_rsp: ReqwestResponse = get(vcek_url.clone())
        .await
        .context("Unable to send request for VCEK")?;

    match vcek_rsp.status() {
        StatusCode::OK => {
            let vcek_rsp_bytes: Vec<u8> = vcek_rsp
                .bytes()
                .await
                .context("Unable to parse VCEK")?
                .to_vec();
            Ok(vcek_rsp_bytes)
        }

        status => bail!("Unable to fetch VCEK from URL: {status:?}, {vcek_url:?}"),
    }"""
    cached = """    // Serialize misses so concurrent appraisals for one platform cannot
    // stampede AMD KDS. Never cache bytes until the AMD chain, report signature,
    // chip identity and reported TCB have all been verified.
    let mut cache = VCEK_CACHE.lock().await;
    if let Some(value) = cache.get(&vcek_url) {
        return Ok(value.clone());
    }
    if cache.len() >= VCEK_CACHE_CAPACITY {
        bail!("VCEK cache capacity exceeded");
    }
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(15))
        .build()
        .context("Unable to construct bounded VCEK client")?;
    let vcek_rsp: ReqwestResponse = client
        .get(vcek_url.clone())
        .send()
        .await
        .context("Unable to send request for VCEK")?;
    if vcek_rsp.status() != StatusCode::OK {
        bail!("Unable to fetch VCEK from URL: {:?}, {:?}", vcek_rsp.status(), vcek_url);
    }
    let value = vcek_rsp
        .bytes()
        .await
        .context("Unable to parse VCEK")?
        .to_vec();
    let vcek = Certificate::from_bytes(&value)
        .context("Failed to convert KDS VCEK into certificate")?;
    Chain {
        ca: CaChain {
            ark: vendor_certs.ark.clone(),
            ask: vendor_certs.ask.clone(),
        },
        vek: vcek.clone(),
    }
    .verify()
    .context("Certificate chain from KDS failed verification")?;
    (&vcek, &att_report)
        .verify()
        .context("Report signature verification against VCEK signature failed")?;
    verify_report_tcb(&att_report, vcek, proc_gen)
        .context("Reported TCB values do not match VCEK")?;
    cache.insert(vcek_url, value.clone());
    Ok(value)"""
    if text.count(request) != 1:
        raise SystemExit("SNP KDS request patch did not match exactly once")
    snp.write_text(text.replace(request, cached))
    verifier_cargo = root / "deps/verifier/Cargo.toml"
    text = verifier_cargo.read_text()
    old_feature = 'snp-verifier = ["asn1-rs", "openssl", "sev", "x509-parser"]'
    if text.count(old_feature) != 1:
        raise SystemExit("SNP Tokio feature patch did not match exactly once")
    verifier_cargo.write_text(
        text.replace(old_feature, 'snp-verifier = ["asn1-rs", "openssl", "sev", "x509-parser", "tokio/sync"]')
    )
    composite(root, guest_source)
    diff = subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--binary"])
    (root / "cvm-boundary.patch").write_bytes(diff)
    print(hashlib.sha256(diff).hexdigest())


def composite(root, guest_source):
    """Backport NVIDIA composite evidence without moving the compatibility pins."""
    guest_pin = "591d0bb45cd7a2c66f3778428940c40f7eec3b7d"
    guest = root / "cvm_guest"
    if guest.exists():
        raise SystemExit("Refusing to replace an existing cvm_guest checkout")
    subprocess.run(["git", "clone", "--no-checkout", guest_source, str(guest)], check=True)
    subprocess.run(["git", "-C", str(guest), "checkout", "--detach", guest_pin], check=True)
    source = Path(__file__).resolve().parent.parent / "trustee"

    def replace(path, old, new):
        text = path.read_text()
        if text.count(old) != 1:
            raise SystemExit(f"Composite patch anchor mismatch: {path.relative_to(root)}")
        path.write_text(text.replace(old, new))

    verifier = root / "deps/verifier/src/lib.rs"
    replace(verifier, "pub mod sample;", '#[cfg(feature = "nvidia-verifier")]\npub mod nvidia;\npub mod sample;')
    replace(
        verifier,
        "Tee::Nvidia => todo!(),",
        """Tee::Nvidia => {
            cfg_if::cfg_if! {
                if #[cfg(feature = "nvidia-verifier")] {
                    Ok(Box::new(nvidia::Nvidia::new()?) as Box<dyn Verifier + Send + Sync>)
                } else { bail!("NVIDIA verifier is not enabled"); }
            }
        },""",
    )
    replace(verifier, "Tee::Sev => todo!(),", 'Tee::Sev => bail!("SEV is not supported"),')
    replace(verifier, "Tee::Tpm => todo!(),", 'Tee::Tpm => bail!("TPM is not supported"),')
    replace(
        verifier,
        """Tee::Sample => Ok(Box::<sample::Sample>::default() as Box<dyn Verifier + Send + Sync>),
        Tee::SampleDevice => Ok(Box::<sample_device::SampleDeviceVerifier>::default()
            as Box<dyn Verifier + Send + Sync>),""",
        'Tee::Sample | Tee::SampleDevice => bail!("Sample evidence is disabled"),',
    )
    target = root / "deps/verifier/src/nvidia.rs"
    target.write_bytes((source / "nvidia_verifier.rs").read_bytes())
    subprocess.run(["git", "-C", str(root), "add", "--intent-to-add", str(target)], check=True)
    replace(
        root / "deps/verifier/Cargo.toml",
        "[dependencies]",
        'nvidia-verifier = ["jsonwebtoken", "openssl", "tokio/time", "reqwest/json"]\n\n[dependencies]',
    )
    replace(
        root / "attestation-service/Cargo.toml",
        "[features]",
        '[features]\nnvidia-verifier = ["verifier/nvidia-verifier"]',
    )
    replace(
        root / "kbs/Cargo.toml", '"snp-verifier", "tdx-verifier",', '"snp-verifier", "tdx-verifier", "nvidia-verifier",'
    )
    # Only NVIDIA emits a list in this backport. Keep other verifier APIs and
    # CPU policy behavior unchanged, while issuing a separate EAR per GPU.
    replace(
        root / "attestation-service/src/lib.rs",
        """            tee_claims.push(TeeClaims {
                tee: verification_request.tee,
                tee_class,
                claims: claims_from_tee_evidence,
                init_data_claims,
                runtime_data_claims,
            });""",
        """            let device_claims = if verification_request.tee == kbs_types::Tee::Nvidia {
                claims_from_tee_evidence.as_array().context("Invalid NVIDIA claims")?.clone()
            } else { vec![claims_from_tee_evidence] };
            for claims in device_claims {
                tee_claims.push(TeeClaims {
                    tee: verification_request.tee,
                    tee_class: tee_class.clone(),
                    claims,
                    init_data_claims: init_data_claims.clone(),
                    runtime_data_claims: runtime_data_claims.clone(),
                });
            }""",
    )

    attester = guest / "attestation-agent/attester/src/lib.rs"
    replace(attester, "pub mod sample;", '#[cfg(feature = "nvidia-attester")]\npub mod nvidia;\npub mod sample;')
    replace(
        attester,
        """            Tee::Sample => Box::<sample::SampleAttester>::default(),
            Tee::SampleDevice => Box::<sample_device::SampleDeviceAttester>::default(),""",
        """            Tee::Sample | Tee::SampleDevice => bail!("Sample attestation is disabled"),
            #[cfg(feature = "nvidia-attester")]
            Tee::Nvidia => Box::<nvidia::NvidiaAttester>::default(),""",
    )
    replace(
        attester,
        """    if sample_device::detect_platform() {
        additional_devices.push(Tee::SampleDevice);
    }""",
        """    #[cfg(feature = "nvidia-attester")]
    if nvidia::detect_platform() {
        additional_devices.push(Tee::Nvidia);
    }""",
    )
    target = guest / "attestation-agent/attester/src/nvidia.rs"
    target.write_bytes((source / "nvidia_attester.rs").read_bytes())
    subprocess.run(["git", "-C", str(guest), "add", "--intent-to-add", str(target)], check=True)
    replace(guest / "attestation-agent/attester/Cargo.toml", "[features]", '[features]\nnvidia-attester = ["tokio"]')
    replace(
        guest / "attestation-agent/kbs_protocol/Cargo.toml",
        "[features]",
        '[features]\nnvidia-attester = ["attester/nvidia-attester"]',
    )
    replace(
        root / "tools/kbs-client/Cargo.toml",
        "[features]",
        '[features]\nnvidia-attester = ["kbs_protocol/nvidia-attester"]',
    )
    replace(
        guest / "attestation-agent/kbs_protocol/src/builder.rs",
        "        let mut http_client_builder = reqwest::Client::builder()",
        """        let request_timeout = KBS_REQ_TIMEOUT_SEC;
        #[cfg(feature = "nvidia-attester")]
        let request_timeout = if attester::nvidia::detect_platform() { 190 } else { request_timeout };
        let mut http_client_builder = reqwest::Client::builder()""",
    )
    replace(
        guest / "attestation-agent/kbs_protocol/src/builder.rs",
        ".timeout(Duration::from_secs(KBS_REQ_TIMEOUT_SEC))",
        ".timeout(Duration::from_secs(request_timeout))",
    )
    for name, path in (("kbs_protocol", "attestation-agent/kbs_protocol"), ("kms", "confidential-data-hub/kms")):
        replace(
            root / "Cargo.toml",
            name
            + ' = { git = "https://github.com/confidential-containers/guest-components.git", rev = "591d0bb", default-features = false }',
            name + ' = { path = "cvm_guest/' + path + '", default-features = false }',
        )
    replace(root / "Cargo.toml", "[workspace]", '[workspace]\nexclude = ["cvm_guest"]')
    # Cargo uses the same exact packages, now from the pinned, patched checkout.
    lock = root / "Cargo.lock"
    text = lock.read_text()
    text = text.replace(
        'source = "git+https://github.com/confidential-containers/guest-components.git?rev=591d0bb#'
        + guest_pin
        + '"\n',
        "",
    )
    lock.write_text(text)
    guest_patch = subprocess.check_output(["git", "-C", str(guest), "diff", "HEAD", "--binary"])
    (root / "cvm_guest.patch").write_bytes(guest_patch)
    subprocess.run(["git", "-C", str(root), "add", "--intent-to-add", "cvm_guest.patch"], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--guest-source", default="https://github.com/confidential-containers/guest-components.git")
    args = parser.parse_args()
    patch(args.source, args.guest_source)
