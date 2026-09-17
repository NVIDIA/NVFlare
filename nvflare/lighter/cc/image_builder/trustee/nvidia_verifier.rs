// Copyright (c) 2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Composite NVIDIA wire format follows Trustee's NRAS verifier (217ed7bf90bf).
// This restricted backport uses an operator-pinned JWKS, validates both JWTs and
// their digest relationship, and binds the signed nonce to RCAR runtime data.
use anyhow::{bail, ensure, Context, Result};
use async_trait::async_trait;
use jsonwebtoken::{decode, decode_header, jwk::JwkSet, Algorithm, DecodingKey, Validation};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::HashSet, path::PathBuf, time::{Duration, SystemTime, UNIX_EPOCH}};
use crate::{regularize_data, InitDataHash, ReportData, TeeClass, TeeEvidence, TeeEvidenceParsedClaim, Verifier};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    mode: String,
    url: String,
    issuer: String,
    jwks: PathBuf,
    jwks_sha256: String,
}

pub struct Nvidia {
    config: Config,
    jwks: JwkSet,
}

impl Nvidia {
    pub fn new() -> Result<Self> {
        let path = std::env::var("CVM_NVIDIA_CONFIG").context("NVIDIA verifier is not configured")?;
        let config: Config = serde_json::from_slice(&std::fs::read(path)?)?;
        ensure!(config.mode == "remote", "Only explicit NRAS remote verification is supported");
        let url = reqwest::Url::parse(&config.url)?;
        ensure!(url.scheme() == "https" && url.host_str().is_some()
            && url.username().is_empty() && url.password().is_none(), "NRAS requires HTTPS");
        ensure!(!config.issuer.is_empty(), "NRAS issuer must be pinned");
        let raw = std::fs::read(&config.jwks)?;
        ensure!(hex::encode(Sha256::digest(&raw)) == config.jwks_sha256, "NRAS JWKS digest mismatch");
        let jwks: JwkSet = serde_json::from_slice(&raw)?;
        ensure!(!jwks.keys.is_empty(), "Empty NRAS JWKS");
        Ok(Self { config, jwks })
    }

    fn jwt(&self, token: &str, nonce: &str) -> Result<Value> {
        ensure!(token.len() <= 128 * 1024, "Oversized NRAS token");
        let header = decode_header(token)?;
        ensure!(header.alg == Algorithm::ES384, "NRAS algorithm mismatch");
        let kid = header.kid.context("Missing NRAS signing key id")?;
        ensure!(self.jwks.keys.iter().filter(|k| k.common.key_id.as_deref() == Some(&kid)).count() == 1,
            "Unknown or ambiguous NRAS signing key");
        let key = DecodingKey::from_jwk(self.jwks.find(&kid).context("Unknown NRAS key")?)?;
        let mut validation = Validation::new(Algorithm::ES384);
        validation.set_issuer(&[&self.config.issuer]);
        validation.set_required_spec_claims(&["iss", "iat", "exp", "nbf"]);
        validation.validate_nbf = true;
        validation.leeway = 0;
        let claims = decode::<Value>(token, &key, &validation)?.claims;
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let iat = claims["iat"].as_u64().context("Invalid NRAS issuance time")?;
        let exp = claims["exp"].as_u64().context("Invalid NRAS expiry")?;
        ensure!(iat <= now + 5 && now.saturating_sub(iat) <= 180 && exp > now && exp > iat,
            "Stale NRAS response");
        ensure!(claims["eat_nonce"].as_str() == Some(nonce), "NRAS nonce mismatch");
        Ok(claims)
    }

    fn response(&self, body: &[u8], nonce: &str) -> Result<Value> {
        let eat: Value = serde_json::from_slice(body)?;
        let parts = eat.as_array().context("Invalid detached NRAS EAT")?;
        ensure!(parts.len() == 2, "Invalid NRAS EAT length");
        let overall = parts[0].as_array().context("Invalid NRAS overall token")?;
        ensure!(overall.len() == 2 && overall[0] == "JWT", "Invalid NRAS token type");
        let overall = self.jwt(overall[1].as_str().context("Missing overall JWT")?, nonce)?;
        ensure!(overall["x-nvidia-ver"] == "3.0", "Unexpected NVIDIA claims version");
        ensure!(overall["x-nvidia-overall-att-result"] == true, "NRAS overall appraisal denied");
        let devices = parts[1].as_object().context("Invalid NRAS devices")?;
        let digests = overall["submods"].as_object().context("Missing NRAS device digests")?;
        ensure!(devices.len() == 1 && digests.len() == 1, "NRAS device count mismatch");
        let (name, token) = devices.iter().next().context("Missing NRAS device")?;
        ensure!(name == "GPU-0", "Unexpected NRAS device class");
        let token = token.as_str().context("Invalid device JWT")?;
        let expected = json!(["DIGEST", ["SHA-256", hex::encode(Sha256::digest(token.as_bytes()))]]);
        ensure!(digests.get(name) == Some(&expected), "NRAS device digest mismatch");
        let mut claims = self.jwt(token, nonce)?;
        // NRAS puts the version in the signed overall JWT and identifies the
        // device class with its signed GPU-0 digest key. Neither field must be
        // repeated in the detached JWT; reject contradictions if supplied.
        ensure!(claims.get("x-nvidia-ver").is_none_or(|value| value == "3.0"),
            "Conflicting NVIDIA device claims version");
        ensure!(claims.get("x-nvidia-device-type").is_none_or(|value| value == "gpu"),
            "Conflicting NVIDIA device type");
        ensure!(claims["x-nvidia-gpu-attestation-report-nonce-match"] == true,
            "NRAS GPU evidence is not bound to the challenge");
        ensure!(claims["ueid"].as_str().is_some_and(|id| !id.is_empty()), "Missing GPU identity");
        claims["x-nvidia-ver"] = overall["x-nvidia-ver"].clone();
        claims["x-nvidia-device-type"] = json!("gpu");
        claims["x-nvidia-overall-att-result"] = json!(true);
        claims["verifier"] = json!("nras-v3");
        Ok(claims)
    }

    async fn devices(&self, evidence: Value, nonce: String) -> Result<Value> {
        let devices = evidence["device_evidence_list"].as_array().context("Missing NVIDIA evidence")?;
        ensure!((1..=8).contains(&devices.len()), "Unsupported GPU count");
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(5)).timeout(Duration::from_secs(20)).build()?;
        let mut claims = Vec::new();
        let mut identities = HashSet::new();
        for device in devices {
            let arch = device["arch"].as_str().context("Missing GPU architecture")?;
            ensure!(arch == "HOPPER" || arch == "BLACKWELL", "Unsupported GPU architecture");
            let report = device["evidence"].as_str().context("Missing GPU report")?;
            let cert = device["certificate"].as_str().context("Missing GPU certificate")?;
            ensure!(report.len() <= 256 * 1024 && cert.len() <= 256 * 1024, "Oversized GPU evidence");
            let request = json!({"nonce": nonce, "arch": arch, "claims_version": "3.0",
                "evidence_list": [{"evidence": report, "certificate": cert}]});
            let mut response = client.post(&self.config.url).json(&request).send().await?;
            ensure!(response.status().is_success(), "NRAS request failed");
            let mut body = Vec::new();
            while let Some(chunk) = response.chunk().await? {
                ensure!(body.len() + chunk.len() <= 512 * 1024, "Oversized NRAS response");
                body.extend_from_slice(&chunk);
            }
            let mut verified = self.response(&body, &nonce)?;
            ensure!(identities.insert(verified["ueid"].as_str().context("Missing GPU identity")?.to_string()),
                "Duplicate GPU evidence");
            verified["arch"] = json!(arch);
            claims.push(verified);
        }
        // A slow later device must not extend an earlier device's NRAS
        // authorization past expiry when the AS issues the composite EAR.
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        for claim in &claims {
            let issued = claim["iat"].as_u64().context("Missing GPU issuance time")?;
            let expires = claim["exp"].as_u64().context("Missing GPU expiry")?;
            ensure!(expires > now && now.saturating_sub(issued) <= 180, "GPU response expired during composite appraisal");
        }
        Ok(json!(claims))
    }
}

#[async_trait]
impl Verifier for Nvidia {
    async fn evaluate(&self, evidence: TeeEvidence, expected: &ReportData,
        _init: &InitDataHash) -> Result<(TeeEvidenceParsedClaim, TeeClass)> {
        let data = match expected {
            ReportData::Value(value) => value,
            ReportData::NotProvided => bail!("NVIDIA appraisal requires RCAR runtime data"),
        };
        let nonce = hex::encode(regularize_data(data, 32, "report_data", "nvidia"));
        let claims = tokio::time::timeout(Duration::from_secs(180), self.devices(evidence, nonce)).await??;
        Ok((claims, "gpu".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
    use jsonwebtoken::{encode, EncodingKey, Header};
    use openssl::{bn::BigNum, ec::{EcGroup, EcKey}, nid::Nid, pkey::PKey};

    fn fixture() -> (Nvidia, EncodingKey, String, Value) {
        let group = EcGroup::from_curve_name(Nid::SECP384R1).unwrap();
        let key = EcKey::generate(&group).unwrap();
        let mut x = BigNum::new().unwrap();
        let mut y = BigNum::new().unwrap();
        let mut ctx = openssl::bn::BigNumContext::new().unwrap();
        key.public_key().affine_coordinates_gfp(&group, &mut x, &mut y, &mut ctx).unwrap();
        let jwks = json!({"keys": [{"kty": "EC", "crv": "P-384", "alg": "ES384", "kid": "test",
            "x": URL_SAFE_NO_PAD.encode(x.to_vec_padded(48).unwrap()),
            "y": URL_SAFE_NO_PAD.encode(y.to_vec_padded(48).unwrap())}]});
        let der = PKey::from_ec_key(key).unwrap().private_key_to_pkcs8().unwrap();
        let signer = EncodingKey::from_ec_der(&der);
        let verifier = Nvidia {
            config: Config { mode: "remote".into(), url: "https://nras.test/v4/attest/gpu".into(),
                issuer: "https://nras.test".into(), jwks: PathBuf::new(), jwks_sha256: String::new() },
            jwks: serde_json::from_value(jwks).unwrap(),
        };
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let nonce = hex::encode([0xab; 32]);
        let claims = json!({"iss": "https://nras.test", "iat": now, "nbf": now - 1, "exp": now + 120,
            "eat_nonce": nonce,
            "x-nvidia-gpu-attestation-report-nonce-match": true, "ueid": "gpu0"});
        (verifier, signer, nonce, claims)
    }
    fn signed(key: &EncodingKey, claims: &Value) -> String {
        let mut header = Header::new(Algorithm::ES384);
        header.kid = Some("test".into());
        encode(&header, claims, key).unwrap()
    }
    fn eat(key: &EncodingKey, claims: &Value, damage: &str) -> Vec<u8> {
        let mut device = claims.clone();
        if damage == "device-nonce" { device["eat_nonce"] = json!("ff"); }
        if damage == "report-nonce" { device["x-nvidia-gpu-attestation-report-nonce-match"] = json!(false); }
        if damage == "device-expired" { device["exp"] = json!(1); }
        if damage == "device-issuer" { device["iss"] = json!("attacker"); }
        if damage == "device-version" { device["x-nvidia-ver"] = json!("2.0"); }
        if damage == "device-type" { device["x-nvidia-device-type"] = json!("switch"); }
        let token = signed(key, &device);
        let mut overall = claims.clone();
        overall["x-nvidia-ver"] = json!("3.0");
        overall["x-nvidia-overall-att-result"] = json!(damage != "overall");
        overall["submods"] = json!({"GPU-0": ["DIGEST", ["SHA-256", hex::encode(Sha256::digest(token.as_bytes()))]]});
        if damage == "digest" { overall["submods"]["GPU-0"][1][1] = json!("00"); }
        if damage == "overall-nonce" { overall["eat_nonce"] = json!("ff"); }
        if damage == "stale" { overall["iat"] = json!(1); }
        if damage == "future" { overall["nbf"] = json!(u32::MAX); }
        if damage == "missing-nbf" { overall.as_object_mut().unwrap().remove("nbf"); }
        if damage == "version" { overall["x-nvidia-ver"] = json!("2.0"); }
        let mut result = json!([["JWT", signed(key, &overall)], {"GPU-0": token}]);
        if damage == "extra-device" { result[1]["GPU-1"] = result[1]["GPU-0"].clone(); }
        if damage == "signature" {
            let (_, wrong, _, _) = fixture();
            result[0][1] = json!(signed(&wrong, &overall));
        }
        serde_json::to_vec(&result).unwrap()
    }

    #[test]
    fn signed_nras_response_binds_both_tokens_digest_and_nonce() {
        let (verifier, key, nonce, claims) = fixture();
        assert_eq!(verifier.response(&eat(&key, &claims, ""), &nonce).unwrap()["verifier"], "nras-v3");
        assert!(verifier.response(&eat(&key, &claims, ""), "another-transaction").is_err());
        for damage in ["device-nonce", "report-nonce", "device-expired", "device-issuer", "overall",
            "digest", "overall-nonce", "stale", "future", "missing-nbf", "version", "extra-device", "signature",
            "device-version", "device-type"] {
            assert!(verifier.response(&eat(&key, &claims, damage), &nonce).is_err(), "accepted {damage}");
        }
        for input in [b"{}".as_slice(), b"null", b"[]", b"[[], {}]"] {
            assert!(verifier.response(input, &nonce).is_err());
        }
    }

    // Opt in via tests/run_nras_policy_test.py with the pinned policy evaluator.
    // The fixture follows the documented NRAS v3 detached shape independently
    // of gpu_policy.json: the overall JWT owns the version, and GPU-0 owns type.
    #[test]
    #[ignore = "Requires the CVM fixture and generated policy evaluator inputs"]
    fn signed_v3_device_claims_reach_generated_policy() {
        let (verifier, key, nonce, fresh) = fixture();
        let root = PathBuf::from(std::env::var("CVM_NRAS_POLICY_TEST").unwrap());
        let mut device: Value = serde_json::from_slice(&std::fs::read(root.join("device.json")).unwrap()).unwrap();
        for name in ["iss", "iat", "nbf", "exp", "eat_nonce"] {
            device[name] = fresh[name].clone();
        }
        assert!(device.get("x-nvidia-ver").is_none());
        assert!(device.get("x-nvidia-device-type").is_none());
        let verified = verifier.response(&eat(&key, &device, ""), &nonce).unwrap();
        for (name, value) in device.as_object().unwrap() {
            assert_eq!(verified.get(name), Some(value), "discarded signed claim {name}");
        }
        assert_eq!(verified["x-nvidia-ver"], "3.0");
        assert_eq!(verified["x-nvidia-device-type"], "gpu");
        let evaluate = |mut claims: Value| {
            // devices() attaches this checked architecture before TeeClaims;
            // the EAR broker namespaces the complete claims under Tee::Nvidia.
            claims["arch"] = json!("HOPPER");
            std::fs::write(root.join("input.json"), serde_json::to_vec(&json!({"nvidia": claims})).unwrap()).unwrap();
            let output = std::process::Command::new(std::env::var("CVM_POLICY_EVAL").unwrap())
                .arg(root.join("policy.rego")).arg(root.join("input.json"))
                .arg(root.join("data.json")).arg("data.policy.approved").output().unwrap();
            assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
            String::from_utf8(output.stdout).unwrap().trim() == "true"
        };
        assert!(evaluate(verified));
        for name in ["secboot", "x-nvidia-gpu-driver-version", "x-nvidia-gpu-vbios-version",
            "x-nvidia-gpu-driver-rim-cert-chain", "x-nvidia-gpu-attestation-report-signature-verified"] {
            let mut missing = device.clone();
            missing.as_object_mut().unwrap().remove(name);
            let verified = verifier.response(&eat(&key, &missing, ""), &nonce).unwrap();
            assert!(!evaluate(verified), "accepted missing signed claim {name}");
        }
        let mut incomplete_ocsp = device.clone();
        incomplete_ocsp["x-nvidia-gpu-driver-rim-cert-chain"].as_object_mut().unwrap()
            .remove("x-nvidia-cert-ocsp-nonce-matches");
        let verified = verifier.response(&eat(&key, &incomplete_ocsp, ""), &nonce).unwrap();
        assert!(!evaluate(verified), "accepted missing OCSP freshness claim");
    }

    #[test]
    fn unsupported_production_tees_return_errors_without_panicking() {
        for tee in [kbs_types::Tee::Sample, kbs_types::Tee::SampleDevice, kbs_types::Tee::Sev, kbs_types::Tee::Tpm] {
            assert!(crate::to_verifier(&tee).is_err());
        }
    }
}
