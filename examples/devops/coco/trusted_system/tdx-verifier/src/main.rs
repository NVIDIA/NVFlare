// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fail-closed verifier for evidence collected on the trusted rehearsal system.
//! This is not a quote decoder: Trustee verifies the Intel DCAP signature,
//! collateral, report-data binding, InitData binding, and all four CCEL RTMRs.

use anyhow::{ensure, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{env, fs, io::Read, path::Path};
use verifier::{tdx::Tdx, InitDataHash, ReportData, Verifier};

const TRUSTEE_REVISION: &str = "338610fbfed57b66c61a8a3a60e0e4386bdce793";
const MAX_INPUT: u64 = 32 * 1024 * 1024;

fn bounded_read(path: &Path) -> Result<Vec<u8>> {
    let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    ensure!(file.metadata()?.is_file(), "input must be a regular file");
    let mut bytes = Vec::new();
    file.take(MAX_INPUT + 1).read_to_end(&mut bytes)?;
    ensure!(!bytes.is_empty(), "input must not be empty");
    ensure!(bytes.len() as u64 <= MAX_INPUT, "input exceeds 32 MiB");
    Ok(bytes)
}

fn require_evidence(value: &Value) -> Result<()> {
    let evidence = value
        .as_object()
        .context("evidence must be a JSON object")?;
    // Reject envelopes, JWTs, extra claims and old evidence shapes rather than
    // guessing which untrusted field was intended as hardware evidence.
    ensure!(
        evidence.len() == 2
            && evidence.contains_key("quote")
            && evidence.contains_key("cc_eventlog"),
        "expected exactly quote and cc_eventlog from the TDX Attestation Agent"
    );
    for name in ["quote", "cc_eventlog"] {
        let encoded = evidence[name]
            .as_str()
            .context("evidence field must be a string")?;
        ensure!(!encoded.is_empty(), "{name} must not be empty");
        let decoded = STANDARD
            .decode(encoded)
            .with_context(|| format!("invalid {name} base64"))?;
        ensure!(!decoded.is_empty(), "{name} must not decode to empty bytes");
    }
    Ok(())
}

fn require_baseline(claims: &Value) -> Result<()> {
    ensure!(
        claims["tcb_status"] == "UpToDate",
        "Intel DCAP TCB is not UpToDate"
    );
    ensure!(
        claims["collateral_expiration_status"] == "0",
        "Intel DCAP collateral is expired or has no verified expiration status"
    );
    if let Some(current) = claims.get("tcb_status_current") {
        ensure!(current == "UpToDate", "Intel current TCB is not UpToDate");
    }
    ensure!(
        claims["td_attributes"]["debug"] == Value::Bool(false),
        "TDX debug attribute must be verified false"
    );
    ensure!(
        claims["uefi_event_logs"]
            .as_array()
            .is_some_and(|events| !events.is_empty()),
        "verified CCEL must contain events"
    );
    Ok(())
}

fn require_initdata(initdata: &[u8]) -> Result<()> {
    // InitData is a TOML document, not a standalone TOML value expression.
    let document: toml::Value = toml::from_str(std::str::from_utf8(initdata)?)?;
    ensure!(
        document.get("version").and_then(toml::Value::as_str) == Some("0.1.0"),
        "unsupported InitData version"
    );
    ensure!(
        document.get("algorithm").and_then(toml::Value::as_str) == Some("sha256"),
        "InitData must use sha256"
    );
    Ok(())
}

async fn run() -> Result<()> {
    let args: Vec<_> = env::args_os().collect();
    if args.len() == 2 && args[1] == "--version" {
        println!("tdx-evidence-verify 0.1.0 trustee={TRUSTEE_REVISION}");
        return Ok(());
    }
    ensure!(
        args.len() == 4,
        "usage: tdx-evidence-verify EVIDENCE.json CHALLENGE.bin INITDATA.toml"
    );
    let evidence: Value = serde_json::from_slice(&bounded_read(Path::new(&args[1]))?)?;
    require_evidence(&evidence)?;
    let challenge = bounded_read(Path::new(&args[2]))?;
    ensure!(
        challenge.len() == 64,
        "challenge must contain exactly 64 bytes"
    );
    ensure!(
        challenge
            .iter()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "REST evidence challenge must be 64 lowercase hexadecimal ASCII bytes"
    );
    let initdata = bounded_read(Path::new(&args[3]))?;
    require_initdata(&initdata)?;
    let init_hash = Sha256::digest(&initdata);
    // TDX's MRCONFIGID is 48 bytes. Trustee pads this exact 32-byte SHA256
    // digest with 16 zero bytes; it does NOT compare a textual hash.
    let results = Tdx::default()
        .evaluate(
            evidence,
            &ReportData::Value(&challenge),
            &InitDataHash::Value(&init_hash),
        )
        .await
        .context("cryptographic TDX evidence verification failed")?;
    ensure!(
        results.len() == 1,
        "expected exactly one verified TDX CPU result"
    );
    let (claims, tee_class) = &results[0];
    ensure!(tee_class == "cpu", "verifier did not return a CPU result");
    require_baseline(claims)?;
    println!("{}", serde_json::to_string(claims)?);
    Ok(())
}

#[tokio::main]
async fn main() {
    if let Err(err) = run().await {
        eprintln!("TDX verification rejected: {err:#}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn baseline() -> Value {
        json!({"tcb_status": "UpToDate", "collateral_expiration_status": "0",
            "td_attributes": {"debug": false}, "uefi_event_logs": [{}]})
    }

    #[test]
    fn accepts_collector_initdata_document() {
        // Same document shape emitted by tdx-reference.py::make_pod, including
        // a literal multiline string containing the guest's CDH configuration.
        let initdata = br#"version = "0.1.0"
algorithm = "sha256"

[data]
"cdh.toml" = '''
[kbc]
name = "offline_fs_kbc"
url = ""
[image]
extra_root_certificates = ["""-----BEGIN CERTIFICATE-----
TEST
-----END CERTIFICATE-----
"""]
[image.registry_config]
unqualified-search-registries = ["docker.io"]
[[image.registry_config.registry]]
location = "192.0.2.1:5443"
insecure = false
'''
"#;
        assert!(require_initdata(initdata).is_ok());
    }

    #[test]
    fn rejects_malformed_or_unsupported_initdata() {
        for initdata in [
            &b"version = \"0.1.0\"\nalgorithm = ["[..],
            &b"version = \"0.2.0\"\nalgorithm = \"sha256\""[..],
            &b"version = \"0.1.0\"\nalgorithm = \"sha384\""[..],
            &b"algorithm = \"sha256\""[..],
            &b"version = \"0.1.0\""[..],
            &b"version = 1\nalgorithm = \"sha256\""[..],
            &b"version = \"0.1.0\"\nalgorithm = 256"[..],
            &b"\xff"[..],
        ] {
            assert!(require_initdata(initdata).is_err());
        }
    }

    #[test]
    fn accepts_only_complete_verified_baseline() {
        assert!(require_baseline(&baseline()).is_ok());
        let mut claims = baseline();
        claims["td_attributes"]["debug"] = json!(true);
        assert!(require_baseline(&claims).is_err());
        for field in [
            "tcb_status",
            "collateral_expiration_status",
            "td_attributes",
            "uefi_event_logs",
        ] {
            let mut claims = baseline();
            claims.as_object_mut().unwrap().remove(field);
            assert!(require_baseline(&claims).is_err());
        }
    }

    #[test]
    fn rejects_outdated_current_tcb_and_collateral() {
        for status in [
            "OutOfDate",
            "SWHardeningNeeded",
            "ConfigurationNeeded",
            "TDRelaunchAdvised",
        ] {
            let mut claims = baseline();
            claims["tcb_status"] = json!(status);
            assert!(require_baseline(&claims).is_err());
            let mut claims = baseline();
            claims["tcb_status_current"] = json!(status);
            assert!(require_baseline(&claims).is_err());
        }
        let mut claims = baseline();
        claims["collateral_expiration_status"] = json!("1");
        assert!(require_baseline(&claims).is_err());
    }

    #[test]
    fn rejects_absent_eventlog_and_untrusted_envelopes() {
        assert!(require_evidence(&json!({"quote": "YQ==", "cc_eventlog": "YQ=="})).is_ok());
        for evidence in [
            json!({"quote": "YQ=="}),
            json!({"quote": "YQ==", "cc_eventlog": null}),
            json!({"quote": "YQ==", "cc_eventlog": ""}),
            json!({"quote": "YQ==", "cc_eventlog": "not base64"}),
            json!({"quote": "YQ==", "cc_eventlog": "YQ==", "tcb_status": "UpToDate"}),
        ] {
            assert!(require_evidence(&evidence).is_err());
        }
    }
}
