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
use key_value_storage::{KvStorageProvider, StorageBackendConfig};
use verifier::{to_verifier, InitDataHash, ReportData, VerifierConfig};

const TRUSTEE_REVISION: &str = "338610fbfed57b66c61a8a3a60e0e4386bdce793";
const MAX_INPUT: u64 = 32 * 1024 * 1024;

const CHANNEL_ENV: &str = "NVFLARE_TDX_TCB_UPDATE_TYPE";

fn collateral_channel(value: Option<&str>) -> Result<&str> {
    let channel = value.unwrap_or("early");
    ensure!(matches!(channel, "early" | "standard"), "{CHANNEL_ENV} must be early or standard");
    Ok(channel)
}

fn channel_config(channel: &str) -> Result<VerifierConfig> {
    let selected = collateral_channel(Some(channel))?;
    // This is an independently supplied operator setting. Never infer it from
    // quote contents, TCB status, or an earlier failed appraisal.
    serde_json::from_value(serde_json::json!({
        "dcap_verifier": {
            "collateral_service": "https://api.trustedservices.intel.com/sgx/certification/v4/",
            "tcb_update_type": selected
        }
    })).context("construct pinned Intel collateral configuration")
}

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

// Diagnostics disclose only bounded, verified status labels; never full claims,
// raw evidence, token responses, measurements, or key-bearing fields.
fn diagnostic_status(claims: &Value, name: &str) -> String {
    match claims.get(name) {
        None => "<absent>".to_owned(),
        Some(Value::String(value))
            if !value.is_empty()
                && value.len() <= 128
                && value.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_') =>
        {
            value.clone()
        }
        _ => "<invalid>".to_owned(),
    }
}

fn require_baseline(claims: &Value) -> Result<()> {
    ensure!(
        claims["tcb_status"] == "UpToDate",
        "Intel DCAP TCB rejected: verified tcb_status={}; verified tcb_status_current={}; expected=UpToDate",
        diagnostic_status(claims, "tcb_status"),
        diagnostic_status(claims, "tcb_status_current")
    );
    ensure!(
        claims["collateral_expiration_status"] == "0",
        "Intel DCAP collateral is expired or has no verified expiration status"
    );
    if let Some(current) = claims.get("tcb_status_current") {
        ensure!(
            current == "UpToDate",
            "Intel current TCB rejected: verified tcb_status={}; verified tcb_status_current={}; expected=UpToDate",
            diagnostic_status(claims, "tcb_status"),
            diagnostic_status(claims, "tcb_status_current")
        );
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
    let configured_channel = match env::var(CHANNEL_ENV) {
        Ok(value) => Some(value),
        Err(env::VarError::NotPresent) => None,
        Err(env::VarError::NotUnicode(_)) => anyhow::bail!("{CHANNEL_ENV} must be early or standard"),
    };
    let channel = collateral_channel(configured_channel.as_deref())?;
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
    eprintln!("Intel DCAP collateral channel selected: {channel}");
    let tee = serde_json::from_value(serde_json::json!("tdx"))?;
    let selected = to_verifier(
        &tee,
        Some(channel_config(channel)?),
        KvStorageProvider::new(StorageBackendConfig::default()),
    ).await?;
    let results = selected
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
    fn independent_collateral_channel_defaults_and_validation() {
        assert_eq!(collateral_channel(None).unwrap(), "early");
        for (channel, debug_variant) in [("early", "Early"), ("standard", "Standard")] {
            assert_eq!(collateral_channel(Some(channel)).unwrap(), channel);
            assert!(format!("{:?}", channel_config(channel).unwrap()).contains(debug_variant));
            assert!(require_baseline(&baseline()).is_ok());
            let mut claims = baseline();
            claims["tcb_status"] = json!("OutOfDate");
            assert!(require_baseline(&claims).is_err());
        }
        for invalid in ["", "Early", "STANDARD", "standard\n", "auto", "../standard"] {
            assert!(collateral_channel(Some(invalid)).is_err());
            assert!(channel_config(invalid).is_err());
        }
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
    fn reports_only_bounded_verified_tcb_diagnostics_without_accepting_them() {
        let mut claims = baseline();
        claims["tcb_status"] = json!("OutOfDate");
        claims["tcb_status_current"] = json!("SWHardeningNeeded");
        let error = require_baseline(&claims).unwrap_err().to_string();
        assert!(error.contains("verified tcb_status=OutOfDate"));
        assert!(error.contains("verified tcb_status_current=SWHardeningNeeded"));
        assert!(error.contains("expected=UpToDate"));
        claims["tcb_status"] = json!("UpToDate");
        let error = require_baseline(&claims).unwrap_err().to_string();
        assert!(error.contains("verified tcb_status_current=SWHardeningNeeded"));
        claims["tcb_status_current"] = json!("secret\nraw-token");
        assert_eq!(diagnostic_status(&claims, "tcb_status_current"), "<invalid>");
        assert!(!require_baseline(&claims).unwrap_err().to_string().contains("secret"));
        assert_eq!(diagnostic_status(&claims, "nonexistent"), "<absent>");
        claims["tcb_status_current"] = json!("a".repeat(129));
        assert_eq!(diagnostic_status(&claims, "tcb_status_current"), "<invalid>");
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
