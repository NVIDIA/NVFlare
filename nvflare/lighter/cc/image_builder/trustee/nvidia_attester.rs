// Copyright (c) 2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Backport the upstream NVIDIA composite evidence format using the measured
// NVAT CLI already shipped in GPU roots. Collect only: no in-guest appraisal.
use super::{Attester, TeeEvidence};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{path::Path, time::Duration};

#[derive(Default)]
pub struct NvidiaAttester;

pub fn detect_platform() -> bool {
    // A present but broken/CC-disabled device must fail collection, never fall
    // back to a fabricated appraisal. CPU-only images have no NVAT executable.
    Path::new("/usr/bin/nvattest").is_file()
}

#[async_trait::async_trait]
impl Attester for NvidiaAttester {
    async fn get_evidence(&self, mut report_data: Vec<u8>) -> Result<TeeEvidence> {
        report_data.resize(32, 0);
        let nonce = hex::encode(report_data);
        let output = tokio::time::timeout(Duration::from_secs(60),
            tokio::process::Command::new("/usr/bin/nvattest")
                .args(["--log-level", "off", "--format", "json", "collect-evidence",
                    "--device", "gpu", "--gpu-evidence-source", "nvml", "--nonce", &nonce])
                .kill_on_drop(true).output()).await??;
        ensure!(output.status.success() && output.stdout.len() <= 4 * 1024 * 1024,
            "NVIDIA evidence collection failed");
        let result: Value = serde_json::from_slice(&output.stdout)?;
        ensure!(result["result_code"] == 0, "NVIDIA evidence collection denied");
        let devices = result["evidences"].as_array().context("Missing GPU evidence list")?;
        ensure!((1..=8).contains(&devices.len()), "Unsupported GPU evidence count");
        Ok(json!({"device_evidence_list": devices}))
    }
}
