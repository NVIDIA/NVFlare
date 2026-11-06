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

# The original BreastG-FCL MIT notice is retained below for the upstream code.
# MIT License
#
# Copyright (c) 2026 IntelliSys-Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Download and normalize the official TCGA-Breast-Radiogenomics artifacts.

This script uses the small, openly downloadable TCIA analysis-result files,
verifies their SHA-256 digests, and creates the patient-level spatial and
temporal tables consumed only by relational-graph generation. It never
substitutes heuristic tumor regions when an official record is unavailable.
"""

import argparse
import hashlib
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd

BASE_URL = "https://www.cancerimagingarchive.net/wp-content/uploads"
ARTIFACTS = {
    "radiologist_reads": {
        "filename": "tcga-breast-radiologist-reads.xls",
        "sha256": "0e476b63990e3d82f427c46bd45554ada19b1b8898bd34282afe0c84126ed2d1",
    },
    "segmentations": {
        "filename": "TCGA_Segmented_Lesions_UofC.zip",
        "sha256": "9c5705c15023accdec5334eaf2195fd73a182a70e7c7f584c6877a2bddcad7f8",
    },
    "radiomics": {
        "filename": "TCGA-Run-2014_91cases_features_UChicago-V2010-MRI-Workstation.xls",
        "sha256": "d70309bd774d6a55a97235c65ad1194013aea4528a7aa4634863222cbebbdbb4",
    },
    "pam50": {
        "filename": "Perou-TCGA-BRCA-MRIsPAM50GHI21NKI70-MAILED.xlsx",
        "sha256": "ca9746b5241269d6d495a64c798098e682e361f312fbea21edb1e1c14501aafe",
    },
    "clinical": {
        "filename": "brca-clinicalforwiki.xls",
        "sha256": "49671bd7299214731840b52ddc67d3f2435eeefb47cec1703482f3c9dc859f77",
    },
}

EXPECTED_LABEL_COUNTS = {
    "Basal": 10,
    "Her2": 5,
    "LumA": 55,
    "LumB": 10,
    "Normal": 4,
}

TEMPORAL_COLUMNS = [
    "Maximum enhancement (K1)",
    "Time to peak (K2)",
    "Uptake rate (K3)",
    "Washout rate (K4)",
    "Curve shape index (K5)",
    "E1 (K6)",
    "Signal Enhancement Ratio (SER) (K7)",
    "Maximum enhancement-variance (E1)",
    "Enhancement-Variance Time to Peak (E2)",
    "Enhancement-variance Increasing Rate (E3)",
    "Enhancement-variance Decreasing Rate (E4)",
]


def parse_args():
    default_output = Path(__file__).resolve().parents[1] / "data" / "tcia_official_radiogenomics"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url, destination, expected_sha256, force=False):
    if destination.exists() and not force:
        observed = sha256(destination)
        if observed == expected_sha256:
            return observed
        raise RuntimeError(f"Existing artifact has the wrong SHA-256: {destination} ({observed})")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "BreastG-FCL-strict-reproduction/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    observed = sha256(destination)
    if observed != expected_sha256:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {observed}")
    return observed


def case_id_from_lesion_name(value):
    match = re.search(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", str(value).upper())
    return match.group(0) if match else None


def normalize_pam50(value):
    aliases = {
        "basal": "Basal",
        "her2": "Her2",
        "luma": "LumA",
        "lumb": "LumB",
        "normal": "Normal",
    }
    return aliases.get(str(value).strip().lower())


def build_tables(source_dir, output_dir):
    radiomics_path = source_dir / ARTIFACTS["radiomics"]["filename"]
    pam50_path = source_dir / ARTIFACTS["pam50"]["filename"]
    clinical_path = source_dir / ARTIFACTS["clinical"]["filename"]

    radiomics = pd.read_excel(radiomics_path)
    pam50 = pd.read_excel(pam50_path)
    clinical = pd.read_excel(clinical_path, sheet_name="Sheet1")

    if "Lesion Name" not in radiomics or "CLID" not in pam50:
        raise RuntimeError("Official workbooks do not contain the expected identifiers")
    missing_temporal = [column for column in TEMPORAL_COLUMNS if column not in radiomics]
    if missing_temporal:
        raise RuntimeError(f"Official radiomics workbook is missing: {missing_temporal}")

    radiomics = radiomics.copy()
    radiomics.insert(
        0,
        "PatientID",
        radiomics["Lesion Name"].map(case_id_from_lesion_name),
    )
    if radiomics["PatientID"].isna().any() or radiomics["PatientID"].duplicated().any():
        raise RuntimeError("Official radiomics identifiers are missing or duplicated")

    pam50 = pam50.copy()
    pam50["PatientID"] = pam50["CLID"].astype(str).str[:12].str.upper()
    pam50["PAM50Call_RNAseq"] = pam50["Pam50.Call"].map(normalize_pam50)
    clinical = clinical.copy()
    clinical["PatientID"] = clinical["bcr_patient_barcode"].astype(str).str[:12].str.upper()

    joined = (
        radiomics[["PatientID"]]
        .merge(
            pam50[["PatientID", "PAM50Call_RNAseq"]],
            on="PatientID",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            clinical[
                [
                    "PatientID",
                    "age_at_initial_pathologic_diagnosis",
                    "ajcc_neoplasm_disease_stage",
                ]
            ],
            on="PatientID",
            how="left",
            validate="one_to_one",
        )
        .dropna(subset=["PAM50Call_RNAseq"])
        .sort_values("PatientID")
        .reset_index(drop=True)
    )

    label_counts = dict(Counter(joined["PAM50Call_RNAseq"]))
    if len(joined) != 84 or label_counts != EXPECTED_LABEL_COUNTS:
        raise RuntimeError("Official cohort validation failed: " f"rows={len(joined)}, labels={label_counts}")
    if joined["ajcc_neoplasm_disease_stage"].isna().any():
        raise RuntimeError("Official 84-case cohort contains missing pathologic stage")

    cohort_ids = set(joined["PatientID"])
    numeric_columns = [column for column in radiomics.columns if column not in {"PatientID", "Lesion Name"}]
    spatial_columns = [column for column in numeric_columns if column not in TEMPORAL_COLUMNS]

    full = radiomics[radiomics["PatientID"].isin(cohort_ids)][["PatientID"] + numeric_columns].sort_values("PatientID")
    spatial = radiomics[radiomics["PatientID"].isin(cohort_ids)][["PatientID"] + spatial_columns].sort_values(
        "PatientID"
    )
    temporal = radiomics[radiomics["PatientID"].isin(cohort_ids)][["PatientID"] + TEMPORAL_COLUMNS].sort_values(
        "PatientID"
    )

    phenotype = pd.DataFrame(
        {
            "sampleID": joined["PatientID"] + "-01",
            "_PATIENT": joined["PatientID"],
            "sample_type": "Primary Tumor",
            "PAM50Call_RNAseq": joined["PAM50Call_RNAseq"],
            "age_at_initial_pathologic_diagnosis": joined["age_at_initial_pathologic_diagnosis"],
            "pathologic_stage": joined["ajcc_neoplasm_disease_stage"],
            "gender": "FEMALE",
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    full.to_csv(output_dir / "official_radiomics_patient_features.csv", index=False)
    spatial.to_csv(output_dir / "official_spatial_patient_features.csv", index=False)
    temporal.to_csv(output_dir / "official_temporal_patient_features.csv", index=False)
    phenotype.to_csv(
        output_dir / "official_pam50_clinical.tsv",
        index=False,
        sep="\t",
    )
    return {
        "official_radiomics_rows": int(len(radiomics)),
        "matched_cohort_rows": int(len(joined)),
        "spatial_feature_dim": int(len(spatial_columns)),
        "temporal_feature_dim": int(len(TEMPORAL_COLUMNS)),
        "input_feature_dim": int(len(numeric_columns)),
        "pam50_label_counts": label_counts,
    }


def main():
    args = parse_args()
    source_dir = args.output_dir / "source"
    source_manifest = {}
    for key, artifact in ARTIFACTS.items():
        filename = artifact["filename"]
        url = f"{BASE_URL}/{filename}"
        destination = source_dir / filename
        observed = download_verified(
            url,
            destination,
            artifact["sha256"],
            force=args.force,
        )
        source_manifest[key] = {
            "url": url,
            "filename": filename,
            "bytes": destination.stat().st_size,
            "sha256": observed,
        }
        print(f"Verified {key}: {destination}")

    table_summary = build_tables(source_dir, args.output_dir)
    manifest = {
        "dataset": "TCGA-Breast-Radiogenomics",
        "source_page": ("https://www.cancerimagingarchive.net/analysis-result/" "tcga-breast-radiogenomics/"),
        "artifacts": source_manifest,
        "derived_tables": table_summary,
    }
    manifest_path = args.output_dir / "official_source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(table_summary, indent=2, sort_keys=True))
    print(f"Wrote provenance manifest: {manifest_path}")


if __name__ == "__main__":
    main()
