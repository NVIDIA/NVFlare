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

import sys
from pathlib import Path

CHECKS_PARENT = Path(__file__).resolve().parents[4] / "dev_tools" / "agent" / "skills"
sys.path.insert(0, str(CHECKS_PARENT))

from checks.lints import run_v1_lints  # noqa: E402


def test_seed_skills_pass_v1_admission_lints():
    repo_root = Path(__file__).resolve().parents[4]

    result = run_v1_lints(repo_root / "skills")

    assert result["status"] == "ok"
    assert result["summary"]["skill_count"] >= 2
    assert result["findings"] == []


def test_diagnose_job_catalog_pins_recovery_categories():
    repo_root = Path(__file__).resolve().parents[4]
    skill_root = repo_root / "skills" / "nvflare-diagnose-job"
    skill_text = skill_root.joinpath("SKILL.md").read_text(encoding="utf-8")
    catalog_text = skill_root.joinpath("references/failure-patterns.md").read_text(encoding="utf-8")
    normalized_catalog = " ".join(catalog_text.split())
    rows = _failure_pattern_rows(catalog_text)

    assert "copying the category from the matched" in skill_text
    assert "Do not infer or override the category" in skill_text
    assert "copy the `Recovery Category` value from that same row exactly" in normalized_catalog
    assert "set `matched_pattern` to `UNKNOWN` and `recovery_category` to `UNKNOWN`" in normalized_catalog

    round_timeout = rows["ROUND_TIMEOUT"]
    assert round_timeout["Recovery Category"] == "`ENVIRONMENT_FAILURE`"
    assert "timeout configuration" not in round_timeout["Next Action"]
    assert "temporary mitigation, not the primary fix" in round_timeout["Next Action"]

    resource_capacity = rows["RESOURCE_EXCEEDS_HOST_CAPACITY"]
    assert resource_capacity["Recovery Category"] == "`FIXABLE_BY_CONFIG`"
    assert "`num_of_gpus specified` exceeds available GPUs" in resource_capacity["Evidence Signals"]
    assert "`Memory per GPU specified` exceeds available GPU memory" in resource_capacity["Evidence Signals"]
    assert "resource requirements in the job or site resource config" in resource_capacity["Next Action"]

    config_validation = rows["CONFIG_FILE_VALIDATION_ERROR"]
    assert config_validation["Recovery Category"] == "`FIXABLE_BY_CONFIG`"
    assert "`config_fed_server.json`" in config_validation["Evidence Signals"]
    assert "`privacy.json`" in config_validation["Evidence Signals"]
    assert "default scope/filter does not exist" in config_validation["Evidence Signals"]
    assert "Correct the referenced server/site config file" in config_validation["Next Action"]

    infrastructure = rows["INFRASTRUCTURE_DEPLOYMENT_FAILURE"]
    assert infrastructure["Recovery Category"] == "`ENVIRONMENT_FAILURE`"
    assert "Kubernetes/Helm cluster unreachable" in infrastructure["Evidence Signals"]
    assert "Docker port already in use" in infrastructure["Evidence Signals"]
    assert "service readiness timeout" in infrastructure["Evidence Signals"]
    assert "Repair the deployment runtime first" in infrastructure["Next Action"]

    partial_logs = rows["PARTIAL_LOG_VISIBILITY"]
    assert partial_logs["Recovery Category"] == "`UNKNOWN`"
    assert "before assigning root cause" in partial_logs["Next Action"]
    assert "do not classify the log-access problem as the job failure cause" in partial_logs["Next Action"]


def _failure_pattern_rows(catalog_text):
    rows = {}
    headers = []
    for line in catalog_text.splitlines():
        if not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells[0] == "Pattern":
            headers = cells
            continue
        if not headers or set(cells[0]) <= {"-", " "}:
            continue
        rows[cells[0].strip("`")] = dict(zip(headers, cells))
    return rows


def test_pytorch_conversion_pins_recipe_key_metric_to_client_metric():
    repo_root = Path(__file__).resolve().parents[4]
    skill_root = repo_root / "skills" / "nvflare-convert-pytorch"
    skill_text = skill_root.joinpath("SKILL.md").read_text(encoding="utf-8")
    recipe_text = skill_root.joinpath("references/recipe-selection.md").read_text(encoding="utf-8")
    client_text = skill_root.joinpath("references/pytorch-client-api-conversion.md").read_text(encoding="utf-8")
    normalized_client = " ".join(client_text.split())
    validation_text = skill_root.joinpath("references/job-validation.md").read_text(encoding="utf-8")

    assert "`FedAvgRecipe.key_metric`" in skill_text
    assert "must exactly match the metric key sent in `FLModel.metrics`" in skill_text
    assert "`val_loss`" not in skill_text
    assert "`neg_loss`" in skill_text
    assert "unprotected recipe or adding only a disclaimer" in skill_text
    assert "key_metric=metric_name" in recipe_text
    assert 'metrics={"f1": f1}' in recipe_text
    assert 'key_metric="f1"' in recipe_text
    assert 'metrics={"neg_loss": -loss}' in recipe_text
    assert 'key_metric="neg_loss"' in recipe_text
    assert "`FLModel.metrics` must exactly match the selected recipe's `key_metric`" in normalized_client
    assert 'metrics={"neg_loss": -loss}' in normalized_client
    assert "recipe's `key_metric` exactly matches one key sent in" in validation_text
    assert "higher-is-better" in validation_text
