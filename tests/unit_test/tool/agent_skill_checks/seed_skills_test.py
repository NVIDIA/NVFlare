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

import json
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


def test_diagnose_job_uses_one_post_load_scope_check():
    repo_root = Path(__file__).resolve().parents[4]
    skill_text = repo_root.joinpath("skills/nvflare-diagnose-job/SKILL.md").read_text(encoding="utf-8")
    normalized_skill = " ".join(skill_text.split())

    assert normalized_skill.count("reported NVFLARE job failure signal") == 3
    assert normalized_skill.count("failed, stalled, timed out") == 1
    assert "Proceed only when the request includes" in normalized_skill
    assert "Stop this skill path and return to normal handling" in normalized_skill
    assert "MUST activate" not in skill_text
    assert "NEVER activate" not in skill_text
    assert "bounded failure-evidence collection for diagnosis" in normalized_skill
    assert "not normal result retrieval from an otherwise healthy, successfully completed job" in normalized_skill


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


def test_pytorch_family_conversion_registers_tensor_decomposer_at_recipe_boundary():
    repo_root = Path(__file__).resolve().parents[4]
    pytorch_root = repo_root / "skills" / "nvflare-convert-pytorch"
    pytorch_skill = pytorch_root.joinpath("SKILL.md").read_text(encoding="utf-8")
    lightning_skill = repo_root.joinpath("skills/nvflare-convert-lightning/SKILL.md").read_text(encoding="utf-8")
    recipe_text = pytorch_root.joinpath("references/recipe-selection.md").read_text(encoding="utf-8")
    validation_text = pytorch_root.joinpath("references/job-validation.md").read_text(encoding="utf-8")
    shared_profile = repo_root.joinpath("skills/nvflare-shared/references/pytorch-model-exchange.md").read_text(
        encoding="utf-8"
    )
    hello_pt_job = repo_root.joinpath("examples/hello-world/hello-pt/job.py").read_text(encoding="utf-8")
    normalized_pytorch_skill = " ".join(pytorch_skill.split())
    normalized_lightning_skill = " ".join(lightning_skill.split())
    normalized_validation = " ".join(validation_text.split())
    normalized_shared_profile = " ".join(shared_profile.split())

    for text in (normalized_pytorch_skill, normalized_lightning_skill):
        assert "`server_expected_format=ExchangeFormat.PYTORCH`" in text
        assert "`enable_tensor_disk_offload=True`" in text
        assert "`recipe.add_decomposers(...)`" in text
        assert "multi-GPU" in text
    assert "multi-process/multi-GPU" in normalized_pytorch_skill
    assert "DDP/multi-GPU" in normalized_lightning_skill
    assert "launch_external_process=True," not in recipe_text
    assert "server_expected_format=ExchangeFormat.PYTORCH," in recipe_text
    assert "enable_tensor_disk_offload=True," in recipe_text
    assert 'recipe.add_decomposers(["nvflare.app_opt.pt.decomposers.TensorDecomposer"])' in recipe_text
    assert "server_expected_format=" not in hello_pt_job
    assert "enable_tensor_disk_offload=" not in hello_pt_job
    assert "intentionally differs" in recipe_text
    assert "both server and client apps before the first" in normalized_shared_profile
    assert "Leave `launch_external_process` unset for CPU or single-GPU training" in normalized_shared_profile
    assert "cannot find handler for Datum Object Type 6" in normalized_validation
    assert "Do not patch NVFLARE runtime modules" in normalized_validation


def test_pytorch_conversion_stops_after_dependency_install_failure():
    repo_root = Path(__file__).resolve().parents[4]
    skill_text = repo_root.joinpath("skills/nvflare-convert-pytorch/SKILL.md").read_text(encoding="utf-8")
    dependency_text = repo_root.joinpath("skills/nvflare-shared/references/dependency-install.md").read_text(
        encoding="utf-8"
    )
    workflow_text = repo_root.joinpath("skills/nvflare-shared/references/conversion-workflow.md").read_text(
        encoding="utf-8"
    )
    eval_data = json.loads(
        repo_root.joinpath("dev_tools/agent/skill_evals/nvflare-convert-pytorch/evals.json").read_text(encoding="utf-8")
    )
    normalized_skill = " ".join(skill_text.split())
    normalized_dependency = " ".join(dependency_text.split())
    normalized_workflow = " ".join(workflow_text.split())
    mandatory_ids = {item["id"] for item in eval_data["evals"][0]["nvflare"]["mandatory_behavior"]}
    prohibited_ids = {item["id"] for item in eval_data["evals"][0]["nvflare"]["prohibited_behavior"]}

    assert (
        "before any Python command imports user, PyTorch, NVFLARE, or declared dependency modules" in normalized_skill
    )
    assert "on a nonzero exit, stop validation" in normalized_skill
    assert "Run the selected canonical install command once." in dependency_text
    assert "stop dependency installation and validation for this conversion run" in normalized_dependency
    assert "Do not retry with another installer, index, backend, package version" in normalized_dependency
    assert "do not purge caches, uninstall packages, or mutate `site-packages` directly" in normalized_dependency
    assert "first canonical install attempt, not autonomous retries or environment repair" in normalized_workflow
    assert "dependency-install-failure-is-terminal" in mandatory_ids
    assert "no-dependency-install-retry-or-environment-surgery" in prohibited_ids


def test_pytorch_conversion_avoids_known_recipe_and_partition_retries():
    repo_root = Path(__file__).resolve().parents[4]
    skill_root = repo_root / "skills" / "nvflare-convert-pytorch"
    recipe_text = skill_root.joinpath("references/recipe-selection.md").read_text(encoding="utf-8")
    client_text = skill_root.joinpath("references/pytorch-client-api-conversion.md").read_text(encoding="utf-8")
    validation_text = repo_root.joinpath("skills/nvflare-shared/references/validation-evidence.md").read_text(
        encoding="utf-8"
    )
    eval_data = json.loads(
        repo_root.joinpath("dev_tools/agent/skill_evals/nvflare-convert-pytorch/evals.json").read_text(encoding="utf-8")
    )
    normalized_recipe = " ".join(recipe_text.split())
    normalized_client = " ".join(client_text.split())
    normalized_validation = " ".join(validation_text.split())
    mandatory_ids = {item["id"] for item in eval_data["evals"][0]["nvflare"]["mandatory_behavior"]}
    prohibited_ids = {item["id"] for item in eval_data["evals"][0]["nvflare"]["prohibited_behavior"]}

    assert "set `best_model_filename` only" in normalized_recipe
    assert "Do not also set `save_filename`" in normalized_recipe
    assert "shuffle writable **positional** index arrays" in client_text
    assert '`positions = np.flatnonzero(frame["label"].to_numpy() == label).copy()`' in client_text
    assert "do not pass positional indices to `DataFrame.loc`" in normalized_client
    assert "validate properties rather than guessed site sizes" in normalized_validation
    assert "complete, non-overlapping coverage" in normalized_validation
    assert "Assert exact per-site row counts only when" in validation_text
    assert {"safe-pandas-partitioning", "invariant-based-partition-validation"} <= mandatory_ids
    assert {"no-deprecated-save-filename-alias", "no-hardcoded-guessed-partition-counts"} <= prohibited_ids
