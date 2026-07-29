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

    datum_type_6 = rows["DATUM_OBJECT_TYPE_6"]
    assert datum_type_6["Recovery Category"] == "`FIXABLE_BY_CONFIG`"
    assert "cannot find handler for Datum Object Type 6" in datum_type_6["Evidence Signals"]
    assert "persists until attempt end" in datum_type_6["Evidence Signals"]
    assert "no later task download, client update, or aggregation succeeds" in datum_type_6["Evidence Signals"]
    assert "no client receives the task" in datum_type_6["Evidence Signals"]
    assert "Do not select `DATUM_OBJECT_TYPE_6` for later-success cases" in normalized_catalog
    assert "Do not recommend registering `TensorDecomposer` inside `client.py`" in normalized_catalog
    assert "unless evidence proves that code runs before task deserialization" in normalized_catalog

    exit_neg9 = rows["PROCESS_EXIT_NEG9"]
    assert exit_neg9["Recovery Category"] == "`UNKNOWN`"
    assert "`rc=-9`" in exit_neg9["Evidence Signals"]
    assert "do not infer OOM from `-9` alone" in exit_neg9["Next Action"]
    assert "at most one retry" in normalized_catalog
    assert "changed causal factor" in normalized_catalog


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
    assert "do not download artifacts for a healthy, successfully completed job" in normalized_skill


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


def test_pytorch_family_construction_owns_best_model_metric_policy():
    repo_root = Path(__file__).resolve().parents[4]
    skill_root = repo_root / "skills" / "nvflare-convert-pytorch"
    skill_text = skill_root.joinpath("SKILL.md").read_text(encoding="utf-8")
    recipe_text = skill_root.joinpath("references/recipe-selection.md").read_text(encoding="utf-8")
    client_text = skill_root.joinpath("references/pytorch-client-api-conversion.md").read_text(encoding="utf-8")
    validation_text = skill_root.joinpath("references/job-validation.md").read_text(encoding="utf-8")
    construction_text = repo_root.joinpath(
        "skills/nvflare-shared/references/pytorch-family-recipe-construction.md"
    ).read_text(encoding="utf-8")
    lightning_ddp_text = repo_root.joinpath(
        "skills/nvflare-convert-lightning/references/lightning-ddp-and-tracking.md"
    ).read_text(encoding="utf-8")
    normalized_construction = " ".join(construction_text.split())
    normalized_lightning_ddp = " ".join(lightning_ddp_text.split())

    assert "## Best-Model Metric" in construction_text
    assert "Only configure a source-derived `key_metric`" in normalized_construction
    assert "selected execution path delivers that exact metric to the server" in normalized_construction
    assert (
        "do not pass a source-derived `key_metric` or claim server-side best-model selection" in normalized_construction
    )
    assert "Its name must exactly match one key delivered by the client" in normalized_construction
    assert 'metrics={"neg_loss": -loss}' in construction_text
    assert 'key_metric="neg_loss"' in construction_text
    assert "ask or fail closed when the metric direction is unclear" in normalized_construction
    assert "`MetaKey.INITIAL_METRICS` bridge" in normalized_lightning_ddp
    assert "Only pass a source-derived `key_metric`" in normalized_lightning_ddp
    assert "unprotected recipe or adding only a disclaimer" in skill_text
    assert "key_metric=metric_name" not in recipe_text
    assert "when the selected execution path delivers that metric to the server" in recipe_text
    for consumer_text in (skill_text, recipe_text, client_text, validation_text):
        assert "pytorch-family-recipe-construction.md" in consumer_text
        assert 'metrics={"neg_loss": -loss}' not in consumer_text


def test_pytorch_model_exchange_owns_plain_pytorch_send_pattern():
    repo_root = Path(__file__).resolve().parents[4]
    model_exchange_text = repo_root.joinpath("skills/nvflare-shared/references/pytorch-model-exchange.md").read_text(
        encoding="utf-8"
    )
    client_reference_text = repo_root.joinpath(
        "skills/nvflare-convert-pytorch/references/pytorch-client-api-conversion.md"
    ).read_text(encoding="utf-8")

    assert "params = {k: v.detach().cpu() for k, v in model.state_dict().items()}" in model_exchange_text
    assert "assert all(isinstance(v, torch.Tensor) for v in params.values())" in model_exchange_text
    assert "v.detach().cpu()" not in client_reference_text
    assert "pytorch-model-exchange.md" in client_reference_text


def test_pytorch_family_construction_policy_is_canonical_and_capability_based():
    repo_root = Path(__file__).resolve().parents[4]
    pytorch_root = repo_root / "skills" / "nvflare-convert-pytorch"
    pytorch_skill = pytorch_root.joinpath("SKILL.md").read_text(encoding="utf-8")
    lightning_skill = repo_root.joinpath("skills/nvflare-convert-lightning/SKILL.md").read_text(encoding="utf-8")
    recipe_text = pytorch_root.joinpath("references/recipe-selection.md").read_text(encoding="utf-8")
    validation_text = pytorch_root.joinpath("references/job-validation.md").read_text(encoding="utf-8")
    construction_path = "../nvflare-shared/references/pytorch-family-recipe-construction.md"
    construction_text = repo_root.joinpath(
        "skills/nvflare-shared/references/pytorch-family-recipe-construction.md"
    ).read_text(encoding="utf-8")
    model_exchange_text = repo_root.joinpath("skills/nvflare-shared/references/pytorch-model-exchange.md").read_text(
        encoding="utf-8"
    )
    workflow_text = repo_root.joinpath("skills/nvflare-shared/references/conversion-workflow.md").read_text(
        encoding="utf-8"
    )
    hello_pt_job = repo_root.joinpath("examples/hello-world/hello-pt/job.py").read_text(encoding="utf-8")
    normalized_construction = " ".join(construction_text.split())
    normalized_model_exchange = " ".join(model_exchange_text.split())
    normalized_validation = " ".join(validation_text.split())

    assert construction_path in pytorch_skill
    assert construction_path in lightning_skill
    assert "whose `name` field is the public constructor keyword" in normalized_construction
    assert "Only pass a recipe keyword when its name is in the exposed-name set" in normalized_construction
    assert "When `server_expected_format` is exposed" in normalized_construction
    assert "When tensor-native transport was selected" in normalized_construction
    assert "Disk offload is a server memory optimization, not a model-exchange format" in normalized_construction
    assert "downloaded to server-side temporary files and materialized lazily" in normalized_construction
    assert "optimization only with `server_expected_format=ExchangeFormat.PYTORCH`" in normalized_construction
    assert "When `params_transfer_type` is exposed" in normalized_construction
    assert "single-process multi-GPU `torch.nn.DataParallel` stay in-process" in normalized_construction
    assert "Do not also pass `save_filename`" in normalized_construction
    assert "disk offload is not part of the model payload or exchange-format contract" in normalized_model_exchange
    assert "params_transfer_type" not in model_exchange_text
    assert "pytorch-family-recipe-construction.md" in workflow_text
    assert "`server_expected_format=ExchangeFormat.PYTORCH`" not in pytorch_skill
    assert "`enable_tensor_disk_offload=True`" not in pytorch_skill
    assert "`server_expected_format=ExchangeFormat.PYTORCH`" not in lightning_skill
    assert "`enable_tensor_disk_offload=True`" not in lightning_skill
    assert "launch_external_process=True," not in recipe_text
    assert "server_expected_format=ExchangeFormat.PYTORCH," not in recipe_text
    assert "enable_tensor_disk_offload=True," not in recipe_text
    assert 'recipe.add_decomposers(["nvflare.app_opt.pt.decomposers.TensorDecomposer"])' not in recipe_text
    assert "server_expected_format=" not in hello_pt_job
    assert "enable_tensor_disk_offload=" not in hello_pt_job
    assert "single source of truth" in recipe_text
    assert "cannot find handler for Datum Object Type 6" in normalized_validation
    assert "tensor-transport/decomposer configuration" in normalized_validation
    assert "server disk offload as an optimization" in normalized_validation
    assert "Do not patch NVFLARE runtime modules" in normalized_validation


def test_seed_skill_versions_stay_at_release_version():
    repo_root = Path(__file__).resolve().parents[4]

    for skill_path in repo_root.joinpath("skills").glob("*/SKILL.md"):
        skill_text = skill_path.read_text(encoding="utf-8")
        assert 'version: "0.1.0"' in skill_text, skill_path


def test_lightning_training_metrics_have_one_canonical_delivery_bridge():
    repo_root = Path(__file__).resolve().parents[4]
    skill_root = repo_root / "skills" / "nvflare-convert-lightning"
    skill_text = skill_root.joinpath("SKILL.md").read_text(encoding="utf-8")
    conversion_text = skill_root.joinpath("references/lightning-conversion.md").read_text(encoding="utf-8")
    validation_text = skill_root.joinpath("references/lightning-validation.md").read_text(encoding="utf-8")
    client_template = skill_root.joinpath("assets/lightning_client.py").read_text(encoding="utf-8")
    workflow_text = repo_root.joinpath("skills/nvflare-shared/references/conversion-workflow.md").read_text(
        encoding="utf-8"
    )
    aggregator_template = repo_root.joinpath("skills/nvflare-shared/assets/aggregator.py").read_text(encoding="utf-8")
    normalized_skill = " ".join(skill_text.split())
    normalized_conversion = " ".join(conversion_text.split())
    normalized_validation = " ".join(validation_text.split())
    normalized_workflow = " ".join(workflow_text.split())

    assert "calling `trainer.validate(...)` alone does not prove" in normalized_skill
    assert "## Training-result metric delivery" in conversion_text
    assert "`train_with_evaluation` setting is disabled or not exposed" in normalized_conversion
    assert "model.__fl_meta__" in conversion_text
    assert "MetaKey.INITIAL_METRICS" in client_template
    assert "trainer.validate" in client_template
    assert "trainer.fit" in client_template
    assert client_template.index("trainer.validate") < client_template.index("trainer.fit")
    assert "A terminal `Finished` state without that metric is incomplete validation" in normalized_validation
    assert "return them in the aggregated `FLModel.metrics`" in normalized_workflow
    assert "metrics=metrics or None" in aggregator_template


def test_pytorch_recipe_capability_profiles_include_non_fedavg_without_disk_offload():
    from nvflare.tool.recipe.recipe_cli import _load_catalog, _recipe_detail

    catalog = {entry["name"]: entry for entry in _load_catalog()}
    parameter_names = {}
    for recipe_name in ("fedavg-pt", "cyclic-pt", "fedeval-pt", "swarm-pt"):
        detail = _recipe_detail(catalog[recipe_name])
        parameter_names[recipe_name] = {parameter["name"] for parameter in detail["parameters"]}

    assert {"server_expected_format", "enable_tensor_disk_offload"} <= parameter_names["fedavg-pt"]
    for recipe_name in ("cyclic-pt", "fedeval-pt"):
        assert "server_expected_format" in parameter_names[recipe_name]
        assert "enable_tensor_disk_offload" not in parameter_names[recipe_name]
    assert "server_expected_format" not in parameter_names["swarm-pt"]
    assert "enable_tensor_disk_offload" not in parameter_names["swarm-pt"]


def test_pytorch_family_capability_evals_cover_fedeval_and_dataparallel():
    repo_root = Path(__file__).resolve().parents[4]
    pytorch_evals = json.loads(
        repo_root.joinpath("dev_tools/agent/skill_evals/nvflare-convert-pytorch/evals.json").read_text(encoding="utf-8")
    )
    lightning_evals = json.loads(
        repo_root.joinpath("dev_tools/agent/skill_evals/nvflare-convert-lightning/evals.json").read_text(
            encoding="utf-8"
        )
    )
    pytorch_by_id = {case["id"]: case for case in pytorch_evals["evals"]}
    lightning_by_id = {case["id"]: case for case in lightning_evals["evals"]}

    data_parallel = pytorch_by_id["pytorch-dataparallel-in-process"]["nvflare"]
    assert {item["id"] for item in data_parallel["mandatory_behavior"]} >= {
        "dataparallel-stays-in-process",
        "recipe-capability-profile",
    }
    assert {item["id"] for item in data_parallel["prohibited_behavior"]} >= {"no-dataparallel-external-process"}

    fed_eval = lightning_by_id["lightning-eval-only"]["nvflare"]
    assert {item["id"] for item in fed_eval["mandatory_behavior"]} >= {"fedeval-capability-profile"}
    assert {item["id"] for item in fed_eval["prohibited_behavior"]} >= {"no-unsupported-fedeval-disk-offload"}

    ddp = lightning_by_id["lightning-ddp-multigpu"]["nvflare"]
    assert {item["id"] for item in ddp["mandatory_behavior"]} >= {"ddp-key-metric-requires-server-delivery"}

    custom_metrics = lightning_by_id["lightning-custom-aggregation-with-server-metrics"]["nvflare"]
    assert {item["id"] for item in custom_metrics["mandatory_behavior"]} >= {
        "preserve-training-result-metrics",
        "custom-aggregator-preserves-metrics",
        "verify-server-metric-evidence",
    }
    assert {item["id"] for item in custom_metrics["prohibited_behavior"]} >= {
        "no-parameters-only-custom-aggregate",
        "no-second-manual-flare-send",
    }


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
    mandatory_by_id = {
        item["id"]: item["description"] for item in eval_data["evals"][0]["nvflare"]["mandatory_behavior"]
    }
    mandatory_ids = set(mandatory_by_id)
    prohibited_ids = {item["id"] for item in eval_data["evals"][0]["nvflare"]["prohibited_behavior"]}

    assert (
        "before any Python command imports user, PyTorch, NVFLARE, or declared dependency modules" in normalized_skill
    )
    assert "on a nonzero exit, stop validation" in normalized_skill
    assert "include every applicable requirements file" in normalized_dependency
    assert "`-r <requirements-a> -r <requirements-b> -c <constraints> nvflare`" in dependency_text
    assert "append `nvflare` to the same command" in normalized_dependency
    assert "authoritative evidence that NVFLARE is present" in normalized_dependency
    assert (
        "Do not append, install, or replace `nvflare` based only on missing package metadata" in normalized_dependency
    )
    assert "skip dependency installation and continue to validation" in normalized_dependency
    assert "Build an install plan only when inventory identifies at least one dependency" in normalized_dependency
    assert "Resolve the host-provided Python interpreter before choosing the installer target" in normalized_dependency
    assert "Never add `uv pip install --system`" in normalized_dependency
    assert (
        "only when the agent host explicitly supplies and identifies that system interpreter" in normalized_dependency
    )
    assert "parts of one planned install, not retries" in normalized_dependency
    assert "Run the selected combined canonical install command once." in dependency_text
    assert "stop dependency installation and validation for this conversion run" in normalized_dependency
    assert "report a redacted form of the command and product error" in normalized_dependency
    assert "replace credential-bearing option or environment values with `<redacted>`" in normalized_dependency
    assert "Do not retry with another installer, index, backend, package version" in normalized_dependency
    assert "do not purge caches, uninstall packages, or mutate `site-packages` directly" in normalized_dependency
    assert "first canonical install attempt, not autonomous retries or environment repair" in normalized_workflow
    assert "dependency-install-failure-is-terminal" in mandatory_ids
    assert (
        "one combined canonical dependency-install command" in mandatory_by_id["dependency-install-failure-is-terminal"]
    )
    assert "reports a redacted failed command" in mandatory_by_id["dependency-install-failure-is-terminal"]
    assert "no-dependency-install-retry-or-environment-surgery" in prohibited_ids


def test_pytorch_conversion_avoids_known_recipe_and_partition_retries():
    repo_root = Path(__file__).resolve().parents[4]
    construction_text = repo_root.joinpath(
        "skills/nvflare-shared/references/pytorch-family-recipe-construction.md"
    ).read_text(encoding="utf-8")
    workflow_text = repo_root.joinpath("skills/nvflare-shared/references/conversion-workflow.md").read_text(
        encoding="utf-8"
    )
    client_text = repo_root.joinpath(
        "skills/nvflare-convert-pytorch/references/pytorch-client-api-conversion.md"
    ).read_text(encoding="utf-8")
    lightning_skill = repo_root.joinpath("skills/nvflare-convert-lightning/SKILL.md").read_text(encoding="utf-8")
    validation_text = repo_root.joinpath("skills/nvflare-shared/references/validation-evidence.md").read_text(
        encoding="utf-8"
    )
    eval_data = json.loads(
        repo_root.joinpath("dev_tools/agent/skill_evals/nvflare-convert-pytorch/evals.json").read_text(encoding="utf-8")
    )
    normalized_construction = " ".join(construction_text.split())
    normalized_workflow = " ".join(workflow_text.split())
    normalized_validation = " ".join(validation_text.split())
    mandatory_ids = {item["id"] for item in eval_data["evals"][0]["nvflare"]["mandatory_behavior"]}
    prohibited_ids = {item["id"] for item in eval_data["evals"][0]["nvflare"]["prohibited_behavior"]}

    assert "pass `best_model_filename` only" in normalized_construction
    assert "Do not also pass `save_filename`" in normalized_construction
    assert "Make every array passed to an in-place shuffle writable" in workflow_text
    assert "positions = np.flatnonzero(frame[label_column].to_numpy() == label).copy()" in workflow_text
    assert "do not pass positional indices to `DataFrame.loc`" in normalized_workflow
    assert "shuffle writable **positional** index arrays" not in client_text
    assert '"Site Data Partitioning"' in lightning_skill
    assert "validate properties rather than guessed site sizes" in normalized_validation
    assert "complete, non-overlapping coverage" in normalized_validation
    assert "Assert exact per-site row counts only when" in validation_text
    assert {"safe-pandas-partitioning", "invariant-based-partition-validation"} <= mandatory_ids
    assert {"no-deprecated-save-filename-alias", "no-hardcoded-guessed-partition-counts"} <= prohibited_ids


def test_fedstats_reuses_named_sites_for_recipe_and_simulation():
    repo_root = Path(__file__).resolve().parents[4]
    skill_text = repo_root.joinpath("skills/nvflare-fed-stats/SKILL.md").read_text(encoding="utf-8")
    validation_text = repo_root.joinpath("skills/nvflare-fed-stats/references/stats-job-validation.md").read_text(
        encoding="utf-8"
    )
    eval_data = json.loads(
        repo_root.joinpath("dev_tools/agent/skill_evals/nvflare-fed-stats/evals.json").read_text(encoding="utf-8")
    )
    first_eval = eval_data["evals"][0]["nvflare"]
    mandatory_ids = {item["id"] for item in first_eval["mandatory_behavior"]}
    prohibited_ids = {item["id"] for item in first_eval["prohibited_behavior"]}
    normalized_skill = " ".join(skill_text.split())
    normalized_validation = " ".join(validation_text.split())

    assert "one site list" in normalized_skill
    assert "`from nvflare.recipe import SimEnv`" in normalized_skill
    assert "`from nvflare.recipe.fedstats import FedStatsRecipe` (never package root)" in normalized_skill
    assert "FedStatsRecipe(...," in skill_text
    assert "sites=sites, ...)" in skill_text
    assert "SimEnv(clients=sites, ...)" in skill_text
    assert "never use `SimEnv(num_clients=...)`" in normalized_skill
    assert "`from nvflare.recipe import SimEnv`" in normalized_validation
    assert "`from nvflare.recipe.fedstats import FedStatsRecipe`" in normalized_validation
    assert "never import `FedStatsRecipe` from `nvflare.recipe`" in normalized_validation
    assert "**Site-ownership preflight**" in validation_text
    assert "passed unchanged to both `FedStatsRecipe(sites=sites)` and `SimEnv(clients=sites)`" in normalized_validation
    assert "`SimEnv(num_clients=...)` is invalid for this recipe" in normalized_validation
    assert "fedstats-public-import-paths" in mandatory_ids
    assert "fedstats-explicit-sim-clients" in mandatory_ids
    assert "no-fedstats-package-root-import" in prohibited_ids
    assert "no-fedstats-sim-num-clients" in prohibited_ids

    recipe_docs = repo_root.joinpath("docs/user_guide/data_scientist_guide/available_recipes.rst").read_text(
        encoding="utf-8"
    )
    assert "from nvflare.recipe import SimEnv" in recipe_docs
    assert "from nvflare.recipe.fedstats import FedStatsRecipe" in recipe_docs

    for relative_path in (
        "examples/advanced/federated-statistics/df_stats/job.py",
        "examples/advanced/federated-statistics/image_stats/job.py",
    ):
        example_text = repo_root.joinpath(relative_path).read_text(encoding="utf-8")
        assert "sites=sites" in example_text
        assert "SimEnv(clients=sites" in example_text


def test_pytorch_family_conversion_documents_fl_entry_packaging_and_metric_keys():
    repo_root = Path(__file__).resolve().parents[4]
    skill_root = repo_root / "skills" / "nvflare-convert-huggingface"
    skill_text = skill_root.joinpath("SKILL.md").read_text(encoding="utf-8")
    conversion_text = skill_root.joinpath("references/huggingface-conversion.md").read_text(encoding="utf-8")
    validation_text = skill_root.joinpath("references/huggingface-validation.md").read_text(encoding="utf-8")
    shared_validation = repo_root.joinpath("skills/nvflare-shared/references/validation-evidence.md").read_text(
        encoding="utf-8"
    )
    shared_workflow = repo_root.joinpath("skills/nvflare-shared/references/conversion-workflow.md").read_text(
        encoding="utf-8"
    )
    pytorch_conversion = repo_root.joinpath(
        "skills/nvflare-convert-pytorch/references/pytorch-client-api-conversion.md"
    ).read_text(encoding="utf-8")
    lightning_conversion = repo_root.joinpath(
        "skills/nvflare-convert-lightning/references/lightning-conversion.md"
    ).read_text(encoding="utf-8")
    client_template = skill_root.joinpath("assets/client_with_eval.py").read_text(encoding="utf-8")
    server_model_template = skill_root.joinpath("assets/server_model.py").read_text(encoding="utf-8")
    job_template = skill_root.joinpath("assets/job.py").read_text(encoding="utf-8")
    recipe_text = repo_root.joinpath(
        "skills/nvflare-shared/references/pytorch-family-recipe-construction.md"
    ).read_text(encoding="utf-8")
    normalized_skill = " ".join(skill_text.split())
    normalized_conversion = " ".join(conversion_text.split())
    normalized_validation = " ".join(validation_text.split())
    normalized_shared_validation = " ".join(shared_validation.split())
    normalized_shared = " ".join(shared_workflow.split())
    normalized_pytorch = " ".join(pytorch_conversion.split())
    normalized_lightning = " ".join(lightning_conversion.split())
    normalized_recipe = " ".join(recipe_text.split())

    assert "generated `client.py` is an FL Client API entry point" in shared_workflow
    assert "auto-detect FL launch from environment variables" in shared_workflow
    assert "Exported app content is target-specific" in shared_workflow
    assert "Before asserting paths, inspect the exported job root" in normalized_shared
    assert "`app/custom`" in shared_workflow
    assert "`app_server/custom` plus each `app_<site>/custom`" in normalized_shared
    assert "Do not reuse a path assumption from another export" in normalized_shared
    assert 'recipe.add_server_file("model.py")' in shared_workflow
    assert "Every generated or project-local module referenced by server-side" in normalized_shared
    assert "Installed NVFLARE, framework, and third-party modules referenced by `class_path`" in normalized_shared
    assert "client apps and is not enough for per-site exports" in normalized_pytorch
    assert "client apps and is not enough for per-site exports" in normalized_lightning
    assert "third-party class paths stay runtime dependencies" in normalized_pytorch
    assert "third-party class paths stay runtime dependencies" in normalized_lightning
    assert "FL-only Client API entry point" in normalized_pytorch
    assert "FL-only Client API entry point" in normalized_lightning
    assert (
        "establish Lightning-local metrics but do not by themselves establish server delivery" in normalized_lightning
    )
    assert "`MetaKey.INITIAL_METRICS` before `trainer.fit(...)`" in normalized_lightning
    phase_markers = [
        "Inspect before editing",
        "Read applicable requirements",
        "Select the recipe from FL intent",
        "Convert with `references/huggingface-conversion.md`",
        "Adapt `assets/server_model.py` and `assets/job.py`",
        "Only after generated files exist",
        "Report the recipe",
    ]
    phase_positions = [skill_text.index(marker) for marker in phase_markers]
    assert phase_positions == sorted(phase_positions)
    assert "Complete each workflow phase before loading the next phase's reference" in normalized_skill
    assert (
        "Do not preload validation, state/DDP, broad workflow, dependency, or reporting references" in normalized_skill
    )
    assert "adapt `assets/client_with_eval.py` rather than drafting a new round loop" in normalized_skill
    assert "do not inspect recipe source or signatures" in normalized_skill.lower()
    assert "do not depend on repository examples" in normalized_skill.lower()
    assert "FL-only" in conversion_text
    assert "preserve source metric names" in skill_text
    assert "source-to-server mapping" in skill_text
    assert "CLIENT_API_TYPE" in conversion_text
    assert "federated=True" in conversion_text
    assert "federated=False" in conversion_text
    assert '`train_script="client.py"`' in normalized_conversion
    assert '`recipe.add_client_file(str(SOURCE_DIR / "client.py"))`' in normalized_conversion
    assert "Reject absolute `task_script_path` values" in normalized_conversion
    assert "client import is not enough when an export separates server and client apps" in normalized_conversion
    assert "Before asserting paths, inspect the exported job root" in normalized_conversion
    assert "`app/custom` for a unified export" in normalized_conversion
    assert "`app_server/custom` and each `app_<site>/custom`" in normalized_conversion
    assert "Do not reuse a path assumption from another export" in normalized_conversion
    assert "third-party class paths stay runtime dependencies" in normalized_conversion
    assert "Do not inspect `PTModel`, persistors, class loaders" in normalized_conversion
    assert "Adapt `../assets/client_with_eval.py`" in normalized_conversion
    assert "Adapt `../assets/server_model.py` and `../assets/job.py`" in normalized_conversion
    assert "same writable source directory" in normalized_conversion
    assert "rejects parent-traversal external-script paths" in normalized_conversion
    assert "inspecting NVFLARE implementation source" in normalized_conversion
    assert "flare.patch(trainer)" in client_template
    assert "HfArgumentParser(dataclass_types, allow_abbrev=False)" in client_template
    assert "while flare.is_running()" in client_template
    assert "return model" in server_model_template
    assert '"class_path": "server_model.ServerModel"' in job_template
    assert 'recipe.add_server_file(str(SOURCE_DIR / "server_model.py"))' in job_template
    assert 'recipe.add_server_file(str(SOURCE_DIR / "model.py"))' in job_template
    assert 'recipe.add_client_file(str(SOURCE_DIR / "client.py"))' in job_template
    assert "../model.py" not in job_template
    assert "SimEnv(" in job_template
    assert "recipe.execute(" in job_template
    assert "PTModel" not in job_template
    assert (
        "client entry's actual argument mechanism and actual dataclass types in parse-only mode"
        in normalized_validation
    )
    assert "generated client uses `HfArgumentParser`" in validation_text
    assert "same project and framework dataclass types" in normalized_validation
    assert "When it preserves `argparse` or another parser" in normalized_validation
    assert "reject every unused argument" in normalized_validation
    assert "`HfArgumentParser.parse_args_into_dataclasses()` can raise `ValueError`" in shared_validation
    assert "the wrapper itself exits zero" in normalized_validation
    assert "Version-check only fields claimed to belong to a framework" in normalized_validation
    assert "project-defined subclass field" in normalized_validation
    assert "actual parser accepts it" in normalized_validation
    assert "reduced one-round topology smoke" in normalized_validation
    assert "full-model validation blocked" in normalized_validation
    assert "estimate server and concurrent client memory" in normalized_validation
    assert "Capability-check optional diagnostics such as `free` or `nvidia-smi`" in normalized_validation
    assert "capacity evidence is unavailable, not conversion failure" in normalized_validation
    assert "reported default `max_steps=10`" in normalized_skill
    assert "Resolve exactly one per-round budget and encode it in one place" in conversion_text
    assert "user-requested local steps become generated `TrainingArguments.max_steps`" in normalized_conversion
    assert "user-requested local epochs become generated" in normalized_conversion
    assert "bounded generated default `max_steps=10`" in normalized_conversion
    assert "Do not also pass the same value through patch `local_steps` or `local_epochs`" in normalized_conversion
    assert "requested budget appears once" in normalized_validation
    assert "whole-job target in its progress bar" in normalized_conversion
    assert "make at most one expensive real-model retry" in normalized_validation
    assert "do not write one-off AST, class-loader, persistor, Recipe-source" in normalized_validation
    assert "`python job.py --export --export-dir <runtime-dir>/job_config`" in normalized_shared_validation
    assert "Do not accept a generated job-local export alias such as `--export_only`" in normalized_shared_validation
    assert "report it as a generated-code violation" in normalized_shared_validation
    assert "not automatically as a POSIX shell command" in normalized_recipe
    assert "default in-process Client API executor" in normalized_recipe
    assert "Do not import or call internal command-splitting helpers" in normalized_recipe
    assert "Do not import internal class loader helpers" in normalized_recipe
    assert "invent alternate export flags such as `--export_only`" in normalized_shared
    assert "import the NVFLARE recipe API before local argument parsing" in normalized_shared
    assert "`argparse.ArgumentParser(allow_abbrev=False)`" in normalized_shared
    assert "`parse_known_args()` is not needed" in normalized_shared
    assert "unknown and abbreviated options fail" in normalized_shared
    assert "`ArgumentParser(allow_abbrev=False)`" in normalized_skill
    assert "strict `parse_args()`; do not use `parse_known_args()`" in normalized_skill
    assert "`recipe show` validates only the selected recipe's module, class" in normalized_recipe
    assert "It does not advertise other symbols from `nvflare.recipe`" in normalized_recipe
    assert "use `SimEnv` with `recipe.execute(...)`" in normalized_recipe
    assert "Do not guess or directly import additional recipe-adjacent symbols" in normalized_recipe
    assert "Do not replace a failed local public check with web search" in normalized_recipe
    assert "Run intentional rejection checks" in normalized_shared_validation
    assert "through an assertion wrapper" in normalized_shared_validation
    assert "expected child failure as a failed top-level validation command" in normalized_shared_validation
    assert "shared assertion-wrapper rule" in normalized_validation
    assert "typo and abbreviation rejection cases" in normalized_validation
    assert "explicit non-client entry parameter" in normalized_validation
    assert "same resolved local model path and cache configuration" in normalized_validation
    assert "Instantiate the adapted `server_model.ServerModel`" in validation_text
    assert "exact full-model key agreement" in normalized_validation
    assert "class_path`, train script, custom aggregator" in normalized_shared
    assert "Its name must exactly match one key delivered by the client" in normalized_recipe
    assert "Do not reduce simulator `num_threads` below the requested client count" in recipe_text
    assert "guessed concurrency settings" in normalized_recipe


def test_unified_export_packages_server_class_path_file_with_server_targeted_api(tmp_path):
    from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe

    client_file = tmp_path / "client.py"
    model_file = tmp_path / "model.py"
    client_file.write_text("import model\n", encoding="utf-8")
    model_file.write_text("class SimpleNetwork:\n    pass\n", encoding="utf-8")

    recipe = FedAvgRecipe(
        name="unified_server_file_export",
        min_clients=2,
        num_rounds=1,
        model={"class_path": "model.SimpleNetwork", "args": {}},
        train_script=str(client_file),
    )
    recipe.add_server_file(str(model_file))

    export_root = tmp_path / "export"
    recipe.export(str(export_root))
    job_root = export_root / recipe.name

    assert (job_root / "app" / "custom" / "model.py").is_file()


def test_per_site_export_packages_server_class_path_file_with_server_targeted_api(tmp_path):
    from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
    from nvflare.recipe import set_per_site_config

    client_file = tmp_path / "client.py"
    model_file = tmp_path / "model.py"
    client_file.write_text("import model\n", encoding="utf-8")
    model_file.write_text("class SimpleNetwork:\n    pass\n", encoding="utf-8")

    recipe = FedAvgRecipe(
        name="server_file_export",
        min_clients=2,
        num_rounds=1,
        model={"class_path": "model.SimpleNetwork", "args": {}},
        train_script=str(client_file),
    )
    set_per_site_config(recipe, {"site-1": {"train_args": ""}, "site-2": {"train_args": ""}})
    recipe.add_server_file(str(model_file))
    recipe.add_client_file(str(model_file), clients=["site-1", "site-2"])

    export_root = tmp_path / "export"
    recipe.export(str(export_root))
    job_root = export_root / recipe.name

    assert (job_root / "app_server" / "custom" / "model.py").is_file()
    assert (job_root / "app_site-1" / "custom" / "model.py").is_file()
    assert (job_root / "app_site-2" / "custom" / "model.py").is_file()


def test_pytorch_family_validation_and_custom_aggregation_metric_contracts():
    repo_root = Path(__file__).resolve().parents[4]
    validation_text = repo_root.joinpath("skills/nvflare-shared/references/validation-evidence.md").read_text(
        encoding="utf-8"
    )
    normalized_validation = " ".join(validation_text.split())

    assert "workflow-native metric evidence" in validation_text
    assert "FedAvg/FedOpt/FedProx/SCAFFOLD-style training workflows" in validation_text
    assert "For FedEval" in validation_text
    assert "Cyclic or Swarm workflows" in validation_text
    expected_metrics_artifact_rule = (
        "missing expected metrics artifact is a validation failure only if the selected recipe/workflow installs"
    )
    assert expected_metrics_artifact_rule in normalized_validation

    eval_root = repo_root / "dev_tools" / "agent" / "skill_evals"
    for skill_name, case_id in {
        "nvflare-convert-pytorch": "pytorch-convert-custom-aggregation",
        "nvflare-convert-lightning": "lightning-convert-custom-aggregation",
        "nvflare-convert-huggingface": "huggingface-convert-custom-aggregation",
    }.items():
        suite = json.loads(eval_root.joinpath(skill_name, "evals.json").read_text(encoding="utf-8"))
        case = next(item for item in suite["evals"] if item["id"] == case_id)
        behavior_ids = {item["id"] for item in case["nvflare"]["mandatory_behavior"]}
        assertions_text = " ".join(case["assertions"])

        assert "custom-aggregator-preserves-metrics" in behavior_ids
        assert "custom-aggregator-missing-metrics-visible" in behavior_ids
        assert "finite numeric and boolean client metrics" in assertions_text
        assert "publishing a partial metric average" in assertions_text
