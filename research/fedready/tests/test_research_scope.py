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

from __future__ import annotations

import inspect
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedready.agents.bridge import _codex_skill_path
from fedready.agents.client import adapter_qc
from fedready.agents.client.adapter import AdapterRuntimeResult
from fedready.agents.client.adapter_qc import ensure_local_adapter_pipeline
from fedready.agents.client.agent import ClientAgent
from fedready.agents.client.visual_qc import _aggregate_visual_qc_sample_outputs
from fedready.agents.common import adapter_source_digest
from fedready.agents.guardrail import ALLOW_LIST_RULES, GuardrailAgent
from fedready.agents.server.agent import ServerAgent
from fedready.data.contracts.base import generated_contract_visual_qc_required
from fedready.data.contracts.training import CLASSIFICATION_TRAINING, SEGMENTATION_TRAINING
from fedready.flare.training_job import CLIENT_TRAINING_LAUNCHER_SCRIPT, _build_fedavg_job_object
from fedready.training_code import _training_code_spec_validation_errors, _training_local_simulation_errors


class ResearchScopeTestCase(unittest.TestCase):
    def test_training_response_does_not_infer_required_fields_from_aliases(self) -> None:
        backend = mock.Mock()
        backend.request.return_value = {
            "schema_version": "fedready.training_code_spec.v1",
            "status": "implemented",
            "package": {"root": "training"},
            "entry_point": {"script": "train.py"},
            "model_class": {"module": "training.model", "class": "Model"},
        }
        turn = ServerAgent(backend).implement_training_code(
            task={"task": "segmentation"}, extraction_summary={}, training_plan={}, code_workspace="/tmp"
        )
        errors = _training_code_spec_validation_errors(turn.output, require_local_simulation=False)
        self.assertEqual(
            {error["key"] for error in errors if error["kind"] == "missing_required_key"},
            {"package_dir", "entry_script", "model_class_path"},
        )

    def test_extraction_summary_is_verifier_owned_without_a_model_call(self) -> None:
        backend = mock.Mock()
        summary = ServerAgent(backend).summarize_extraction_results(
            task={"task": "segmentation"}, extraction_results={}
        )
        backend.request.assert_not_called()
        self.assertEqual(summary.output["extracted_client_count"], 0)
        self.assertEqual(summary.output["total_extracted_samples"], 0)

    def test_agent_classes_reject_missing_backend(self) -> None:
        with self.assertRaisesRegex(ValueError, "live Codex agent backend is required"):
            ServerAgent(None)
        with self.assertRaisesRegex(ValueError, "live Codex agent backend is required"):
            ClientAgent("site-1", None)
        with self.assertRaisesRegex(ValueError, "live Codex agent backend is required"):
            GuardrailAgent(party_role="server", party_id="server", agent_backend=None)

    def test_deterministic_training_specs_cannot_bypass_live_preflight(self) -> None:
        errors = _training_local_simulation_errors({"deterministic_ground_truth": True})
        self.assertEqual("local_simulation_missing", errors[0]["kind"])
        validation_errors = _training_code_spec_validation_errors(
            {"deterministic_ground_truth": True},
            require_local_simulation=False,
        )
        self.assertIn(
            "deterministic_training_spec",
            {error["kind"] for error in validation_errors},
        )

    def test_local_visual_review_rejects_deterministic_output(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "deterministic_ground_truth"):
            _aggregate_visual_qc_sample_outputs(
                base_output={"deterministic_ground_truth": True},
                sample_outputs=[],
                qc_context={},
            )

    def test_adapter_pipeline_requires_local_visual_backend_inputs(self) -> None:
        parameters = inspect.signature(ensure_local_adapter_pipeline).parameters
        self.assertNotIn("reuse_existing_attestations", parameters)
        for required in (
            "local_vlm_model",
            "local_vlm_base_url",
            "local_vlm_api_key_env",
            "local_vlm_max_tokens",
            "query_image",
        ):
            self.assertIn(required, parameters)

    def test_adapter_pipeline_rechecks_forged_pass_states(self) -> None:
        for failed_stage in ("executable_preflight", "visual_review"):
            with self.subTest(stage=failed_stage), TemporaryDirectory() as tmp:
                root = Path(tmp)
                script = root / "adapter.py"
                script.write_text("pass\n")
                forged = {"status": "passed", "adapter_sha256": adapter_source_digest(script)}
                for name in (adapter_qc.ADAPTER_PREFLIGHT_STATE_FILE, adapter_qc.ADAPTER_PREFLIGHT_QC_STATE_FILE):
                    (root / name).write_text(json.dumps(forged))
                preflight = AdapterRuntimeResult(
                    stage="executable_preflight",
                    status="failed" if failed_stage == "executable_preflight" else "passed",
                    diagnostic="fresh executable check",
                )
                with (
                    mock.patch.object(adapter_qc, "execute_local_adapter", return_value=preflight) as execute,
                    mock.patch.object(
                        adapter_qc,
                        "review_local_adapter_preflight",
                        return_value=AdapterRuntimeResult(
                            stage="visual_review", status="failed", diagnostic="fresh visual check"
                        ),
                    ) as review,
                ):
                    result = ensure_local_adapter_pipeline(
                        workspace=root,
                        local_data_path=root,
                        task="optic disc segmentation",
                        max_records=10,
                        min_records=1,
                        timeout_seconds=1,
                        local_vlm_model="local",
                        local_vlm_base_url="http://127.0.0.1:8001/v1",
                        local_vlm_api_key_env="LOCAL_KEY",
                        local_vlm_max_tokens=1,
                        query_image=mock.Mock(),
                    )
                    self.assertFalse(result.passed)
                    execute.assert_called_once()
                    self.assertEqual(review.call_count, int(failed_stage == "visual_review"))

    def test_training_contracts_require_visual_review_only_for_spatial_labels(
        self,
    ) -> None:
        self.assertTrue(SEGMENTATION_TRAINING.qc_contract["visual_qc_required"])
        self.assertFalse(CLASSIFICATION_TRAINING.qc_contract["visual_qc_required"])
        self.assertTrue(generated_contract_visual_qc_required({"visual_qc": {"required": True}}))
        self.assertFalse(generated_contract_visual_qc_required({"visual_qc": {"required": False}}))

    def test_visual_review_is_allow_listed_and_codex_skill_is_available(self) -> None:
        self.assertIn("CLIENT.VISUAL_QC_EXTRACTION", {key[3] for key in ALLOW_LIST_RULES})
        self.assertTrue(_codex_skill_path("nvflare-orient").is_file())

    def test_training_job_always_uses_the_validating_client_runtime(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "fedready_task_training"
            package.mkdir()
            (package / "__init__.py").touch()
            (package / "train.py").write_text("pass\n")
            (package / "model.py").write_text("class Model: pass\n")
            job = _build_fedavg_job_object(
                job_name="test_runtime_gate",
                training_plan={"algorithm": {"num_rounds": 2}},
                training_code_spec={
                    "package_dir": str(package),
                    "model_class_path": "fedready_task_training.model.Model",
                    "entry_script": "train.py",
                },
                client_configs={
                    "SITE_A": {
                        "status": "ready",
                        "task_script_path": "bypass.py",
                        "task_script_args": "--dataset-root data --client-id SITE_A",
                    }
                },
                min_clients=None,
            )
            job.export_job(str(root / "jobs"))
            client = root / "jobs/test_runtime_gate/app_SITE_A"
            executor = json.loads((client / "config/config_fed_client.json").read_text())["executors"][0]["executor"]
            self.assertTrue(executor["path"].endswith(".ClientAPIExecutor"))
            self.assertEqual(executor["args"]["execution_mode"], "in_process")
            self.assertEqual(executor["args"]["task_script_path"], CLIENT_TRAINING_LAUNCHER_SCRIPT)
            self.assertTrue((client / "custom" / CLIENT_TRAINING_LAUNCHER_SCRIPT).is_file())


if __name__ == "__main__":
    unittest.main()
