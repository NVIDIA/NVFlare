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
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedready.agents import (
    ALLOW_LIST_RULES,
    ClientAgent,
    GuardrailAgent,
    ServerAgent,
    _aggregate_visual_qc_sample_outputs,
    _visual_qc_backend_mode,
)
from fedready.agents.bridge import CodexWorkerBackend, _codex_skill_path, build_agent_backend
from fedready.agents.local_adapter import ensure_local_adapter_pipeline
from fedready.data.contracts.base import generated_contract_visual_qc_required
from fedready.data.contracts.training import CLASSIFICATION_TRAINING, SEGMENTATION_TRAINING
from fedready.job_data import run_fed_job_recipe
from fedready.job_train import (
    _build_fedavg_job_object,
    _training_code_spec_validation_errors,
    _training_local_simulation_errors,
    export_fedavg_training_job,
)


class ResearchScopeTestCase(unittest.TestCase):
    def test_public_backend_factory_is_codex_only(self) -> None:
        with TemporaryDirectory() as tmp:
            backend = build_agent_backend(
                kind="codex",
                run_dir=tmp,
                session_id="scope-test",
            )
            self.assertIsInstance(backend, CodexWorkerBackend)

            for unsupported in ("file", "openhands", "deterministic", "none"):
                with self.subTest(unsupported=unsupported):
                    with self.assertRaisesRegex(ValueError, "only 'codex'"):
                        build_agent_backend(
                            kind=unsupported,
                            run_dir=tmp,
                            session_id="scope-test",
                        )

    def test_agent_classes_reject_missing_backend(self) -> None:
        with self.assertRaisesRegex(ValueError, "live Codex agent backend is required"):
            ServerAgent(None)
        with self.assertRaisesRegex(ValueError, "live Codex agent backend is required"):
            ClientAgent("site-1", None)
        with self.assertRaisesRegex(ValueError, "live Codex agent backend is required"):
            GuardrailAgent(party_role="server", party_id="server", agent_backend=None)

    def test_visual_backend_is_local_vlm_only(self) -> None:
        with patch.dict("os.environ", {"FEDREADY_VISUAL_QC_BACKEND": "local_vlm"}):
            self.assertEqual(_visual_qc_backend_mode(), "local_vlm")
        for unsupported in ("agent", "disabled", "deterministic"):
            with self.subTest(unsupported=unsupported):
                with patch.dict("os.environ", {"FEDREADY_VISUAL_QC_BACKEND": unsupported}):
                    with self.assertRaisesRegex(ValueError, "only 'local_vlm'"):
                        _visual_qc_backend_mode()

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
        for required in (
            "local_vlm_model",
            "local_vlm_base_url",
            "local_vlm_api_key_env",
            "local_vlm_max_tokens",
            "query_image",
        ):
            self.assertIn(required, parameters)

    def test_training_contracts_require_visual_review_only_for_spatial_labels(
        self,
    ) -> None:
        self.assertTrue(SEGMENTATION_TRAINING.qc_contract["visual_qc_required"])
        self.assertFalse(CLASSIFICATION_TRAINING.qc_contract["visual_qc_required"])
        self.assertTrue(generated_contract_visual_qc_required({"visual_qc": {"required": True}}))
        self.assertFalse(generated_contract_visual_qc_required({"visual_qc": {"required": False}}))

    def test_visual_review_is_present_but_acquisition_quality_and_other_backends_are_not(
        self,
    ) -> None:
        research_dir = Path(__file__).resolve().parents[1]
        package_dir = research_dir / "fedready"
        self.assertTrue((package_dir / "job_data.py").exists())
        self.assertTrue((package_dir / "job_train.py").exists())
        self.assertFalse((package_dir / "job.py").exists())
        self.assertFalse((package_dir / "training_job.py").exists())
        self.assertFalse((package_dir / "cli.py").exists())
        self.assertFalse((package_dir / "openhands_tools.py").exists())
        self.assertTrue((package_dir / "data" / "qc.py").exists())
        self.assertFalse((package_dir / "reference_preparer.py").exists())
        self.assertTrue((research_dir / "scripts" / "prepare_references.py").exists())
        self.assertFalse((package_dir / "sample_quality.py").exists())
        self.assertFalse((package_dir / "sample_quality_contract.py").exists())
        self.assertTrue(_codex_skill_path("nvflare-orient").is_file())

        source = "\n".join(path.read_text(encoding="utf-8") for path in package_dir.rglob("*.py"))
        self.assertNotIn("OpenHands", source)
        self.assertNotIn("sample_quality", source)
        self.assertIn("local_vlm", source)
        self.assertIn("CLIENT.VISUAL_QC_EXTRACTION", source)
        self.assertIn("CLIENT.VISUAL_QC_EXTRACTION", {key[3] for key in ALLOW_LIST_RULES})

    def test_training_job_uses_current_client_api_executor(self) -> None:
        source = inspect.getsource(_build_fedavg_job_object)
        self.assertIn("ClientAPIExecutor", source)
        self.assertIn("ExecutionMode.IN_PROCESS", source)
        self.assertNotIn("PTInProcessClientAPIExecutor", source)
        self.assertIn("_build_fedavg_job_object", inspect.getsource(export_fedavg_training_job))

    def test_recipe_runner_uses_current_execute_entry_point(self) -> None:
        source = inspect.getsource(run_fed_job_recipe)
        self.assertIn(".execute(env)", source)
        self.assertNotIn(".run(env)", source)


if __name__ == "__main__":
    unittest.main()
