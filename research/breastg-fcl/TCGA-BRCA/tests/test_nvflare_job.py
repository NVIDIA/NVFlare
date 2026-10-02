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

import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import job as collab_job
import prepare_data
from configs.TCGA_BRCA import parse_args
from federated.state import capture_final_state
from model.server import Server


class NVFlareJobTest(unittest.TestCase):
    def setUp(self):
        self.opt = SimpleNamespace(
            num_clients=2,
            num_task=3,
            num_classes=2,
            batch_size=2,
            device="cpu",
            client_spatial_features=[np.ones((2, 3), dtype=np.float32) for _ in range(3)],
            client_temporal_features=[np.ones((2, 2), dtype=np.float32) for _ in range(3)],
            nvflare_workspace=None,
            nvflare_threads=None,
            nvflare_gpu=None,
        )
        self.loaders = {}
        for client in range(2):
            self.loaders[client] = {}
            for task in range(3):
                features = torch.arange(24, dtype=torch.float32).reshape(6, 4) + 100 * client + 10 * task
                labels = (torch.arange(6) + client + task) % 2
                dataset = TensorDataset(features, labels)
                self.loaders[client][task] = {
                    "train": DataLoader(Subset(dataset, [4, 1, 3]), batch_size=2, shuffle=True),
                    "test": DataLoader(Subset(dataset, [5, 0, 2]), batch_size=2),
                }
        clients = []
        for client in range(2):
            proxy = Mock()
            proxy.get_weights.return_value = {
                key: {"weight": torch.full((2, 2), float(client))} for key in ("encoder", "predictor", "generator")
            }
            proxy.get_training_state.return_value = {
                "optimizer": {"state": {}},
                "scheduler": {},
                "task_label_counts": {},
                "task_batch_sizes": {},
            }
            clients.append(proxy)
        server = Mock()
        server.get_discriminator.return_value = {"weight": torch.ones(2, 2)}
        attention = Mock()
        attention.state_dict.return_value = {"spatial_attention.weight": torch.ones(2, 2)}
        self.workflow = SimpleNamespace(
            clients=clients,
            dataloaders=self.loaders,
            server=server,
            dygat=attention,
            device=torch.device("cpu"),
        )

    def prepare(self):
        with patch.object(prepare_data, "ParallelServerGFedCL", return_value=self.workflow):
            return prepare_data.prepare_bundles(self.opt)

    def assert_no_raw_data(self, value):
        if isinstance(value, dict):
            for key, child in value.items():
                self.assertNotIn(key, ("data", "x", "y", "dataset", "dataloader"))
                self.assert_no_raw_data(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                self.assert_no_raw_data(child)

    def assert_split_matches(self, exported, dataset):
        expected_features = torch.stack([dataset[index][0] for index in range(len(dataset))])
        expected_labels = torch.tensor([dataset[index][1] for index in range(len(dataset))])
        torch.testing.assert_close(exported["x"], expected_features, rtol=0, atol=0)
        torch.testing.assert_close(exported["y"], expected_labels, rtol=0, atol=0)

    def test_preparation_preserves_each_existing_split_and_subset_order(self):
        workflow, server, sites = self.prepare()

        self.assertIs(workflow, self.workflow)
        self.assertEqual(len(sites), 2)
        self.assert_no_raw_data(server)
        for client_id, site in enumerate(sites):
            self.assertEqual(site["client_id"], client_id)
            self.assertNotIn("client_spatial_features", site["opt"])
            self.assertNotIn("client_temporal_features", site["opt"])
            for task in range(3):
                for split in ("train", "test"):
                    self.assert_split_matches(site["data"][task][split], self.loaders[client_id][task][split].dataset)
                labels = site["data"][task]["train"]["y"]
                metadata = server["clients"][client_id]["task_metadata"][task]
                torch.testing.assert_close(metadata["label_counts"], torch.bincount(labels, minlength=2))
                self.assertEqual(metadata["batch_sizes"], [2, 1])

    def test_export_keeps_raw_data_in_separate_site_apps_and_copies_only_code(self):
        _workflow, server, sites = self.prepare()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "source"
            project.mkdir()
            for package in ("model", "utils", "configs", "federated"):
                (project / package).mkdir()
                (project / package / "__init__.py").write_text("# code fixture\n")
                (project / package / "private.bin").write_bytes(b"must not be copied")
            (project / "breastgfcl.py").write_text("# coordinator fixture\n")
            (project / "data").mkdir()
            (project / "data" / "raw.pt").write_bytes(b"private source data")
            job_dir = root / "exported job"
            with patch.object(collab_job, "PROJECT", project):
                job_dir = collab_job.export_job(job_dir, server, sites)

            meta = json.loads((job_dir / "meta.json").read_text())
            self.assertEqual(
                meta["deploy_map"],
                {
                    "app_server": ["server"],
                    "app_site-1": ["site-1"],
                    "app_site-2": ["site-2"],
                },
            )
            self.assertEqual(meta["min_clients"], 2)
            server_files = list((job_dir / "app_server" / "config" / "data").iterdir())
            self.assertEqual([path.name for path in server_files], ["server.pt"])
            self.assert_no_raw_data(torch.load(server_files[0], weights_only=False))
            for client_id in range(2):
                app = job_dir / f"app_site-{client_id + 1}"
                self.assertEqual([path.name for path in (app / "config" / "data").iterdir()], ["site.pt"])
                local = torch.load(app / "config" / "data" / "site.pt", weights_only=False)
                self.assertEqual(local["client_id"], client_id)
                for task in range(3):
                    for split in ("train", "test"):
                        self.assert_split_matches(
                            local["data"][task][split], self.loaders[client_id][task][split].dataset
                        )
            for app_name in meta["deploy_map"]:
                custom_files = [path for path in (job_dir / app_name / "custom").rglob("*") if path.is_file()]
                self.assertTrue(custom_files)
                self.assertTrue(all(path.suffix == ".py" for path in custom_files))
                self.assertFalse((job_dir / app_name / "custom" / "data").exists())

    def test_simulator_runs_in_a_fresh_interpreter_with_explicit_client_count(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            job_dir = output / "job with spaces"
            self.opt.nvflare_threads = 2

            def child_process(command, **kwargs):
                self.assertEqual(
                    command[:3], [sys.executable, str(collab_job.PROJECT / "job.py"), "--run-exported-job"]
                )
                self.assertEqual(command[3], str(job_dir))
                self.assertEqual(command[command.index("--num-clients") + 1], "2")
                self.assertEqual(command[command.index("--nvflare-threads") + 1], "2")
                self.assertNotIn("--nvflare-gpu", command)
                self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
                self.assertFalse(kwargs.get("shell", False))
                kwargs["stdout"].write("child simulator completed\n")
                torch.save({"completed": True}, output / "final_state.pt")
                return SimpleNamespace(returncode=0)

            with patch.object(collab_job.subprocess, "run", side_effect=child_process) as run:
                collab_job.run_simulator(self.opt, job_dir, output)
            run.assert_called_once()
            self.assertIn("child simulator completed", (output / "simulator.log").read_text())

    def test_simulator_passes_explicit_workspace_and_gpu_to_child(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.opt.nvflare_workspace = str(output / "separate workspace")
            self.opt.nvflare_gpu = "1"
            torch.save({}, output / "final_state.pt")
            with patch.object(collab_job.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
                collab_job.run_simulator(self.opt, output / "job", output)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--nvflare-workspace") + 1], self.opt.nvflare_workspace)
            self.assertEqual(command[command.index("--nvflare-gpu") + 1], "1")

    def test_exported_recipe_executes_with_public_simenv_and_its_own_site_bundles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.opt.run_exported_job = str(root / "exported job")
            self.opt.nvflare_workspace = str(root / "workspace")
            self.opt.nvflare_threads = 2
            self.opt.nvflare_gpu = "1"
            with patch.object(collab_job, "make_recipe") as make_recipe:
                collab_job.execute_exported_job(self.opt)
            prepared = Path(self.opt.run_exported_job)
            make_recipe.assert_called_once_with(
                prepared / "app_server" / "config" / "data" / "server.pt",
                [prepared / f"app_site-{i}" / "config" / "data" / "site.pt" for i in (1, 2)],
            )
            execution_environment = make_recipe.return_value.execute.call_args.args[0]
            self.assertIsInstance(execution_environment, collab_job.SimEnv)
            self.assertEqual(execution_environment.clients, ["site-1", "site-2"])
            self.assertEqual(execution_environment.num_clients, 2)
            self.assertEqual(execution_environment.num_threads, 2)
            self.assertEqual(execution_environment.gpu_config, "1")
            self.assertEqual(execution_environment.workspace_root, self.opt.nvflare_workspace)

    def test_simulator_options_accept_one_resident_worker_per_client_and_single_gpu_group(self):
        for threads in (None, 2):
            for gpu in (None, "0", "1", "[0,1]", "[ 0 , 1 ]"):
                with self.subTest(threads=threads, gpu=gpu):
                    opt = copy.copy(self.opt)
                    opt.nvflare_threads, opt.nvflare_gpu = threads, gpu
                    collab_job.validate_simulator_options(opt)

    def test_invalid_simulator_options_fail_before_preparation_or_subprocess(self):
        cases = [
            ("--num-clients", "0"),
            ("--num-clients", "-1"),
            ("--nvflare-threads", "0"),
            ("--nvflare-threads", "-1"),
            ("--nvflare-threads", "1"),
            ("--nvflare-threads", "3"),
        ]
        cases.extend(
            ("--nvflare-gpu", gpu) for gpu in ("0,1", "[0],[1]", "", " ", "a", "[]", "[0,]", "[0,1", "0,1]", "[[0,1]]")
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for flag, value in cases:
                with self.subTest(flag=flag, value=value):
                    with (
                        patch.object(collab_job, "prepare_bundles") as prepare,
                        patch.object(collab_job, "execute_exported_job") as execute,
                        patch.object(collab_job.subprocess, "run") as run,
                        patch("sys.stderr", new_callable=io.StringIO) as error_output,
                    ):
                        with self.assertRaises(SystemExit) as caught:
                            collab_job.main(
                                ["--num-clients", "2", "--device", "cpu", "--output-dir", directory, flag, value]
                            )
                        self.assertEqual(caught.exception.code, 2)
                        if flag in ("--num-clients", "--nvflare-threads") and int(value) <= 0:
                            self.assertIn("positive", error_output.getvalue())
                        prepare.assert_not_called()
                        execute.assert_not_called()
                        run.assert_not_called()
                    opt = copy.copy(self.opt)
                    name = flag.removeprefix("--").replace("-", "_")
                    setattr(opt, name, value if name == "nvflare_gpu" else int(value))
                    opt.run_exported_job = str(output / "exported job")
                    with (
                        patch.object(collab_job, "make_recipe") as recipe,
                        patch.object(collab_job.subprocess, "run") as run,
                    ):
                        with self.assertRaises(ValueError):
                            collab_job.execute_exported_job(opt)
                        with self.assertRaises(ValueError):
                            collab_job.run_simulator(opt, output / "exported job", output)
                        recipe.assert_not_called()
                        run.assert_not_called()

    def test_simulator_requires_successful_exit_and_final_state_artifact(self):
        for return_code, artifact in ((1, True), (0, False)):
            with self.subTest(return_code=return_code, artifact=artifact):
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory)
                    if artifact:
                        torch.save({}, output / "final_state.pt")
                    with patch.object(
                        collab_job.subprocess, "run", return_value=SimpleNamespace(returncode=return_code)
                    ):
                        with self.assertRaisesRegex(RuntimeError, "NVFlare failed or produced no final state"):
                            collab_job.run_simulator(self.opt, output / "job", output)

    def test_final_state_captures_discriminator_adam_schedule_and_detached_copies(self):
        server = Server(SimpleNamespace(device="cpu", nh=8, nt=2, lr_d=0.001, beta1=0.9))
        latent = torch.randn(4, 8)
        graph_rows = torch.rand(4, 2)
        server.train_discriminator([latent], [graph_rows])
        server.update_learning_rate()
        self.workflow.server = server
        source_weight = torch.randn(2, 2, requires_grad=True)
        self.workflow.clients[0].get_weights.return_value["encoder"]["weight"] = source_weight

        snapshot = capture_final_state(self.workflow, {"accuracy": [0.5]}, [0])

        optimizer = snapshot["discriminator_optimizer"]
        self.assertTrue(optimizer["state"])
        self.assertEqual(snapshot["discriminator_scheduler"]["last_epoch"], 1)
        self.assertEqual(optimizer["param_groups"][0]["lr"], server.optimizer_D.param_groups[0]["lr"])
        for parameter_state in optimizer["state"].values():
            self.assertEqual(parameter_state["step"].item(), 1)
            for key in ("exp_avg", "exp_avg_sq"):
                self.assertEqual(parameter_state[key].device.type, "cpu")
                self.assertFalse(parameter_state[key].requires_grad)
        exported_weight = snapshot["weights"]["encoder"]["weight"]
        self.assertEqual(exported_weight.device.type, "cpu")
        self.assertFalse(exported_weight.requires_grad)
        torch.testing.assert_close(exported_weight, source_weight.detach())
        with torch.no_grad():
            source_weight.add_(1)
        server.train_discriminator([latent], [graph_rows])
        self.assertFalse(torch.equal(exported_weight, source_weight.detach()))
        self.assertTrue(all(state["step"].item() == 1 for state in optimizer["state"].values()))

    def test_concurrency_defaults_follow_client_count_and_allow_explicit_override(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments = ["--device", "cpu", "--output-dir", directory]
            for extra, expected in (([], 4), (["--num-clients", "12"], 8), (["--max-in-flight", "2"], 2)):
                with self.subTest(extra=extra):
                    opt = parse_args(arguments + extra)

                    self.assertEqual(opt.max_in_flight, expected)
                    self.assertEqual((opt.num_task, opt.nh, opt.noise_dim, opt.replay), (3, 800, 100, True))
                    self.assertFalse(any(key.startswith("ray_") for key in vars(opt)))

    def test_nonpositive_concurrency_is_rejected_before_data_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            for limit in (0, -1):
                with self.subTest(limit=limit), patch.object(collab_job, "prepare_bundles") as prepare:
                    with self.assertRaises(ValueError):
                        collab_job.main(
                            [
                                "--device",
                                "cpu",
                                "--smoke",
                                "--export-only",
                                "--output-dir",
                                directory,
                                "--max-in-flight",
                                str(limit),
                            ]
                        )
                    prepare.assert_not_called()

    def test_nvflare_entrypoint_exports_a_smoke_job_without_ray(self):
        script = textwrap.dedent(
            """
            import importlib.abc
            import sys

            class UnavailableRay(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "ray" or fullname.startswith("ray."):
                        raise ModuleNotFoundError("Ray is unavailable in this test", name=fullname)
                    return None

            sys.meta_path.insert(0, UnavailableRay())
            import main
            from federated.client import BreastGFCLClient
            from federated.server import BreastGFCLServer

            main.main(sys.argv[1:])
            assert not any(name == "ray" or name.startswith("ray.") for name in sys.modules)
        """
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke export"
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    "--device",
                    "cpu",
                    "--smoke",
                    "--export-only",
                    "--output-dir",
                    str(output),
                ],
                cwd=PROJECT_DIR,
                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            job = output / "nvflare_job" / "breastg_fcl"
            metadata = json.loads((job / "meta.json").read_text())
            self.assertEqual(metadata["min_clients"], 4)
            self.assertEqual(set(metadata["deploy_map"]), {"app_server", *(f"app_site-{i}" for i in range(1, 5))})
            server = torch.load(job / "app_server" / "config" / "data" / "server.pt", weights_only=False)
            self.assertEqual(len(server["clients"]), 4)
            self.assertEqual(server["opt"]["max_in_flight"], 4)
            self.assertEqual(server["opt"]["num_task"], 3)
            self.assertTrue(server["opt"]["replay"])
            self.assertFalse(any(key.startswith("ray_") or key == "verify_ray" for key in server["opt"]))
            for index in range(1, 5):
                site = torch.load(job / f"app_site-{index}" / "config" / "data" / "site.pt", weights_only=False)
                self.assertEqual(site["client_id"], index - 1)
                self.assertEqual(site["opt"]["max_in_flight"], 4)
                self.assertEqual(set(site["data"]), {0, 1, 2})
                for task in site["data"].values():
                    self.assertEqual(task["train"]["x"].shape, (8, 8))
                    self.assertEqual(task["test"]["x"].shape, (8, 8))
            self.assertFalse((output / "final_state.pt").exists())


if __name__ == "__main__":
    unittest.main()
