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
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from breastgfcl import ParallelServerGFedCL
from federated import client as client_module
from federated import state as state_module
from federated import transport as transport_module
from federated.runtime import execute_client, operation_seed
from model.client import ModifiedClient
from model.server import Server

from nvflare.collab.api.exceptions import CollabCallError, RunAborted
from nvflare.collab.api.group_call_context import FrozenResultQueue, ResultQueue


class UnreadableLoader:
    def __iter__(self):
        raise AssertionError("A server proxy must never read raw site data")


def completed_stream(site, payload=None, error=None):
    stream = ResultQueue(1, retain_history=False)
    if error is None:
        stream.append((site, payload))
    else:
        stream.append_failure(site, error)
    return stream


class CollabTransportTest(unittest.TestCase):
    def setUp(self):
        transport_collab = patch.object(transport_module, "collab", SimpleNamespace(is_aborted=False))
        transport_collab.start()
        self.addCleanup(transport_collab.stop)
        torch.manual_seed(107)
        self.opt = SimpleNamespace(
            device="cpu",
            batch_size=4,
            input_dim=8,
            nh=16,
            ni=16,
            nt=4,
            nd_out=4,
            num_clients=4,
            num_classes=2,
            num_task=3,
            noise_dim=5,
            no_bn=False,
            p=0.2,
            lr_e=0.001,
            lr_f=0.002,
            lr_g=0.003,
            lr_d=0.004,
            beta1=0.9,
            beta2=0.999,
            lambda_gan=0.5,
            replay=True,
            b=0.0,
            shuffle=True,
            seed=71,
            nvflare_timeout=1,
        )
        self.inputs = torch.randn(4, 8)
        self.labels = torch.tensor([0, 1, 0, 1])
        self.loader = DataLoader(TensorDataset(self.inputs, self.labels), batch_size=4, shuffle=True)
        self.graphs = [np.roll(np.eye(4, dtype=np.float32), task, axis=1) for task in range(3)]
        self.client = ModifiedClient(0, self.opt)
        self.discriminator = Server(self.opt).get_discriminator()
        self.client.set_server_discriminator(self.discriminator)
        self.proxy_bundle = {
            "weights": self.client.get_weights(),
            "training_state": self.client.get_training_state(),
            "task_metadata": {task: {"label_counts": torch.tensor([2, 2]), "batch_sizes": [4]} for task in range(3)},
        }
        for task in range(3):
            self.client.register_task(task, self.loader)

    def assert_nested_equal(self, first, second):
        if torch.is_tensor(first):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        elif isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second)
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_nested_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(first, second)

    def rng_state(self):
        return [random.getstate(), np.random.get_state(), torch.get_rng_state().clone()]

    def proxy(self, transport=None, bundle=None):
        proxy = transport_module.ProxyClient(0, self.opt, self.proxy_bundle if bundle is None else bundle, transport)
        proxy.set_server_discriminator(self.discriminator)
        return proxy

    def request(self):
        return {
            "task": 2,
            "epochs": 1,
            "seed": 711,
            "graphs": self.graphs,
            "weights": self.client.get_weights(),
            "training_state": self.client.get_training_state(),
            "discriminator": self.discriminator,
        }

    def site_bundle(self):
        return {
            "opt": vars(self.opt),
            "client_id": 0,
            "data": {
                task: {split: {"x": self.inputs, "y": self.labels} for split in ("train", "test")} for task in range(3)
            },
        }

    def collab_context(self, site="site-1"):
        return SimpleNamespace(site_name=site, is_aborted=False)

    def test_cpu_snapshot_detaches_arrays_and_populated_adam_state(self):
        result = execute_client("train", self.client, 2, self.graphs, self.loader, seed=117)
        noncontiguous = torch.arange(12, dtype=torch.float32).reshape(3, 4).T.requires_grad_()
        original = {"graphs": self.graphs, "result": result, "view": noncontiguous, "tuple": (1, None)}
        wire = state_module.cpu_tree(original)
        self.assertTrue(wire["view"].is_contiguous())
        self.assertFalse(wire["view"].requires_grad)
        self.assertTrue(wire["result"]["training_state"]["optimizer"]["state"])
        self.assert_nested_equal(wire, original)
        with torch.no_grad():
            noncontiguous.fill_(-1)
        self.assertFalse(torch.equal(wire["view"], noncontiguous))
        self.graphs[0].fill(-1)
        self.assertTrue(np.all(wire["graphs"][0] >= 0))

    def test_proxy_registers_exact_replay_metadata_without_reading_samples(self):
        proxy = self.proxy()
        for task in range(3):
            proxy.register_task(task, UnreadableLoader())
        state = proxy.get_training_state()
        self.assertEqual(set(state["task_label_counts"]), {0, 1, 2})
        for task in range(3):
            torch.testing.assert_close(state["task_label_counts"][task], torch.tensor([2, 2]))
            self.assertEqual(state["task_batch_sizes"][task], [4])
        state["task_label_counts"][0].fill_(0)
        torch.testing.assert_close(proxy.get_training_state()["task_label_counts"][0], torch.tensor([2, 2]))
        before = proxy.get_training_state()
        proxy.register_task(0, UnreadableLoader())
        self.assert_nested_equal(proxy.get_training_state(), before)

    def test_proxy_rejects_inconsistent_metadata(self):
        bundle = copy.deepcopy(self.proxy_bundle)
        bundle["task_metadata"][0]["label_counts"] = torch.tensor([2, 1])
        proxy = self.proxy(bundle=bundle)
        with self.assertRaisesRegex(ValueError, "replay metadata"):
            proxy.register_task(0, UnreadableLoader())
        self.assertNotIn(0, proxy.get_training_state()["task_label_counts"])

    def test_client_matches_shared_runtime_and_preserves_rng_for_every_operation(self):
        worker = client_module.BreastGFCLClient()
        with (
            patch.object(client_module, "collab", self.collab_context()),
            patch.object(client_module, "load_bundle", return_value=self.site_bundle()) as load,
        ):
            before = self.rng_state()
            worker.initialize()
            self.assert_nested_equal(self.rng_state(), before)
            load.assert_called_once_with("site.pt")
            for operation in ("encode", "train", "test"):
                with self.subTest(operation=operation):
                    request = self.request()
                    local = ModifiedClient(0, self.opt)
                    local.set_weights(request["weights"])
                    local.set_training_state(request["training_state"])
                    local.set_server_discriminator(self.discriminator)
                    loader = (
                        self.loader
                        if operation != "test"
                        else DataLoader(TensorDataset(self.inputs, self.labels), batch_size=4, shuffle=False)
                    )
                    expected = execute_client(operation, local, 2, self.graphs, loader, seed=request["seed"])
                    before = self.rng_state()
                    with patch.object(client_module, "execute_client", wraps=execute_client) as shared_runtime:
                        actual = getattr(worker, operation)(**request)
                    shared_runtime.assert_called_once()
                    self.assert_nested_equal(actual, expected)
                    self.assert_nested_equal(self.rng_state(), before)
            load.assert_called_once()

    def test_client_continues_from_returned_optimizer_and_scheduler_state(self):
        with (
            patch.object(client_module, "collab", self.collab_context()),
            patch.object(client_module, "load_bundle", return_value=self.site_bundle()),
        ):
            worker = client_module.BreastGFCLClient()
            worker.initialize()
            first = worker.train(**self.request())
            resumed = client_module.BreastGFCLClient()
            resumed.initialize()
            request = self.request()
            request.update(weights=first, training_state=first["training_state"], seed=712)
            expected = worker.train(**request)
            actual = resumed.train(**request)
            self.assert_nested_equal(actual, expected)
            self.assertEqual(actual["training_state"]["scheduler"]["last_epoch"], 2)
            self.assertTrue(actual["training_state"]["optimizer"]["state"])

    def test_client_rejects_bundle_for_another_site(self):
        worker = client_module.BreastGFCLClient()
        with (
            patch.object(client_module, "collab", self.collab_context("site-2")),
            patch.object(client_module, "load_bundle", return_value=self.site_bundle()),
        ):
            with self.assertRaises((RuntimeError, ValueError)):
                worker.initialize()

    def test_client_abort_does_not_load_data_or_execute_training(self):
        worker = client_module.BreastGFCLClient()
        context = self.collab_context()
        context.is_aborted = True
        with (
            patch.object(client_module, "collab", context),
            patch.object(client_module, "load_bundle") as load,
            patch.object(client_module, "execute_client") as execute,
        ):
            with self.assertRaisesRegex(RuntimeError, "aborted"):
                worker.initialize()
            with self.assertRaisesRegex(RuntimeError, "aborted"):
                worker.train(**self.request())
            load.assert_not_called()
            execute.assert_not_called()

    def test_bundle_loader_reads_only_the_current_application_data(self):
        with tempfile.TemporaryDirectory() as directory:
            app_dir = Path(directory)
            (app_dir / "config" / "data").mkdir(parents=True)
            bundle_path = app_dir / "config" / "data" / "site.pt"
            torch.save(self.site_bundle(), bundle_path)
            context = Mock()
            context.get_job_id.return_value = "prepared-job"
            workspace = Mock()
            workspace.get_app_dir.return_value = str(app_dir)
            collab_context = SimpleNamespace(workspace=workspace, fl_ctx=context)
            with patch.object(state_module, "collab", collab_context):
                loaded = state_module.load_bundle("site.pt")
                self.assert_nested_equal(loaded, self.site_bundle())
                workspace.get_app_dir.assert_called_once_with("prepared-job")
                for invalid in ("../other/site.pt", "/tmp/other.pt"):
                    with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                        state_module.load_bundle(invalid)

    def test_transport_addresses_one_site_and_preserves_payload(self):
        transport = transport_module.CollabTransport(self.opt)
        proxy = self.proxy(transport)
        transport.set_round(2, 3)
        payload = {"encodings": [torch.ones(2, 16)], "graph_embeddings": [torch.ones(2, 4)]}
        group = Mock()
        group.encode.return_value = completed_stream("site-1", payload)
        collab_context = Mock(is_aborted=False)
        collab_context.get_clients.return_value.return_value = group
        with patch.object(transport_module, "collab", collab_context):
            pending = transport.generate_encodings(proxy, 2, self.graphs, UnreadableLoader())
        collab_context.get_clients.assert_called_once_with(["site-1"])
        call_options = collab_context.get_clients.return_value.call_args.kwargs
        self.assertFalse(call_options["blocking"])
        self.assertEqual(call_options["timeout"], self.opt.nvflare_timeout)
        request = group.encode.call_args.kwargs
        self.assertEqual(request["seed"], operation_seed(self.opt.seed, 2, 3, 0, "encode"))
        self.assertNotIn("data", request)
        self.assertNotIn("dataloader", request)
        self.assert_nested_equal(request["graphs"], self.graphs)
        self.assert_nested_equal(transport.gather([pending]), [payload])

    def test_gather_preserves_requested_client_order(self):
        transport = transport_module.CollabTransport(self.opt)
        second = completed_stream("site-2", {"client": 2})
        first = completed_stream("site-1", {"client": 1})
        self.assertEqual(transport.gather([("site-1", first), ("site-2", second)]), [{"client": 1}, {"client": 2}])

    def test_failed_or_timed_out_site_never_returns_partial_batch(self):
        transport = transport_module.CollabTransport(self.opt)
        for cause in (ValueError("training failed"), TimeoutError("site timeout"), RunAborted("run aborted")):
            with self.subTest(cause=type(cause).__name__):
                error = CollabCallError("site-2", "train", cause)
                first = completed_stream("site-1", {"encoder": {"weight": torch.ones(2, 2)}})
                second = completed_stream("site-2", error=error)
                with self.assertRaises(CollabCallError) as caught:
                    transport.gather([("site-1", first), ("site-2", second)])
                self.assertIs(caught.exception, error)

    def test_failed_client_round_stops_before_aggregation_or_optimizer_advancement(self):
        opt = copy.copy(self.opt)
        opt.num_task, opt.num_rounds, opt.num_local_epochs, opt.max_in_flight = 1, 1, 1, 2
        transport = transport_module.CollabTransport(opt)
        clients = [transport_module.ProxyClient(i, opt, self.proxy_bundle, transport) for i in range(2)]
        server = Mock()
        server.get_discriminator.return_value = self.discriminator
        loaders = {i: {0: {"train": UnreadableLoader(), "test": UnreadableLoader()}} for i in range(2)}
        workflow = ParallelServerGFedCL.from_components(opt, server, Mock(), clients, loaders, transport)
        failure = CollabCallError("site-2", "train", TimeoutError("site timed out"))

        def addressed_group(sites):
            site = sites[0]
            group = Mock()
            group.encode.return_value = completed_stream(site, {"encodings": [], "graph_embeddings": []})
            group.train.return_value = completed_stream(
                site, self.client.get_weights(), error=failure if site == "site-2" else None
            )
            return Mock(return_value=group)

        context = Mock(is_aborted=False)
        context.get_clients.side_effect = addressed_group
        before = [client.get_weights() for client in clients]
        with (
            patch.object(transport_module, "collab", context),
            patch.object(workflow, "_generate_relational_graph", return_value=self.graphs[0]),
            patch("utils.server_utils.average_weights") as average,
        ):
            with self.assertRaises(CollabCallError) as caught:
                workflow.train_GFedCL()
            self.assertIs(caught.exception, failure)
            average.assert_not_called()
        server.update_learning_rate.assert_not_called()
        for client, previous in zip(clients, before):
            self.assert_nested_equal(client.get_weights(), previous)

    def test_gather_rejects_missing_duplicate_and_wrong_site_results(self):
        transport = transport_module.CollabTransport(self.opt)
        for items in ([], [("site-2", {})], [("site-1", {}), ("site-1", {})]):
            with self.subTest(items=items):
                stream = FrozenResultQueue(items, {}, len(items))
                with self.assertRaises((RuntimeError, ValueError, TypeError)):
                    transport.gather([("site-1", stream)])


if __name__ == "__main__":
    unittest.main()
