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

import copy
import errno
import json
import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

import nvflare.app_opt.pt.disk_fedavg as disk
import nvflare.app_opt.pt.lazy_tensor_dict as lazy
from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, Task, TaskCompletionStatus
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, FLMetaKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.wf_comm_server import WFCommServer
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.abstract.model import make_model_learnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.lazy_tensor_dict import _TempDirRef, safetensors_refs
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.app_opt.pt.recipes.fedprox import FedProxRecipe
from nvflare.app_opt.pt.tensor_downloader import TensorDownloadable
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.fuel.utils.class_utils import instantiate_class
from nvflare.fuel.utils.fobs import FOBSContextKey


class Cell:
    def __init__(self):
        self.context = {FOBSContextKey.TENSOR_DISK_OFFLOAD: True}

    def get_fobs_context(self):
        return dict(self.context)

    def update_fobs_context(self, props):
        self.context.update(props)


def context(root):
    ctx = FLContext()
    ctx.set_prop(FLContextKey.APP_ROOT, str(root), private=True, sticky=False)
    ctx.set_prop(ReservedKey.RUN_NUM, "disk-fedavg-test", private=True, sticky=False)
    ctx.set_prop(AppConstants.CURRENT_ROUND, 0, private=True, sticky=False)
    ctx.set_prop(FLContextKey.RUN_ABORT_SIGNAL, Signal(), private=True, sticky=False)
    cell = Cell()
    engine = SimpleNamespace(get_cell=lambda: cell, fire_event=lambda *_: None)
    ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=False)
    return ctx


def refs(root, name, tensors, owned=True):
    directory = root / name
    directory.mkdir()
    path = directory / "model.safetensors"
    save_file(tensors, path)
    return safetensors_refs(str(path), _TempDirRef(str(directory)) if owned else None)


def model(params, client="site-1", steps=1, metrics=None, kind=ParamsType.FULL):
    return FLModel(
        params=params,
        params_type=kind,
        metrics=metrics,
        meta={"client_name": client, FLMetaKey.NUM_STEPS_CURRENT_ROUND: steps},
    )


def aggregator(root, **kwargs):
    result = disk.DiskFedAvgAggregator(**kwargs)
    result.handle_event(EventType.START_RUN, context(root))
    return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.bool])
def test_full_weights_scalar_buffers_metrics_and_cleanup(tmp_path, dtype):
    first = torch.tensor([1, 0], dtype=dtype)
    second = torch.tensor([0, 1], dtype=dtype)
    aggr = aggregator(tmp_path, aggregation_weights={"site-1": 3.0})
    aggr.accept_model(
        model(refs(tmp_path, "one", {"w": first, "count": torch.tensor(1)}), steps=2, metrics={"loss": 2.0})
    )
    aggr.accept_model(
        model(refs(tmp_path, "two", {"w": second, "count": torch.tensor(8)}), client="site-2", metrics={"loss": 9.0})
    )
    result = aggr.aggregate_model()
    expected = ((first.float() * 6 + second.float()) / 7).to(dtype if dtype.is_floating_point else torch.float32)
    assert torch.equal(result.params["w"].materialize(), expected)
    assert result.params["count"].materialize().item() == 2.0
    assert result.metrics["loss"] == 3.0
    info = result.meta[AppConstants.METRICS_AGGREGATION_INFO]
    assert [site["weight"] for site in info["site_weights"]] == [6.0, 1.0]
    assert aggr.fl_ctx.get_prop(AppConstants.AGGREGATION_STATS)["fully_matched_keys"] == 2
    assert not (tmp_path / "one").exists()
    assert not (tmp_path / "two").exists()
    assert not (tmp_path / (disk.CURRENT_MODEL + ".next")).exists()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", [ParamsType.FULL, ParamsType.DIFF])
@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_matches_existing_fedavg_rounding(tmp_path, dtype, kind, order):
    aggr = aggregator(tmp_path, aggregation_weights={"site-2": 3.0})
    generator = torch.Generator().manual_seed(731)
    values = [torch.randn(1024, generator=generator).to(dtype) for _ in range(3)]
    base = refs(tmp_path, "initial", {"w": values[2]}, owned=False)
    aggr.fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, make_model_learnable(base, {}), private=True, sticky=True)
    for i in order:
        aggr.accept_model(model(refs(tmp_path, str(i), {"w": values[i]}), client=f"site-{i + 1}", steps=2, kind=kind))
    weights = [2.0, 6.0]
    expected = values[order[0]].mul(weights[order[0]])
    expected.add_(values[order[1]], alpha=weights[order[1]])
    expected.div_(sum(weights))
    if kind == ParamsType.DIFF:
        expected = values[2] + expected
    tensor = aggr.aggregate_model().params["w"].materialize()
    assert tensor.dtype == dtype
    assert torch.equal(tensor, expected)


def test_aggregation_publishes_stats_in_current_round_context(tmp_path):
    aggr = aggregator(tmp_path)
    aggr.accept_model(model(refs(tmp_path, "one", {"w": torch.ones(1)})))
    round_ctx = context(tmp_path)
    round_ctx.set_prop(AppConstants.CURRENT_ROUND, 1, private=True, sticky=False)
    aggr.handle_event(AppEventType.BEFORE_AGGREGATION, round_ctx)
    result = aggr.aggregate_model()
    assert result.current_round == 1
    assert round_ctx.get_prop(AppConstants.AGGREGATION_STATS)["accepted_contributions"] == 1


@pytest.mark.parametrize("kind", [ParamsType.FULL, ParamsType.DIFF])
def test_sparse_keys_exclusions_and_per_key_denominator(tmp_path, kind):
    aggr = aggregator(tmp_path, exclude_vars="skip", aggregation_weights={"site-2": 3.0})
    base = refs(
        tmp_path,
        "initial",
        {"w": torch.tensor([10.0]), "only": torch.tensor([20.0]), "skip": torch.tensor(7.0)},
        owned=False,
    )
    aggr.fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, make_model_learnable(base, {}), private=True, sticky=True)
    aggr.accept_model(
        model(
            refs(tmp_path, "one", {"w": torch.tensor([2.0]), "only": torch.tensor([6.0]), "skip": torch.tensor(99.0)}),
            kind=kind,
        )
    )
    aggr.accept_model(model(refs(tmp_path, "two", {"w": torch.tensor([4.0])}), client="site-2", kind=kind))
    result = aggr.aggregate_model()
    assert result.params_type == ParamsType.FULL
    assert result.params["w"].materialize().item() == (13.5 if kind == ParamsType.DIFF else 3.5)
    assert result.params["only"].materialize().item() == (26.0 if kind == ParamsType.DIFF else 6.0)
    if kind == ParamsType.DIFF:
        assert result.params["skip"].materialize().item() == 7.0
    else:
        assert "skip" not in result.params
    assert base["w"].materialize().item() == 10.0
    stats = aggr.fl_ctx.get_prop(AppConstants.AGGREGATION_STATS)
    assert stats["fully_matched_keys"] == 1
    assert stats["partially_matched_keys"] == 1
    assert stats["skipped_keys"] == 1


@pytest.mark.parametrize("bad", ["shape", "dtype", "memory", "mixed"])
def test_invalid_contribution_fails_round_without_publication(tmp_path, bad):
    saved = tmp_path / disk.SAVED_MODEL
    save_file({"w": torch.full((2,), 9.0)}, saved)
    before = saved.read_bytes()
    aggr = aggregator(tmp_path)
    aggr.accept_model(model(refs(tmp_path, "one", {"w": torch.ones(2)})))
    value = torch.ones(1) if bad == "shape" else torch.ones(2, dtype=torch.float64 if bad == "dtype" else torch.float32)
    params = {"w": value} if bad == "memory" else refs(tmp_path, "bad", {"w": value})
    with pytest.raises((ValueError, TypeError)):
        aggr.accept_model(model(params, kind=ParamsType.DIFF if bad == "mixed" else ParamsType.FULL))
    with pytest.raises(RuntimeError, match="already failed"):
        aggr.accept_model(model(refs(tmp_path, "late", {"w": torch.ones(2)})))
    with pytest.raises(RuntimeError, match="refusing to publish"):
        aggr.aggregate_model()
    assert saved.read_bytes() == before
    assert not (tmp_path / disk.CURRENT_MODEL).exists()
    assert not (tmp_path / (disk.CURRENT_MODEL + ".next")).exists()
    assert not any((tmp_path / name).exists() for name in ("one", "bad", "late"))


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool])
@pytest.mark.parametrize("promoted_diff", [False, True])
def test_multiple_diff_rounds_accept_promoted_integer_base(tmp_path, dtype, promoted_diff):
    aggr = aggregator(tmp_path)
    base = refs(tmp_path, "initial", {"count": torch.tensor(0, dtype=dtype)}, owned=False)
    for round_number in range(2):
        aggr.fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, make_model_learnable(base, {}), private=True, sticky=True)
        aggr.accept_model(
            model(
                refs(
                    tmp_path,
                    str(round_number),
                    {"count": torch.tensor(1, dtype=torch.float32 if promoted_diff else dtype)},
                ),
                kind=ParamsType.DIFF,
            )
        )
        base = aggr.aggregate_model().params
        assert base["count"].materialize().dtype == torch.float32
        assert base["count"].materialize().item() == round_number + 1


@pytest.mark.parametrize("metrics", [None, {"nested": {"x": 2}, "loss": 4.0}])
def test_missing_and_non_scalar_metrics_follow_fedavg(tmp_path, metrics):
    aggr = aggregator(tmp_path)
    aggr.accept_model(model(refs(tmp_path, "one", {"w": torch.ones(1)}), metrics={"loss": 2.0}))
    aggr.accept_model(model(refs(tmp_path, "two", {"w": torch.ones(1)}), client="site-2", metrics=metrics))
    assert aggr.aggregate_model().metrics == (None if metrics is None else {"loss": 3.0})


@pytest.mark.parametrize("failure", ["abort", "enospc", "corrupt", "diff"])
def test_failed_round_keeps_saved_output_and_cleans_staging(tmp_path, monkeypatch, failure):
    saved = tmp_path / disk.SAVED_MODEL
    save_file({"w": torch.tensor([9.0])}, saved)
    before = saved.read_bytes()
    aggr = aggregator(tmp_path)
    params = refs(tmp_path, "one", {"w": torch.ones(1)})
    aggr.accept_model(model(params, kind=ParamsType.DIFF if failure == "diff" else ParamsType.FULL))
    if failure == "abort":
        aggr.fl_ctx.get_run_abort_signal().trigger(True)
    elif failure == "corrupt":
        (tmp_path / "one" / "model.safetensors").write_bytes(b"broken")
    elif failure == "enospc":

        def write_partial(path, metadata, tensors):
            with open(path, "wb") as f:
                f.write(b"partial")
            raise OSError(errno.ENOSPC, "no disk space")

        monkeypatch.setattr(disk, "write_safetensors", write_partial)
    with pytest.raises(Exception):
        aggr.aggregate_model()
    assert saved.read_bytes() == before
    assert not (tmp_path / (disk.CURRENT_MODEL + ".next")).exists()
    assert not (tmp_path / "one").exists()


def test_aggregation_releases_previous_output_before_next_key(tmp_path, monkeypatch):
    aggr = aggregator(tmp_path)
    aggr.accept_model(model(refs(tmp_path, "one", {key: torch.ones(4) for key in ("a", "b", "c")})))
    original = aggr._aggregate_key
    previous = []

    def tracked(key, base):
        assert all(ref() is None for ref in previous)
        tensor = original(key, base)
        previous.append(weakref.ref(tensor))
        return tensor

    monkeypatch.setattr(aggr, "_aggregate_key", tracked)
    aggr.aggregate_model()
    assert all(ref() is None for ref in previous)


def test_sharded_checkpoint_saved_reload_and_stale_slots(tmp_path):
    initial = tmp_path / "initial"
    initial.mkdir()
    save_file({"a": torch.ones(1), "unlisted": torch.zeros(1)}, initial / "one.safetensors")
    save_file({"b": torch.ones(2)}, initial / "two.safetensors")
    (initial / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "one.safetensors", "b": "two.safetensors"}})
    )
    ctx = context(tmp_path)
    persistor = disk.DiskFedAvgPersistor(str(initial))
    loaded = persistor.load_model(ctx)["weights"]
    assert set(loaded) == {"a", "b"}
    aggr = disk.DiskFedAvgAggregator()
    aggr.fl_ctx = ctx
    aggr.accept_model(model(refs(tmp_path, "client", {"a": torch.full((1,), 3.0), "b": torch.full((2,), 4.0)})))
    result = aggr.aggregate_model()
    persistor.save_model(make_model_learnable(result.params, {}), ctx)
    assert not (tmp_path / disk.CURRENT_MODEL).exists()
    assert result.params["b"].materialize().tolist() == [4.0, 4.0]
    assert all(ref.file_path == str(tmp_path / disk.SAVED_MODEL) for ref in result.params.values())
    (tmp_path / disk.CURRENT_MODEL).write_bytes(b"stale")
    (tmp_path / (disk.CURRENT_MODEL + ".next")).write_bytes(b"partial")
    restarted = disk.DiskFedAvgPersistor(str(initial)).load_model(ctx)
    assert restarted["weights"]["a"].materialize().item() == 3.0


@pytest.mark.parametrize("replace_before_open", [True, False])
def test_retired_task_download_cannot_mix_checkpoint_slots(tmp_path, monkeypatch, replace_before_open):
    from safetensors.torch import load

    aggr = aggregator(tmp_path)
    ctx = aggr.fl_ctx
    persistor = disk.DiskFedAvgPersistor("unused")
    aggr.accept_model(model(refs(tmp_path, "first", {key: torch.ones(1) for key in ("a", "b")})))
    persistor.save_model(make_model_learnable(aggr.aggregate_model().params, {}), ctx)
    saved = str(tmp_path / disk.SAVED_MODEL)
    reader = TensorDownloadable(safetensors_refs(saved), max_chunk_size=1)
    _, items, state = reader.produce({}, "slow-client")
    assert load(items[0])["a"].item() == 1.0

    # Task retirement deletes its message root, but tensor downloads have their own lifetime.
    wf = WFCommServer()
    wf._engine = ctx.get_engine()
    wf._engine.new_context = lambda: ctx
    task = Task("train", Shareable())
    task.completion_status = TaskCompletionStatus.CLIENT_DEAD
    task.msg_root_id = "retired-disk-fedavg-test"
    wf._tasks.append(task)
    wf.check_tasks()
    assert wf.get_num_standing_tasks() == 0

    aggr.accept_model(model(refs(tmp_path, "next", {key: torch.full((1,), 2.0) for key in ("a", "b")})))
    replacement = make_model_learnable(aggr.aggregate_model().params, {})
    if replace_before_open:
        persistor.save_model(replacement, ctx)
        with pytest.raises(RuntimeError, match="model was replaced"):
            reader.produce(state, "slow-client")
    else:
        real_open = lazy.safe_open

        @contextmanager
        def open_then_replace(*args, **kwargs):
            with real_open(*args, **kwargs) as opened:
                persistor.save_model(replacement, ctx)
                yield opened

        monkeypatch.setattr(lazy, "safe_open", open_then_replace)
        _, items, _ = reader.produce(state, "slow-client")
        assert load(items[0])["b"].item() == 1.0  # The opened file, not its replaced path, owns the read.
    reader.release()


@pytest.mark.parametrize("stop_cond,failed_client", [(None, False), ("loss <= 0", False), (None, True)])
def test_existing_controller_runs_multiple_diff_rounds_and_stopping(tmp_path, stop_cond, failed_client):
    save_file({"w": torch.zeros(1)}, tmp_path / "initial.safetensors")
    ctx = context(tmp_path)
    aggr = disk.DiskFedAvgAggregator()
    aggr.fl_ctx = ctx
    persistor = disk.DiskFedAvgPersistor(str(tmp_path / "initial.safetensors"))
    ctl = FedAvg(
        num_clients=2,
        num_rounds=3,
        aggregator=aggr,
        stop_cond=stop_cond,
        patience=1 if stop_cond else None,
        enable_tensor_disk_offload=True,
    )
    ctl.persistor = persistor
    ctl.engine = ctx.get_engine()
    ctl.fl_ctx = ctx
    ctl.abort_signal = ctx.get_run_abort_signal()
    events = []
    ctl.event = lambda event: events.append(event)
    ctl.fire_event_with_data = lambda event, *args: events.append(event)
    ctl.info = lambda *_: None
    ctl.sample_clients = lambda _: ["site-1", "site-2", "site-3"] if failed_client else ["site-1", "site-2"]
    ctl.get_num_standing_tasks = lambda: 0
    ctl._accept_train_result = lambda **_: True

    def send(task_name, targets, data, callback):
        snapshot = copy.deepcopy(data.params)
        assert snapshot["w"].materialize().item() == ctl.current_round
        task = Task(task_name, FLModelUtils.to_shareable(data))
        task.set_prop(AppConstants.TASK_PROP_CALLBACK, callback)
        task.set_prop(AppConstants.META_DATA, {})
        for client in targets:
            size = 2 if failed_client and client == "site-2" else 1
            params = refs(tmp_path, f"{ctl.current_round}-{client}", {"w": torch.ones(size)})
            ct = ClientTask(Client(client, client), task)
            ct.result = FLModelUtils.to_shareable(
                model(params, client=client, metrics={"loss": 1.0 + ctl.current_round}, kind=ParamsType.DIFF)
            )
            ctl._process_result(ct, ctx)

    ctl.send_model = send
    if failed_client:
        with pytest.raises(RuntimeError, match="refusing to publish"):
            ctl.run()
        assert ctl._received_count == 1
        assert not (tmp_path / disk.SAVED_MODEL).exists()
        assert AppEventType.AFTER_AGGREGATION not in events
        assert AppEventType.AFTER_LEARNABLE_PERSIST not in events
        return
    ctl.run()
    saved = load_file(tmp_path / disk.SAVED_MODEL)["w"].item()
    assert saved == (1.0 if stop_cond else 3.0)
    if stop_cond:
        assert load_file(tmp_path / disk.CURRENT_MODEL)["w"].item() == 2.0
    assert AppEventType.AFTER_AGGREGATION in events
    assert AppEventType.AFTER_LEARNABLE_PERSIST in events


@pytest.mark.parametrize("recipe_cls", [FedAvgRecipe, FedProxRecipe])
def test_recipe_composes_disk_components_and_preserves_memory_default(tmp_path, recipe_cls):
    initial = str(tmp_path / "initial.safetensors")
    save_file({"w": torch.ones(1)}, initial)
    recipe = recipe_cls(min_clients=2, train_script="train.py", initial_ckpt=initial, model_storage="disk")
    assert isinstance(recipe.aggregator, disk.DiskFedAvgAggregator)
    assert isinstance(recipe.model_persistor, disk.DiskFedAvgPersistor)
    assert recipe.server_expected_format == ExchangeFormat.PYTORCH
    assert recipe.enable_tensor_disk_offload
    assert recipe._create_client_runner({}) is not None
    with pytest.raises(ValueError, match="PyTorch exchange"):
        recipe._create_client_runner({"server_expected_format": ExchangeFormat.NUMPY})
    normal = recipe_cls(min_clients=2, train_script="train.py", model=torch.nn.Linear(1, 1))
    assert normal.model_storage == "memory"
    assert normal.aggregator is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": torch.nn.Linear(1, 1)},
        {"aggregator": object()},
        {"model_persistor": object()},
        {"model_locator": object()},
        {"best_model_filename": "custom.pt"},
        {"initial_ckpt": None},
    ],
)
def test_disk_recipe_rejects_incompatible_components(tmp_path, kwargs):
    options = {"initial_ckpt": str(tmp_path / "model.safetensors"), **kwargs}
    with pytest.raises(ValueError):
        FedAvgRecipe(min_clients=1, train_script="train.py", model_storage="disk", **options)


def test_disk_recipe_exports_reconstructable_diff_and_exclusions(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("pass\n")
    save_file({"w": torch.ones(1)}, tmp_path / "model.safetensors")
    recipe = FedAvgRecipe(
        min_clients=1,
        train_script=str(script),
        initial_ckpt=str(tmp_path / "model.safetensors"),
        model_storage="disk",
        params_transfer_type=TransferType.DIFF,
        exclude_vars="skip",
        stop_cond="loss < 0.1",
    )
    assert recipe.params_transfer_type == TransferType.DIFF
    recipe.export(str(tmp_path / "job"))
    config = json.loads(next((tmp_path / "job").rglob("config_fed_server.json")).read_text())
    exported = config["workflows"][0]["args"]["aggregator"]
    assert exported["args"]["exclude_vars"] == "skip"
    rebuilt = instantiate_class(exported["path"], exported["args"])
    assert isinstance(rebuilt, disk.DiskFedAvgAggregator)
    assert rebuilt.exclude_vars == "skip"
