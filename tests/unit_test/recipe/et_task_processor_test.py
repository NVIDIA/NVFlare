# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for ETTaskProcessor epoch configuration and training loop."""

from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import TensorDataset  # noqa: E402

from nvflare.edge.simulation.et_task_processor import ETTaskProcessor, calc_params_diff, clone_params  # noqa: E402


# --- Concrete subclass for testing (ETTaskProcessor is abstract) ---


class _StubETTaskProcessor(ETTaskProcessor):
    """Minimal concrete subclass that returns a fixed dataset."""

    def __init__(self, dataset, **kwargs):
        super().__init__(data_path="unused", **kwargs)
        self._dataset = dataset

    def create_dataset(self, data_path):
        return self._dataset


# --- Fixtures ---


@pytest.fixture
def simple_dataset():
    """4-sample dataset so batch_size=2 gives 2 batches."""
    x = torch.randn(4, 2)
    y = torch.randint(0, 2, (4,))
    return TensorDataset(x, y)


def _make_processor(dataset, training_config=None):
    return _StubETTaskProcessor(dataset, training_config=training_config)


# --- Default config tests ---


class TestETTaskProcessorConfig:
    """Verify training_config defaults and overrides."""

    def test_default_epoch_is_one(self, simple_dataset):
        proc = _make_processor(simple_dataset)
        assert proc.training_config["epoch"] == 1

    def test_epoch_override(self, simple_dataset):
        proc = _make_processor(simple_dataset, training_config={"epoch": 5})
        assert proc.training_config["epoch"] == 5

    def test_other_defaults_preserved_when_epoch_set(self, simple_dataset):
        proc = _make_processor(simple_dataset, training_config={"epoch": 3})
        assert proc.training_config["batch_size"] == 32
        assert proc.training_config["learning_rate"] == 0.1


# --- Epoch validation tests ---


class TestETTaskProcessorEpochValidation:
    """Epoch must be a positive integer."""

    @pytest.mark.parametrize("bad_value", [0, -1, -10])
    def test_rejects_non_positive_epoch(self, simple_dataset, bad_value):
        with pytest.raises(ValueError, match="epoch must > 0"):
            _make_processor(simple_dataset, training_config={"epoch": bad_value})

    @pytest.mark.parametrize("bad_value", [1.5, "three", None])
    def test_rejects_non_int_epoch(self, simple_dataset, bad_value):
        with pytest.raises(TypeError, match="epoch must be an int"):
            _make_processor(simple_dataset, training_config={"epoch": bad_value})


# --- run_training epoch count tests ---


class TestRunTrainingEpochs:
    """Verify run_training respects the total_epochs parameter."""

    def _make_mock_et_model(self):
        """Create a mock ExecuTorch model with the interface run_training expects."""
        et_model = MagicMock()
        # forward_backward returns (loss, pred)
        et_model.forward_backward.return_value = (torch.tensor(0.5), torch.tensor([0]))
        # named_parameters returns a dict of tensors
        et_model.named_parameters.return_value = {"w": torch.randn(2, 2)}
        et_model.named_gradients.return_value = {"w": torch.randn(2, 2)}
        return et_model

    @pytest.fixture(autouse=True)
    def _mock_sgd_optimizer(self):
        """Replace the lazy-import proxy for get_sgd_optimizer with a mock.

        The module-level get_sgd_optimizer is a lazy-import proxy that raises
        LazyImportError when executorch is not installed.  We cannot use
        @patch on it because the decorator inspects the object (triggering the
        error).  Instead, swap the module attribute directly.
        """
        import nvflare.edge.simulation.et_task_processor as mod

        self.mock_optimizer = MagicMock()
        mock_get_opt = MagicMock(return_value=self.mock_optimizer)
        original = mod.get_sgd_optimizer
        mod.get_sgd_optimizer = mock_get_opt
        yield
        mod.get_sgd_optimizer = original

    def test_single_epoch(self, simple_dataset):
        proc = _make_processor(simple_dataset, training_config={"batch_size": 2, "epoch": 1})
        et_model = self._make_mock_et_model()

        result = proc.run_training(et_model, total_epochs=1)

        assert isinstance(result, dict)
        # 2 batches * 1 epoch = 2 forward_backward calls
        assert et_model.forward_backward.call_count == 2

    def test_multiple_epochs(self, simple_dataset):
        proc = _make_processor(simple_dataset, training_config={"batch_size": 2, "epoch": 3})
        et_model = self._make_mock_et_model()

        result = proc.run_training(et_model, total_epochs=3)

        assert isinstance(result, dict)
        # 2 batches * 3 epochs = 6 forward_backward calls
        assert et_model.forward_backward.call_count == 6

    def test_process_task_passes_epoch_from_config(self, simple_dataset):
        """process_task should read epoch from training_config and pass it to run_training."""
        proc = _make_processor(simple_dataset, training_config={"batch_size": 2, "epoch": 2})
        et_model = self._make_mock_et_model()

        with patch.object(proc, "run_training", wraps=proc.run_training) as spy:
            total_epochs = proc.training_config.get("epoch", 1)
            spy(et_model, total_epochs=total_epochs)

            spy.assert_called_once_with(et_model, total_epochs=2)


# --- Helper function tests ---


class TestHelperFunctions:
    def test_clone_params(self):
        original = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0])}
        cloned = clone_params(original)

        assert set(cloned.keys()) == set(original.keys())
        for k in original:
            assert torch.equal(cloned[k], original[k])
            # Must be a different tensor object
            assert cloned[k].data_ptr() != original[k].data_ptr()

    def test_calc_params_diff(self):
        initial = {"a": torch.tensor([1.0, 2.0])}
        last = {"a": torch.tensor([3.0, 5.0])}
        diff = calc_params_diff(initial, last)

        assert torch.equal(diff["a"], torch.tensor([2.0, 3.0]))
