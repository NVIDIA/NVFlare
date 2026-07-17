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

import pytest

from nvflare.fuel.utils import fobs
from tests.integration_test.src.validators.tf_model_validator import TFModelValidator


def _model_path(tmp_path, model_file_name):
    model_path = tmp_path / "workspace" / "app_server" / model_file_name
    model_path.parent.mkdir(parents=True)
    return model_path


def test_tf_model_validator_accepts_legacy_fobs_model(tmp_path):
    model_path = _model_path(tmp_path, "tf2weights.fobs")
    with model_path.open("wb") as model_file:
        fobs.dump({"weights": {}, "meta": {}}, model_file)

    validator = TFModelValidator()

    assert validator.validate_finished_results({"workspace_root": str(tmp_path / "workspace")}, [])


def test_tf_model_validator_accepts_recipe_hdf5_weights(tmp_path):
    model_path = _model_path(tmp_path, "tf_model.weights.h5")
    model_path.write_bytes(b"\x89HDF\r\n\x1a\nweights")
    validator = TFModelValidator(model_file_name="tf_model.weights.h5")

    assert validator.validate_finished_results({"workspace_root": str(tmp_path / "workspace")}, [])


def test_tf_model_validator_rejects_invalid_hdf5_weights(tmp_path):
    model_path = _model_path(tmp_path, "tf_model.weights.h5")
    model_path.write_bytes(b"not HDF5")
    validator = TFModelValidator(model_file_name="tf_model.weights.h5")

    assert not validator.validate_finished_results({"workspace_root": str(tmp_path / "workspace")}, [])


def test_tf_model_validator_rejects_model_path(tmp_path):
    with pytest.raises(ValueError, match="without directory components"):
        TFModelValidator(model_file_name="../tf_model.weights.h5")
