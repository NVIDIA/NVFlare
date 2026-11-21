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

"""Unit tests for TFModelPersistor source_ckpt_file_full_name support."""

import json

import numpy as np
import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnableKey
from nvflare.fuel.sec.audit import Auditor, AuditService


class TestTFModelPersistorInit:
    """Tests for TFModelPersistor initialization with source_ckpt_file_full_name."""

    def test_init_without_model_and_ckpt_allowed(self):
        """Init without model or source_ckpt should be allowed at init time."""
        try:
            from nvflare.app_opt.tf.model_persistor import TFModelPersistor

            # This should not raise - error only happens at load_model time
            persistor = TFModelPersistor(model=None)

            assert persistor.model is None
            assert persistor.source_ckpt_file_full_name is None
        except ImportError:
            pytest.skip("TensorFlow not installed")


class TestTFModelPersistorLoadModel:
    """Tests for TFModelPersistor initialization with source_ckpt_file_full_name."""

    def test_init_without_source_ckpt(self):
        """Init without source_ckpt should work."""
        try:
            import tensorflow as tf

            from nvflare.app_opt.tf.model_persistor import TFModelPersistor

            model = tf.keras.Sequential([tf.keras.layers.Dense(2)])
            persistor = TFModelPersistor(model=model)

            assert persistor.source_ckpt_file_full_name is None
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_init_with_source_ckpt_stores_path(self):
        """Init should store the source_ckpt path."""
        try:
            import tensorflow as tf

            from nvflare.app_opt.tf.model_persistor import TFModelPersistor

            model = tf.keras.Sequential([tf.keras.layers.Dense(2)])
            persistor = TFModelPersistor(
                model=model,
                source_ckpt_file_full_name="/data/pretrained/model.h5",
            )

            assert persistor.source_ckpt_file_full_name == "/data/pretrained/model.h5"
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_init_without_model_with_ckpt(self):
        """Init without model but with source_ckpt should work (TF can load full model)."""
        try:
            from nvflare.app_opt.tf.model_persistor import TFModelPersistor

            persistor = TFModelPersistor(
                model=None,
                source_ckpt_file_full_name="/data/pretrained/model.h5",
            )

            assert persistor.model is None
            assert persistor.source_ckpt_file_full_name == "/data/pretrained/model.h5"
        except ImportError:
            pytest.skip("TensorFlow not installed")


@pytest.mark.parametrize("scenario", ["read", "missing", "corrupt", "logging_error"])
def test_tf_named_retrieval_observes_only_the_weight_load(tmp_path, monkeypatch, scenario):
    tf = pytest.importorskip("tensorflow")
    from nvflare.app_opt.tf.model_persistor import TFModelPersistor

    model = tf.keras.Sequential([tf.keras.Input(shape=(1,)), tf.keras.layers.Dense(1, use_bias=False)])
    original_weights = [np.array([[621.0]], dtype=np.float32)]
    model.set_weights(original_weights)
    saved = tmp_path / "private-model.weights.h5"
    if scenario == "corrupt":
        saved.write_bytes(b"private-corrupt-checkpoint")
    else:
        model.save_weights(str(saved))

    model.set_weights([np.zeros((1, 1), dtype=np.float32)])
    persistor = TFModelPersistor(model=model, source_ckpt_file_full_name=str(saved))
    if scenario == "logging_error":

        def fail_log(_message):
            raise RuntimeError("private-log-error")

        monkeypatch.setattr(persistor.logger, "info", fail_log)

    filename = "private-missing.weights.h5" if scenario == "missing" else saved.name
    audit_path = tmp_path / "model-audit.jsonl"
    writer = Auditor(str(audit_path))
    monkeypatch.setattr(AuditService, "the_auditor", writer)
    try:
        result = persistor.get(filename, FLContext())
    finally:
        writer.close()

    if scenario == "read":
        weights = list(result[ModelLearnableKey.WEIGHTS].values())
        assert len(weights) == 1
        np.testing.assert_array_equal(weights[0], original_weights[0])
    else:
        assert result is None
    expected_reads = {
        "read": ["model.retrieve.stage_returned"],
        "missing": [],
        "corrupt": ["model.retrieve.stage_error"],
        "logging_error": [],
    }
    records = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert [
        record["action"] for record in records if "stage=file_read" in record.get("message", "").split()
    ] == expected_reads[scenario]
    assert records[-1]["action"] == "model.retrieve.returned"
    assert ("status=value_returned" if scenario == "read" else "status=no_value") in records[-1]["message"].split()
    serialized_records = json.dumps(records)
    assert "private-" not in serialized_records
    assert str(tmp_path) not in serialized_records
    assert all("621" not in record.get("message", "") for record in records)
