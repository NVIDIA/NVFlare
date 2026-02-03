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

"""Test that all persistors fail fast when source_ckpt_file_full_name doesn't exist at runtime."""

import tempfile
import unittest
from unittest.mock import MagicMock, patch

from nvflare.apis.fl_context import FLContext


class TestCheckpointValidation(unittest.TestCase):
    """Test checkpoint validation across all persistors."""

    def setUp(self):
        """Set up test fixtures."""
        self.fl_ctx = MagicMock(spec=FLContext)
        self.nonexistent_path = "/tmp/nonexistent_checkpoint_12345.ckpt"

    def test_sklearn_persistor_fails_fast_on_missing_checkpoint(self):
        """Test JoblibModelParamPersistor fails when checkpoint doesn't exist."""
        from nvflare.app_opt.sklearn.joblib_model_param_persistor import JoblibModelParamPersistor

        persistor = JoblibModelParamPersistor(source_ckpt_file_full_name=self.nonexistent_path)

        # Mock _initialize to set save_path
        with tempfile.TemporaryDirectory() as tmpdir:
            persistor.log_dir = tmpdir
            persistor.save_path = f"{tmpdir}/model.joblib"

            with self.assertRaises(ValueError) as context:
                persistor.load_model(self.fl_ctx)

            self.assertIn("Source checkpoint not found", str(context.exception))
            self.assertIn(self.nonexistent_path, str(context.exception))
            self.assertIn("Check that the checkpoint exists at runtime", str(context.exception))

    def test_numpy_utils_fails_fast_on_missing_checkpoint(self):
        """Test load_numpy_model utility fails when checkpoint doesn't exist."""
        import numpy as np

        from nvflare.app_common.np.utils import load_numpy_model

        mock_logger = MagicMock()

        with self.assertRaises(ValueError) as context:
            load_numpy_model(
                fl_ctx=self.fl_ctx,
                logger=mock_logger,
                source_ckpt_file_full_name=self.nonexistent_path,
                model_file_path="/tmp/dummy.npy",
                get_fallback_data=lambda: np.array([1, 2, 3]),
            )

        self.assertIn("Source checkpoint not found", str(context.exception))
        self.assertIn(self.nonexistent_path, str(context.exception))
        self.assertIn("Check that the checkpoint exists at runtime", str(context.exception))

    def test_tf_persistor_fails_fast_on_missing_checkpoint(self):
        """Test TFModelPersistor fails when checkpoint doesn't exist."""
        try:
            from nvflare.app_opt.tf.model_persistor import TFModelPersistor
        except ImportError:
            self.skipTest("TensorFlow not installed")
            return

        persistor = TFModelPersistor(source_ckpt_file_full_name=self.nonexistent_path)

        # Mock _initialize
        with tempfile.TemporaryDirectory() as tmpdir:
            persistor._model_save_path = f"{tmpdir}/model.h5"
            persistor.log_dir = tmpdir

            # load_model should call system_panic, which we need to verify
            with patch.object(persistor, "system_panic") as mock_panic:
                persistor.load_model(self.fl_ctx)

                # Should have called system_panic with file not found error
                mock_panic.assert_called_once()
                call_args = mock_panic.call_args
                reason = call_args.kwargs.get("reason", "")
                self.assertIn("Source checkpoint not found", reason)
                self.assertIn(self.nonexistent_path, reason)

    def test_pt_persistor_fails_fast_on_missing_checkpoint(self):
        """Test PTFileModelPersistor fails when checkpoint doesn't exist."""
        try:
            from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
        except ImportError:
            self.skipTest("PyTorch not installed")
            return

        persistor = PTFileModelPersistor(source_ckpt_file_full_name=self.nonexistent_path)

        # Mock _initialize
        with tempfile.TemporaryDirectory() as tmpdir:
            persistor.log_dir = tmpdir
            persistor.ckpt_preload_path = None

            # load_model should call system_panic, which we need to verify
            with patch.object(persistor, "system_panic") as mock_panic:
                persistor.load_model(self.fl_ctx)

                # Should have called system_panic with file not found error
                mock_panic.assert_called_once()
                call_args = mock_panic.call_args
                reason = call_args.kwargs.get("reason", "")
                self.assertIn("Source checkpoint not found", reason)
                self.assertIn(self.nonexistent_path, reason)
                self.assertIn("Check that the checkpoint exists at runtime", reason)


if __name__ == "__main__":
    unittest.main()
