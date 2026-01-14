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

import tensorflow as tf

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_opt.tf.utils import unflat_layer_weights_dict
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.security.logging import secure_format_exception


class TFValidator(Executor):
    def __init__(
        self,
        model: tf.keras.Model,
        data_loader=None,
        metric_fn=None,
    ):
        """Component-based TensorFlow Validator for cross-site evaluation.

        This validator provides an alternative to the Client API pattern (flare.is_evaluate()) for
        TensorFlow cross-site evaluation. Use this when you prefer a component-based approach or when
        your validation logic is separate from your training script.

        **When to use TFValidator vs Client API pattern:**

        Use TFValidator (component-based) when:
        - You prefer explicit validation components separate from training logic
        - Your validation code is reusable across multiple projects
        - You want validation to work without modifying your training script
        - Your recipe is CSE-only (no training, only validation)

        Use Client API pattern (flare.is_evaluate()) when:
        - Your validation logic is tightly coupled with training (e.g., same preprocessing)
        - You want a single script that handles both training and validation
        - You're already using Client API for training (recommended for most cases)

        **Default recommendation**: Use Client API pattern (like PyTorch) for consistency and simplicity.
        TFValidator is provided for flexibility and backward compatibility with NumPy-style workflows.

        Example (Component-based with TFValidator):
            ```python
            from nvflare.app_opt.tf.recipes import FedAvgRecipe
            from nvflare.app_opt.tf.tf_validator import TFValidator
            from nvflare.recipe.utils import add_cross_site_evaluation

            # Create recipe
            recipe = FedAvgRecipe(
                name="my-job",
                min_clients=2,
                num_rounds=3,
                initial_model=my_model,
                train_script="client.py"
            )

            # Add CSE
            add_cross_site_evaluation(recipe)

            # Manually add TFValidator if you prefer component-based validation
            validator = TFValidator(
                model=my_model,
                data_loader=test_loader,
                metric_fn=lambda model, loader: {"accuracy": model.evaluate(loader)[1]}
            )
            recipe.job.to_clients(validator, tasks=["validate"])
            ```

        Example (Client API pattern - recommended):
            ```python
            # In client.py:
            while flare.is_running():
                input_model = flare.receive()
                # Load model weights...

                metrics = evaluate(model, test_loader)

                if flare.is_evaluate():  # CSE validation task
                    flare.send(flare.FLModel(metrics=metrics))
                    continue

                # Normal training...
            ```

        Args:
            model: TensorFlow Keras model to validate
            data_loader: Optional data loader for validation. If None, user must provide validation logic.
            metric_fn: Optional metric function that takes (model, data_loader) and returns dict of metrics.
                      If None, uses default accuracy evaluation.
        """
        super().__init__()

        self.logger = get_obj_logger(self)
        self.model = model
        self.data_loader = data_loader
        self.metric_fn = metric_fn
        self._validate_task_name = AppConstants.TASK_VALIDATION

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        if task_name == self._validate_task_name:
            try:
                # Extract DXO from shareable
                try:
                    model_dxo = from_shareable(shareable)
                except Exception as e:
                    self.log_error(
                        fl_ctx, f"Unable to extract model dxo from shareable. Exception: {secure_format_exception(e)}"
                    )
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Get model from shareable. data_kind must be WEIGHTS.
                if not model_dxo.data or model_dxo.data_kind != DataKind.WEIGHTS:
                    self.log_error(
                        fl_ctx, "Model DXO doesn't have data or is not of type DataKind.WEIGHTS. Unable to validate."
                    )
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Get model owner info
                model_name = shareable.get_header(AppConstants.MODEL_OWNER, "?")
                self.log_info(fl_ctx, f"Validating model from {model_name} on {fl_ctx.get_identity_name()}")

                # Check abort signal
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                # Load weights into model
                try:
                    weights_dict = unflat_layer_weights_dict(model_dxo.data)
                    for layer_name, weights in weights_dict.items():
                        layer = self.model.get_layer(name=layer_name)
                        layer.set_weights(weights)
                except Exception as e:
                    self.log_error(fl_ctx, f"Error loading weights: {secure_format_exception(e)}")
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)

                # Check abort signal
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                # Perform validation
                try:
                    if self.metric_fn:
                        # Use custom metric function
                        val_results = self.metric_fn(self.model, self.data_loader)
                    elif self.data_loader:
                        # Use default evaluation
                        loss, accuracy = self.model.evaluate(self.data_loader, verbose=0)
                        val_results = {"loss": float(loss), "accuracy": float(accuracy)}
                    else:
                        self.log_error(
                            fl_ctx,
                            "No data_loader or metric_fn provided. Cannot perform validation. "
                            "Please provide either data_loader or metric_fn when creating TFValidator.",
                        )
                        return make_reply(ReturnCode.EXECUTION_EXCEPTION)
                except Exception as e:
                    self.log_error(fl_ctx, f"Error during validation: {secure_format_exception(e)}")
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)

                # Check abort signal
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                self.log_info(fl_ctx, f"Validation result: {val_results}")

                # Create DXO for metrics and return shareable
                metric_dxo = DXO(data_kind=DataKind.METRICS, data=val_results)
                return metric_dxo.to_shareable()
            except Exception as e:
                self.log_exception(fl_ctx, f"Exception in TFValidator execute: {secure_format_exception(e)}.")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)
