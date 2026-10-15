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

from pathlib import Path

from nvflare.app_common.abstract.model import ModelLearnableKey
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor


class HuggingFacePersistor(PTFileModelPersistor):
    """Save each round locally and upload the final round's Qwen LoRA adapter from SJ.

    This example uses a fixed number of rounds starting at zero, without early stopping.
    Hugging Face credentials are supplied by the server job's runtime environment.
    """

    def __init__(self, model_name_or_path: str, repo_id: str):
        super().__init__(
            model={"path": "model.QwenLoRAModel", "args": {"model_name_or_path": model_name_or_path}},
            allow_numpy_conversion=False,
        )
        self.model_name_or_path = model_name_or_path
        self.repo_id = repo_id

    def save_model(self, ml, fl_ctx):
        super().save_model(ml, fl_ctx)
        if fl_ctx.get_prop(AppConstants.CURRENT_ROUND) != fl_ctx.get_prop(AppConstants.NUM_ROUNDS) - 1:
            return

        from huggingface_hub import HfApi
        from transformers import AutoTokenizer

        # The persistor's model still holds the initial weights. Restore the aggregate
        # into that existing model before exporting the adapter.
        weights = ml[ModelLearnableKey.WEIGHTS]
        if set(weights) != set(self.model.state_dict()):
            raise ValueError(
                "Checkpoint adapter keys do not match the model. Use the same model and LoRA configuration."
            )
        self.model.load_state_dict(weights)
        output_dir = Path(self.log_dir) / "huggingface-adapter"
        self.model.model.save_pretrained(output_dir)
        AutoTokenizer.from_pretrained(self.model_name_or_path).save_pretrained(output_dir)

        # HfApi reads HF_TOKEN or the server runtime's saved Hugging Face token.
        api = HfApi()
        api.create_repo(repo_id=self.repo_id, repo_type="model", private=True, exist_ok=True)
        commit = api.upload_folder(
            repo_id=self.repo_id,
            repo_type="model",
            folder_path=str(output_dir),
            commit_message="Upload final federated LoRA adapter",
        )
        self.log_info(fl_ctx, f"Hugging Face upload: {commit.commit_url}")
