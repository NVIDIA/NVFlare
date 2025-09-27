# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import json
import os

import torch
import torch.optim as optim

# Import original FedHCA2 components
from fedhca2_core.aggregate import aggregate, update_hyperweight
from fedhca2_core.models.hyperweight import HyperAggWeight, HyperCrossAttention

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.app_constant import AppConstants


class FedHCA2Aggregator(Aggregator):
    """NVFLARE aggregator for FedHCA2 federated multi-task learning"""

    def __init__(
        self,
        train_config_filename="config_train.json",
    ):
        super().__init__()

        self.train_config_filename = train_config_filename
        self.config_info = None

        # FedHCA2 algorithm parameters
        self.encoder_agg = None
        self.decoder_agg = None
        self.ca_c = None
        self.enc_alpha_init = None
        self.dec_beta_init = None
        self.hyperweight_lr = None
        self.backbone_type = None
        self.backbone_pretrained = None

        # State management
        self.submissions = {}
        self.all_clients_info = []
        self.last_ckpt = {}
        self.hyperweight = {}
        self.current_round = 0
        self.initialized = False

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        """Accept model submission from client"""
        try:
            from nvflare.app_common.utils.fl_model_utils import FLModelUtils

            fl_model = FLModelUtils.from_shareable(shareable, fl_ctx)

            client_name = fl_model.meta.get("client_name", f"client_{len(self.submissions)}")
            self.submissions[client_name] = fl_model
            return True
        except Exception as e:
            print(f"Error accepting model: {e}")
            return False

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        """Aggregate models using original FedHCA2 algorithm"""
        try:
            self.current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND, 0)

            print(f"Round {self.current_round}: Aggregating {len(self.submissions)} models using FedHCA2")

            # Initialize on first round
            if not self.initialized:
                if self.config_info is None:
                    self._load_train_config(fl_ctx)
                self._initialize_fedhca2()
                self.initialized = True

            # Convert submissions to format expected by original FedHCA2
            all_clients, save_ckpt = self._prepare_client_data()

            # Initialize last_ckpt on first round
            if self.current_round == 0:
                self.last_ckpt = copy.deepcopy(save_ckpt)

            # Apply FedHCA2 aggregation
            aggregate(
                all_clients=all_clients,
                save_ckpt=save_ckpt,
                last_ckpt=self.last_ckpt,
                hyperweight=self.hyperweight,
                encoder_agg=self.encoder_agg,
                decoder_agg=self.decoder_agg,
                ca_c=self.ca_c,
            )

            # Update hyperweights after aggregation
            if self.current_round > 0:
                update_hyperweight(all_clients, self.hyperweight, save_ckpt, self.last_ckpt)

            # Extract updated parameters from aggregated models
            updated_ckpt_list = []
            updated_ckpt_by_name = {}
            for i, client_info in enumerate(self.all_clients_info):
                client_name = client_info["name"]
                state_dict = copy.deepcopy(all_clients[i]['model'].state_dict())
                updated_ckpt_list.append(state_dict)
                updated_ckpt_by_name[client_name] = state_dict

            # Update last_ckpt for next round
            self.last_ckpt = copy.deepcopy(updated_ckpt_list)

            # Return first client's model as global model for NVFLARE compatibility
            first_client_params = {k: v.cpu().numpy() for k, v in updated_ckpt_list[0].items()}

            # Include all personalized models in metadata
            personalized_models = {
                name: {k: v.cpu().numpy() for k, v in params.items()} for name, params in updated_ckpt_by_name.items()
            }

            result_model = FLModel(params=first_client_params, meta={"personalized_models": personalized_models})

            # Clear submissions for next round
            self.submissions.clear()

            # Convert result to shareable
            from nvflare.app_common.utils.fl_model_utils import FLModelUtils

            return FLModelUtils.to_shareable(result_model)

        except Exception as e:
            print(f"Aggregation error: {e}")
            import traceback

            traceback.print_exc()

            # Fallback: return simple average
            save_ckpt = {
                name: {k: torch.tensor(v) for k, v in model.params.items()} for name, model in self.submissions.items()
            }
            avg_params = self._simple_average(save_ckpt)
            result_model = FLModel(params=avg_params)

            from nvflare.app_common.utils.fl_model_utils import FLModelUtils

            return FLModelUtils.to_shareable(result_model)

    def _initialize_fedhca2(self):
        """Initialize FedHCA2 components"""
        # Build client info from submissions
        self.all_clients_info = []
        for name, model in self.submissions.items():
            client_info = {
                "name": name,
                "tasks": model.meta.get("tasks", ["semseg"]),
                "dataname": model.meta.get("dataname", "pascalcontext"),
                "model": self._create_model_from_params(model.params),
            }
            self.all_clients_info.append(client_info)

        # Sort clients: ST first, then MT
        self.all_clients_info.sort(key=lambda x: len(x["tasks"]))

        # Initialize hyperweights
        N = len(self.all_clients_info)
        n_decoders = sum([len(client["tasks"]) for client in self.all_clients_info])

        if self.encoder_agg == "conflict_averse":
            hypernet = HyperAggWeight(K=N, init_alpha=self.enc_alpha_init)
            self.hyperweight['enc'] = hypernet
            self.hyperweight['enc_optimizer'] = optim.SGD(hypernet.parameters(), lr=self.hyperweight_lr)

        if self.decoder_agg == "cross_attention":
            first_client = self.all_clients_info[0]
            sample_decoder = self._create_decoder_model(first_client["tasks"], first_client["dataname"])
            hypernet = HyperCrossAttention(model=sample_decoder, K=n_decoders, init_beta=self.dec_beta_init)
            self.hyperweight['dec'] = hypernet
            self.hyperweight['dec_optimizer'] = optim.SGD(hypernet.parameters(), lr=self.hyperweight_lr)

    def _prepare_client_data(self):
        """Convert NVFLARE data to format expected by original FedHCA2"""
        all_clients = []
        save_ckpt = {}

        # Build client list in the order of all_clients_info
        for client_info in self.all_clients_info:
            client_name = client_info["name"]
            model_submission = self.submissions[client_name]

            # Convert parameters to tensors
            state_dict = {k: torch.tensor(v) for k, v in model_submission.params.items()}
            save_ckpt[client_name] = state_dict

            # Create real model for original FedHCA2 interface
            real_model = self._create_model_with_state(client_info["tasks"], client_info["dataname"], state_dict)

            client = {'tasks': client_info["tasks"], 'dataname': client_info["dataname"], 'model': real_model}
            all_clients.append(client)

        # Convert save_ckpt to list format expected by original code
        save_ckpt_list = [save_ckpt[client_info["name"]] for client_info in self.all_clients_info]

        return all_clients, save_ckpt_list

    def _create_model_from_params(self, params):
        """Create a real model with the given parameters"""
        # Use the original FedHCA2 model building logic
        from fedhca2_core.models.build_models import build_model

        # Infer task information from parameter keys
        tasks = self._infer_tasks_from_params(params)
        dataname = self._infer_dataname_from_params(params)

        # Build the actual model
        model = build_model(
            tasks=tasks,
            dataname=dataname,
            backbone_type=self.backbone_type,
            backbone_pretrained=False,  # Don't load pretrained weights here
        )

        return model

    def _create_model_with_state(self, tasks, dataname, state_dict):
        """Create real model and load state dict"""
        from fedhca2_core.models.build_models import build_model

        model = build_model(tasks=tasks, dataname=dataname, backbone_type=self.backbone_type, backbone_pretrained=False)

        # Load the state dict
        model.load_state_dict(state_dict, strict=False)
        return model

    def _create_decoder_model(self, tasks, dataname):
        """Create a real decoder model for hyperweight initialization"""
        from fedhca2_core.models.build_models import build_model

        # Build a minimal model to get decoder structure
        model = build_model(tasks=tasks, dataname=dataname, backbone_type=self.backbone_type, backbone_pretrained=False)

        # Return the decoder part
        return model.decoders[tasks[0]]  # Use first task's decoder as template

    def _infer_tasks_from_params(self, params):
        """Infer tasks from parameter names"""
        tasks = set()
        for key in params.keys():
            if 'decoders.' in key:
                task = key.split('decoders.')[1].split('.')[0]
                tasks.add(task)

        # Default fallback
        if not tasks:
            return ['semseg']
        return list(tasks)

    def _infer_dataname_from_params(self, params):
        """Infer dataset name from parameter structure"""
        # Simple heuristic: if we see depth-related parameters, it's likely NYUD
        for key in params.keys():
            if 'depth' in key:
                return 'nyud'
        return 'pascalcontext'

    def _simple_average(self, models_list):
        """Simple averaging of model parameters"""
        if not models_list:
            return {}

        # models_list is already a list from aggregate function
        model_list = models_list
        if not model_list:
            return {}

        # Get first model as template
        first_model = model_list[0]
        avg_params = {}

        for name in first_model.keys():
            param_tensors = []
            for model in model_list:
                param = model[name]
                if isinstance(param, torch.Tensor):
                    param_tensors.append(param)

            if param_tensors:
                avg_params[name] = torch.mean(torch.stack(param_tensors), dim=0)
            else:
                avg_params[name] = first_model[name]

        # Convert back to numpy for NVFLARE
        result = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in avg_params.items()}

        return result

    def _load_train_config(self, fl_ctx: FLContext):
        """Load training configuration using standard NVFLARE pattern"""
        # Load training configurations json
        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        app_config_dir = ws.get_app_config_dir(fl_ctx.get_job_id())
        train_config_file_path = os.path.join(app_config_dir, self.train_config_filename)

        if not os.path.isfile(train_config_file_path):
            raise FileNotFoundError(f"Training configuration file does not exist at {train_config_file_path}")

        with open(train_config_file_path) as file:
            self.config_info = json.load(file)

        # Load FedHCA2 algorithm parameters
        algorithm_config = self.config_info["algorithm"]
        self.encoder_agg = algorithm_config["encoder_agg"]
        self.decoder_agg = algorithm_config["decoder_agg"]
        self.ca_c = algorithm_config["ca_c"]
        self.enc_alpha_init = algorithm_config["enc_alpha_init"]
        self.dec_beta_init = algorithm_config["dec_beta_init"]

        # Load hyperweight configuration
        hyperweight_config = self.config_info["hyperweight"]
        self.hyperweight_lr = hyperweight_config["learning_rate"]

        # Load model configuration
        model_config = self.config_info["model"]
        self.backbone_type = model_config["backbone_type"]
        self.backbone_pretrained = model_config["backbone_pretrained"]
