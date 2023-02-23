# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

"""Codes for homomorphic encryption."""

from nvflare.app_opt.he.homomorphic_encrypt import count_encrypted_layers, load_tenseal_context_from_workspace
from nvflare.app_opt.he.intime_accumulate_model_aggregator import HEInTimeAccumulateWeightedAggregator
from nvflare.app_opt.he.model_decryptor import HEModelDecryptor
from nvflare.app_opt.he.model_encryptor import HEModelEncryptor
from nvflare.app_opt.he.model_shareable_generator import HEModelShareableGenerator
