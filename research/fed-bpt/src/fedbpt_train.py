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

# Part of this code is adopted from BBT (https://github.com/txsun1997/Black-Box-Tuning)

# MIT License
#
# Copyright (c) 2022 Tianxiang Sun
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

from cma_decomposer import register_decomposers

import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter

# initializes NVFlare client API
flare.init()
# We serialize CMAEvolutionStrategy object directly. This requires registering custom decomposers.
register_decomposers()
# Use SummaryWriter to stream metrics to the server
writer = SummaryWriter()

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import copy
import random
import time
import warnings

import cma
import numpy as np
import torch

warnings.simplefilter("ignore", cma.evolution_strategy.InjectionWarning)

from cma.recombination_weights import RecombinationWeights
from data_process import construct_true_few_shot_data, data_processor, perturb_dataset, split_data
from LMForwardAPI import LMForwardAPI

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="roberta-large", choices=["roberta-base", "roberta-large"], type=str)
parser.add_argument("--task_name", default="sst2", type=str)
parser.add_argument("--n_prompt_tokens", default=50, type=int)
parser.add_argument("--intrinsic_dim", default=500, type=int)
parser.add_argument("--k_shot", default=16, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--bound", default=0, type=int)
parser.add_argument("--sigma", default=1, type=float)
parser.add_argument("--alpha", default=1, type=float)
parser.add_argument("--print_every", default=50, type=int)
parser.add_argument("--eval_every", default=100, type=int)
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--alg", default="CMA", type=str)
parser.add_argument("--random_proj", default="normal", type=str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--loss_type", default="ce", type=str)
parser.add_argument("--cat_or_add", default="add", type=str)
parser.add_argument("--parallel", action="store_true", help="Whether to allow parallel evaluation")
# fl args
parser.add_argument(
    "--eval_clients",
    default="site-1",
    type=str,
    help="Provide the name of client that should evaluate the global model. Can be comma-spearated list, e.g., `site-1,site-2`",
)
parser.add_argument("--num_users", default=10, type=int)
parser.add_argument("--iid", default=1, type=int)
parser.add_argument("--local_popsize", default=20, type=int)
parser.add_argument("--local_iter", default=8, type=int)
parser.add_argument("--alpha_dir", default=0.5, type=float)
parser.add_argument("--perturb_rate", default=0.5, type=float)
parser.add_argument("--perturb", default=0, type=int)
parser.add_argument("--note", default=None, type=str)
parser.add_argument("--llama_causal", default=0, type=int)
parser.add_argument("--norm_prompt", default=0, type=int)
parser.add_argument("--init_score_path", default=None, type=str)
parser.add_argument("--prompt_norm_threshold", default=15, type=float)
parser.add_argument("--prompt_norm_threshold_upper", default=20, type=float)
parser.add_argument("--save_prompt", default=0, type=int)
parser.add_argument(
    "--inference_framework",
    default="pt",
    type=str,
    help="""Which inference framework to use. 
         Currently supports `pt` and `ort`, standing for pytorch and Microsoft onnxruntime respectively""",
)
parser.add_argument("--onnx_model_path", default=None, type=str, help="Path to your onnx model.")
args = parser.parse_args()

model_name = args.model_name
task_name = args.task_name
n_prompt_tokens = args.n_prompt_tokens
intrinsic_dim = args.intrinsic_dim
k_shot = args.k_shot
batch_size = args.batch_size
bound = args.bound
sigma = args.sigma
alpha = args.alpha
eval_clients = args.eval_clients.split(",")

if args.local_popsize > 0:
    args.local_popsize = args.local_popsize
else:
    args.local_popsize = 4 + 3 * np.log(intrinsic_dim)

device = args.device
alg = args.alg
random_proj = args.random_proj
seed = args.seed
loss_type = args.loss_type
print_every = args.print_every
eval_every = args.eval_every
cat_or_add = args.cat_or_add
parallel = args.parallel
inference_framework = args.inference_framework
onnx_model_path = args.onnx_model_path

if inference_framework not in ["pt", "ort"]:
    raise ValueError(f'inference_framework only supports "pt", "ort", got `{inference_framework}` instead.')
if inference_framework == "ort":
    assert onnx_model_path is not None, "Path to onnx model is required, got None instead."
    assert os.path.exists(onnx_model_path), f"In valid onnx model path `{onnx_model_path}`"

# fixed hyper-params
if cat_or_add == "add":
    init_prompt_path = None
else:
    init_prompt_path = "./nli_base_prompt.pt"


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

# Initialize API

model_forward_api = LMForwardAPI(args=args, init_prompt_path=init_prompt_path)

global_api_setting = model_forward_api.client_record()

# Initialize data processor

data_processor = data_processor(args)


data_bundle = data_processor.get_data()
if task_name in ["agnews", "yelpp", "dbpedia", "snli"]:
    train_data, test_data = data_bundle.get_dataset("train"), data_bundle.get_dataset("test")
else:
    train_data, test_data = data_bundle.get_dataset("train"), data_bundle.get_dataset("validation")

train_data, dev_data = construct_true_few_shot_data(args, train_data, k_shot)

for ds in [train_data, dev_data, test_data]:
    ds.set_pad_val(
        "input_ids", data_processor.tokenizer.pad_token_id if data_processor.tokenizer.pad_token_id is not None else 0
    )
    ds.set_pad_val("attention_mask", 0)
print("# of train data: {}".format(len(train_data)))
print("Example:")
print(train_data[0])
print("\n# of dev data: {}".format(len(dev_data)))
print("Example:")
print(dev_data[0])
print("\n# of test data: {}".format(len(test_data)))
print("Example:")
print(test_data[0])

# Split dataset
user_dict_train, user_dict_dev = split_data(args, train_data, dev_data)

# use site name index to access data shards and track outputs
site_name = flare.get_site_name()
idx = int(site_name.split("-")[1]) - 1
print(f"idx from site name {site_name}: {idx}")

client_fitnesses_orig_dict = {idx: []}
client_fitnesses_pert_dict = {idx: []}
client_prompt_dict = {idx: []}

local_cma_mu = RecombinationWeights(args.local_popsize).mu

client_api_setting_list = {idx: model_forward_api.client_record()}

best_test_acc = 0
train_step = 0

# Run this loop every round
while flare.is_running():
    input_model = flare.receive()
    global_es = input_model.params["global_es"]
    current_round = input_model.current_round
    print(f"Running current_round={current_round}")
    print(
        f"Received global_es.sigma={global_es.sigma} and global_es.mean: len={len(global_es.mean)}, mean={np.mean(global_es.mean)}, std={np.std(global_es.mean)}"
    )

    local_es = global_es._copy_light(
        inopts={"seed": seed, "maxiter": args.local_iter, "popsize": args.local_popsize, "CMA_mu": None}
    )
    local_sigma_current = global_es.sigma

    if flare.get_site_name() in eval_clients:
        print("Global es evaluate on test data...")
        global_api_setting["best_prompt"] = local_es.mean
        model_forward_api.load_client_record(global_api_setting)
        global_test_acc = model_forward_api.eval(prompt_embedding=local_es.mean, test_data=test_data)
        print("Global test acc: {}".format(round(global_test_acc, 4)))
        print("Global prompt norm: {}".format(np.linalg.norm(local_es.mean)))
        writer.add_scalar("global_test_acc", global_test_acc, current_round)

        if args.norm_prompt and np.linalg.norm(local_es.mean) < args.prompt_norm_threshold_upper:
            args.prompt_norm_threshold += 1
            model_forward_api.args = args
            print("Set prompt_norm_threshold as {}".format(args.prompt_norm_threshold))
        if args.save_prompt:
            if global_test_acc > best_test_acc:
                best_test_acc = global_test_acc
                torch.save(
                    model_forward_api.model.prompt_embedding.cpu().detach(),
                    "results/llama/sst2/larger_global_pop_new_sigma_pert/fl_prompt.pt",
                )
    else:
        global_test_acc = None

    client_sigmas = {}

    model_forward_api.load_client_record(client_api_setting_list[idx])
    # initialize local data

    train_sample_idxs, dev_sample_idxs = user_dict_train[idx], user_dict_dev[idx]
    print(f"Client {idx} execute local training on {len(train_sample_idxs)} samples...")
    print(f"Client {idx} train_sample_idxs {train_sample_idxs}")

    local_train_data = {
        "input_ids": torch.tensor(train_data["input_ids"].get(train_sample_idxs)),
        "attention_mask": torch.tensor(train_data["attention_mask"].get(train_sample_idxs)),
        "mask_pos": torch.tensor(train_data["mask_pos"].get(train_sample_idxs)),
        "labels": torch.tensor(train_data["labels"].get(train_sample_idxs)),
    }
    local_dev_data = {
        "input_ids": torch.tensor(dev_data["input_ids"].get(dev_sample_idxs)),
        "attention_mask": torch.tensor(dev_data["attention_mask"].get(dev_sample_idxs)),
        "mask_pos": torch.tensor(dev_data["mask_pos"].get(dev_sample_idxs)),
        "labels": torch.tensor(dev_data["labels"].get(dev_sample_idxs)),
    }

    print("Population Size: {}".format(local_es.popsize))
    print("{} Evaluation.".format("Parallel" if parallel else "Serial"))
    if parallel:
        # expand training data to a larger batch for parallel evaluation
        train_data["input_ids"] = train_data["input_ids"].repeat(local_es.popsize, 1)
        train_data["attention_mask"] = train_data["attention_mask"].repeat(local_es.popsize, 1)
        train_data["mask_pos"] = train_data["mask_pos"].repeat(local_es.popsize)
        train_data["labels"] = train_data["labels"].repeat(local_es.popsize)

    local_train_data_aux = perturb_dataset(args, local_train_data, model_forward_api.config)

    model_forward_api.set_dataset(local_train_data, local_dev_data, local_train_data_aux)

    # opt = cma.CMAOptions()
    local_sigmas = []
    start_time = time.time()
    while not local_es.stop():
        local_sigmas.append(local_es.sigma)
        solutions = local_es.ask()
        if args.norm_prompt:
            for i in range(len(solutions)):
                if np.linalg.norm(solutions[i]) > args.prompt_norm_threshold:
                    solutions[i] = solutions[i] / np.linalg.norm(solutions[i]) * args.prompt_norm_threshold
        if parallel:
            fitnesses_orig = model_forward_api.eval(solutions)
            fitnesses_pert = model_forward_api.eval_perturb(solutions)
            if args.perturb != 0:
                fitnesses = fitnesses_orig / fitnesses_pert
            else:
                fitnesses = fitnesses_orig
        else:
            if args.perturb != 0:
                fitnesses = [model_forward_api.eval(x) / model_forward_api.eval_perturb(x) for x in solutions]
            else:
                fitnesses = [model_forward_api.eval(x) for x in solutions]
        local_es.tell(solutions, fitnesses)
        if len(local_sigmas) % 10 == 0:
            test_acc = model_forward_api.eval(prompt_embedding=local_es.mean, test_data=test_data)
            print(f"Local test acc at local iter {len(local_sigmas)}: {round(test_acc, 4)}")
            writer.add_scalar("local_test_acc", test_acc, train_step)
        train_step += 1

    end_time = time.time()
    print("Done. Elapsed time: {} (mins)".format((end_time - start_time) / 60))

    client_prompt_dict[idx].append(copy.deepcopy(local_es.mean))

    # Generate solutions uploaded to the server
    solutions = [local_es.mean]
    if args.norm_prompt:
        for i in range(len(solutions)):
            if np.linalg.norm(solutions[i]) > args.prompt_norm_threshold:
                solutions[i] = solutions[i] / np.linalg.norm(solutions[i]) * args.prompt_norm_threshold
    if parallel:
        fitnesses_orig = model_forward_api.eval(solutions)
        fitnesses_pert = model_forward_api.eval_perturb(solutions)
        if args.perturb != 0:
            fitnesses = fitnesses_orig / fitnesses_pert
        else:
            fitnesses = fitnesses_orig
    else:
        fitnesses_orig = np.array([model_forward_api.eval(x) for x in solutions])
        fitnesses_pert = np.array([model_forward_api.eval_perturb(x) for x in solutions])
        if args.perturb != 0:
            fitnesses = fitnesses_orig / fitnesses_pert
        else:
            fitnesses = fitnesses_orig

    test_acc = model_forward_api.eval(prompt_embedding=local_es.mean, test_data=test_data)
    print(f"Local test acc after current_round {current_round}: {round(test_acc, 4)}")

    print(f"client sigma: {local_sigmas}")

    client_fitnesses_orig_dict[idx].append(copy.deepcopy(fitnesses_orig))
    client_fitnesses_pert_dict[idx].append(copy.deepcopy(fitnesses_pert))

    client_api_setting_list[idx] = model_forward_api.client_record()

    global_api_setting = model_forward_api.client_record()

    # construct trained FL model update
    output_model = flare.FLModel(
        params={
            "solutions": solutions,
            "fitnesses": fitnesses,
            "local_sigmas": local_sigmas,
            "local_cma_mu": local_cma_mu,
        },
        metrics={"global_test_accuracy": global_test_acc},
    )
    # send model back to NVFlare
    flare.send(output_model)
    print("Send params back", list(output_model.params.keys()))
