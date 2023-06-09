# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import json
import os
import shutil


def load_config(config_file):
    with open(config_file, "r") as f:
        return json.load(f)


def save_config(config_file, config):
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job_folder",
        type=str,
        help="Folder containing job config files in JSON format.",
    )
    parser.add_argument(
        "--template_folder", type=str, help="Client config template in JSON format.", default="jobs/templates"
    )
    parser.add_argument("--num_clients", type=int, help="Number of client app folders to generate.", default=3)
    parser.add_argument("--aggregation_epochs", type=int, help="Local number of aggregation epochs.", default=1)
    parser.add_argument("--num_rounds", type=int, help="Number of FL rounds.", default=1)
    parser.add_argument("--devices", type=int, help="Number of GPU devices per client.", default=1)
    parser.add_argument(
        "--root_dir", type=str, help="Root folder containing the example with data and models.", default=os.getcwd()
    )
    parser.add_argument(
        "--val_ds_files",
        type=str,
        help="Validation files.",
        default="data/FinancialPhraseBank-v1.0/financial_phrase_bank_val.jsonl",
    )
    parser.add_argument(
        "--train_ds_files_prefix",
        type=str,
        help="Training files prefix.",
        default="data/FinancialPhraseBank-v1.0_split/site-",
    )

    args = parser.parse_args()

    # create client app folders
    for i in range(args.num_clients):
        app_folder = os.path.join(args.job_folder, f"app{i+1}")
        client_cfg_file = os.path.join(app_folder, "config", "config_fed_client.json")
        shutil.copytree(os.path.join(args.template_folder, "client"), app_folder, dirs_exist_ok=True)

        # remove unused client config
        if isinstance(args.devices, int) and args.devices == 1:
            os.remove(os.path.join(app_folder, "config", "config_fed_client_multiprocess.json"))
        elif isinstance(args.devices, int) and args.devices > 1:
            shutil.move(os.path.join(app_folder, "config", "config_fed_client_multiprocess.json"), client_cfg_file)
        else:
            raise ValueError(f"Number client devices should be positive integer but was {args.devices}")

        # modify client configs
        client_cfg = load_config(client_cfg_file)
        client_cfg["ROOT_DIR"] = args.root_dir
        client_cfg["train_ds_files"] = os.path.join(args.root_dir, f"{args.train_ds_files_prefix}{i + 1}.jsonl")
        client_cfg["val_ds_files"] = os.path.join(args.root_dir, args.val_ds_files)
        client_cfg["aggregation_epochs"] = args.aggregation_epochs
        if args.devices > 1:
            client_cfg["devices"] = args.devices
        save_config(client_cfg_file, client_cfg)

    # modify server config
    app_folder = os.path.join(args.job_folder, "server")
    shutil.copytree(os.path.join(args.template_folder, "server"), app_folder, dirs_exist_ok=True)

    server_cfg_file = os.path.join(app_folder, "config", "config_fed_server.json")
    server_cfg = load_config(server_cfg_file)
    server_cfg["min_clients"] = args.num_clients
    server_cfg["num_rounds"] = args.num_rounds
    if args.devices > 1:
        server_cfg["hidden_size"] = 6144  # use for 20B GPT model
    save_config(server_cfg_file, server_cfg)

    # modify meta.json
    meta_cfg_file = os.path.join(args.job_folder, "meta.json")
    shutil.copyfile(os.path.join(args.template_folder, "meta.json"), meta_cfg_file)
    meta_cfg = load_config(meta_cfg_file)
    meta_cfg["name"] = os.path.basename(args.job_folder)
    meta_cfg["deploy_map"] = {"server": ["server"]}
    for i in range(args.num_clients):
        meta_cfg["deploy_map"][f"app{i+1}"] = [f"site-{i+1}"]
    save_config(meta_cfg_file, meta_cfg)

    print(f"Created configs for {args.num_clients} clients and set ROOT_DIR to {args.root_dir}")


if __name__ == "__main__":
    main()
