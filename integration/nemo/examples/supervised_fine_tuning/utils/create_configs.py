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
        help="Target job folder containing config files.",
    )
    parser.add_argument("--template_folder", type=str, help="Template job folder", default="jobs/templates")
    parser.add_argument("--num_clients", type=int, help="Number of client app folders to generate.", default=1)
    parser.add_argument("--devices", type=int, help="Number of GPU devices per client.", default=1)
    parser.add_argument(
        "--validation_ds_files",
        nargs="+",
        help="Validation files, one per client.",
    )
    parser.add_argument(
        "--train_ds_files",
        nargs="+",
        help="Training files files, one per client.",
        default="data/FinancialPhraseBank-v1.0_split/site-",
    )

    args = parser.parse_args()
    assert (
        args.num_clients == len(args.validation_ds_files) == len(args.train_ds_files)
    ), "Number of clients should match number of validation and training files."

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
        client_cfg["train_ds_files"] = args.train_ds_files[i]
        client_cfg["validation_ds_files"] = args.validation_ds_files[i]
        if args.devices > 1:
            client_cfg["devices"] = args.devices
        save_config(client_cfg_file, client_cfg)

    # modify server config
    app_folder = os.path.join(args.job_folder, "server")
    shutil.copytree(os.path.join(args.template_folder, "server"), app_folder, dirs_exist_ok=True)

    server_cfg_file = os.path.join(app_folder, "config", "config_fed_server.json")
    server_cfg = load_config(server_cfg_file)
    server_cfg["min_clients"] = args.num_clients
    save_config(server_cfg_file, server_cfg)

    # modify meta.json
    meta_cfg_file = os.path.join(args.job_folder, "meta.json")
    shutil.copyfile(os.path.join(args.template_folder, "meta.json"), meta_cfg_file)
    meta_cfg = load_config(meta_cfg_file)
    meta_cfg["name"] = os.path.basename(args.job_folder)
    meta_cfg["deploy_map"] = {"server": ["server"]}
    for i in range(args.num_clients):
        meta_cfg["deploy_map"][f"app{i+1}"] = [f"site-{i + 1}"]
    save_config(meta_cfg_file, meta_cfg)

    print(f"Created configs for {args.num_clients} clients")


if __name__ == "__main__":
    main()
