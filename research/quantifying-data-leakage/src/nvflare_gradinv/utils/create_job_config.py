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
        json.dump(config, f, indent=4)


def create_server_config(server_config_file, args):
    server_config = load_config(server_config_file)

    server_config["min_clients"] = args.n_clients
    server_config["num_rounds"] = args.num_rounds

    out_server_config_file = os.path.join(args.output, "server", "config", "config_fed_server.json")
    if not os.path.isdir(os.path.dirname(out_server_config_file)):
        os.makedirs(os.path.dirname(out_server_config_file))

    save_config(out_server_config_file, server_config)
    print(
        f"Created {out_server_config_file} to use min_clients={server_config['min_clients']} and num_rounds={server_config['num_rounds']}"
    )


def create_client_config(client_config_file, args, client, bs):
    client_config = load_config(client_config_file)

    client_config["DATA_ROOT"] = args.data_root
    client_config["DATASET_JSON"] = args.dataset_json_prefix + f"{client}.json"

    client_config["BATCH_SIZE"] = bs

    client_config["SIGMA0"] = args.sigma0

    if args.prior_file and "PRIOR_FILE" in client_config:
        client_config["PRIOR_FILE"] = args.prior_file

    out_client_config_file = os.path.join(args.output, f"site-{client}", "config", "config_fed_client.json")
    if not os.path.isdir(os.path.dirname(out_client_config_file)):
        os.makedirs(os.path.dirname(out_client_config_file))

    save_config(out_client_config_file, client_config)

    print(
        f"Created {out_client_config_file} to use DATA_ROOT={client_config['DATA_ROOT']}, DATASET_JSON={client_config['DATASET_JSON']}, and SIGMA0={client_config['SIGMA0']}"
    )
    if "PRIOR_FILE" in client_config:
        print(f"Used prior_file {client_config['PRIOR_FILE']}")


def create_gradinv_config(inv_config_file, args, client):
    inv_config = load_config(inv_config_file)

    inv_config["img_prior"] = args.prior_file

    out_inv_config_file = os.path.join(args.output, f"site-{client}", "config", "config_inversion.json")
    if not os.path.isdir(os.path.dirname(out_inv_config_file)):
        os.makedirs(os.path.dirname(out_inv_config_file))

    save_config(out_inv_config_file, inv_config)

    print(f"Created {out_inv_config_file} to use prior_file {inv_config['img_prior']}")


def create_meta_config(clients, meta_config_file, args):
    meta_config = load_config(meta_config_file)

    deploy_map = {"server": ["server"]}
    for client in clients:
        deploy_map.update({f"site-{client}": [f"site-{client}"]})

    meta_config["deploy_map"] = deploy_map

    out_meta_config_file = os.path.join(args.output, "meta.json")
    if not os.path.isdir(os.path.dirname(out_meta_config_file)):
        os.makedirs(os.path.dirname(out_meta_config_file))

    save_config(out_meta_config_file, meta_config)

    print(f"Created {out_meta_config_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--app_folder",
        type=str,
        help="Folder containing app config files in JSON format.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output folder containing the job config.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        help="Data root.",
    )
    parser.add_argument("--prior_file", type=str, help="Prior image filename")
    parser.add_argument(
        "--dataset_json_prefix",
        type=str,
        default="data_200val+rest_client",
        help="Prefix name for dataset JSON file.",
    )
    parser.add_argument(
        "--n_clients",
        type=int,
        default=1,
        help="Number of clients to generate.",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=100,
        help="Number of federated learning rounds.",
    )
    parser.add_argument(
        "--invert_clients",
        type=str,
        default="9",
        help="Comma-separated list of client numbers to invert. Defaults to `high-risk` client 9.",
    )
    parser.add_argument(
        "--batch_sizes",
        type=str,
        default="1",
        help="Comma-separated list of batches for each client.",
    )
    parser.add_argument(
        "--sigma0",
        type=float,
        default=0.0,
        help="Noise level for `GaussianPrivacy` filter.",
    )
    args = parser.parse_args()

    if os.path.isdir(args.output):
        print(f"Deleting previous job_config at {args.output}")
        shutil.rmtree(args.output)

    invert_clients = [int(x) for x in args.invert_clients.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    # Check template app folder
    assert os.path.isdir(args.app_folder), f"app_folder does not exist at {args.app_folder}"

    # create server config
    server_config_file = os.path.join(args.app_folder, "server", "config", "config_fed_server.json")
    create_server_config(server_config_file, args)

    # if n_clients is 1, we use the "high-risk" client 9's dataset
    if args.n_clients == 1:
        clients = [9]
    else:
        clients = list(range(1, args.n_clients + 1))

    # create meta.json
    meta_config_file = os.path.join(args.app_folder, "meta.json")
    create_meta_config(clients, meta_config_file, args)

    # create client configs
    for client, bs in zip(clients, batch_sizes):
        # if n_clients is 1, we use the "high-risk" client 9's dataset
        if args.n_clients == 1:
            client = 9

        if client in invert_clients:
            print(f"Inverting site-{client} ...")
            client_config_file = os.path.join(args.app_folder, "client", "config", "config_fed_client_inv.json")
            inv_config_file = os.path.join(args.app_folder, "client", "config", "config_inversion.json")

            # create gradient inversion config
            create_gradinv_config(inv_config_file, args, client)
        else:
            client_config_file = os.path.join(args.app_folder, "client", "config", "config_fed_client_noinv.json")

        # create client config file
        create_client_config(client_config_file, args, client, bs)


if __name__ == "__main__":
    main()
