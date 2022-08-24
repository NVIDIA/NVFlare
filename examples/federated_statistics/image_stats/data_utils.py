import argparse
import os
import sys

from nvflare.lighter.poc_commands import is_poc_ready, get_nvflare_home


def get_poc_workspace():
    poc_workspace = os.getenv("NVFLARE_POC_WORKSPACE")
    if poc_workspace is None or len(poc_workspace.strip()) == 0:
        poc_workspace = "/tmp/nvflare/poc"
    return poc_workspace


def parse_args(prog_name: str):
    _parser = argparse.ArgumentParser(description=prog_name)
    _parser.add_argument(
        "--prepare-data",
        dest="prepare_data",
        action="store_const",
        const=prepare_data,
        help="prepare data",
    )
    _parser.add_argument(
        "-src_path",
        type=str,
        nargs="?",
        default="",
        help="source data path",
    )
    return _parser, _parser.parse_args()


def get_clients(poc_workspace):
    clients = []
    for root, dirs, files in os.walk(poc_workspace):
        if root == poc_workspace:
            for dir_name in dirs:
                if dir_name.startswith("site"):
                    clients.append(dir_name)
    return clients


def partition_data(poc_workspace, src_dir):
    data_partitions = {}
    sites = get_clients(poc_workspace)
    sites_size = len(sites)
    for root, dirs, files in os.walk(src_dir):
        if root == src_dir:
            for i, dir_name in enumerate(dirs):
                k = i % sites_size
                site = f"site-{k+1}"
                if site in data_partitions:
                    paths = data_partitions[site]
                else:
                    paths = []

                dir_path = os.path.join(root, dir_name)
                paths.append(dir_path)
                data_partitions.update({site: paths})

    return data_partitions


def prepare_data(poc_workspace, src_dir):
    print(f"prepare data for poc workspace:{poc_workspace}")
    if not is_poc_ready(poc_workspace):
        print("poc workspace is not ready, please use `nvflare poc -w <poc_workspace> --prepare` to setup first")
        sys.exit(1)

    client_data_partitions = partition_data(poc_workspace, src_dir)
    for client in client_data_partitions:
        data_path = os.path.join(poc_workspace, client, "data")
        print(f"prepare data for client:{client} at path: {data_path}")
        if not os.path.exists(data_path):
            os.mkdir(data_path)

        for src_path in client_data_partitions[client]:
            dest_path = os.path.join(data_path, os.path.basename(src_path))

            if os.path.exists(dest_path) or os.path.islink(dest_path):
                os.remove(dest_path)
            try:
                os.symlink(src_path, dest_path)
            except Exception as e:
                print(f"{dest_path} exists ?= ", os.path.exists(dest_path))
                raise e


def main():
    prog_name = "data_utils"
    parser, args = parse_args(prog_name)

    if args.prepare_data:
        prepare_data(get_poc_workspace(), args.src_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
