import argparse
import os
import shutil
import sys
import wget

from nvflare.lighter.poc_commands import is_poc_ready


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
        help="prepare data based on configuration",
    )
    return _parser, _parser.parse_args()


def get_data_url() -> dict:
    client_data = {"site-1": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                   "site-2": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"}
    return client_data


def prepare_data(poc_workspace):
    print(f"prepare data for poc workspace:{poc_workspace}")
    if not is_poc_ready(poc_workspace):
        print("poc workspace is not ready, please use `nvflare poc -w <poc_workspace> --prepare` to setup first")
        sys.exit(1)

    client_data_urls = get_data_url()
    for client in client_data_urls:
        dest = os.path.join(poc_workspace, f"{client}/data.csv")
        print(f"remove existing data at {dest}")
        shutil.rmtree(dest, ignore_errors=True)

        print(f"wget download to {dest}")
        url = client_data_urls[client]
        response = wget.download(url, dest)
    print("done with prepare data")


def main():
    prog_name = "data_utils"
    parser, args = parse_args(prog_name)

    if args.prepare_data:
        prepare_data(get_poc_workspace())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
