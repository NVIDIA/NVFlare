import argparse
import os
import shutil

import wget


def parse_args(prog_name: str):
    _parser = argparse.ArgumentParser(description=prog_name)
    _parser.add_argument(
        "--prepare-data",
        dest="prepare_data",
        action="store_const",
        const=prepare_data,
        help="prepare data based on configuration",
    )
    _parser.add_argument(
        "-d",
        "--dest",
        type=str,
        nargs="?",
        default="",
        help="destination directory where the data to download to",
    )
    return _parser, _parser.parse_args()


def get_data_url() -> dict:
    client_data = {
        "site-1": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "site-2": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    }
    return client_data


def prepare_data(data_root_dir: str):
    print(f"prepare data for data directory {data_root_dir}")
    client_data_urls = get_data_url()
    for client in client_data_urls:
        client_data_dir = os.path.join(data_root_dir, client)
        if not os.path.exists(client_data_dir):
            os.makedirs(client_data_dir, exist_ok=True)

        dest = os.path.join(client_data_dir, "data.csv")
        if os.path.exists(dest):
            print(f"\nremove existing data at {dest}")
            shutil.rmtree(dest, ignore_errors=True)

        print(f"\nwget download to {dest}")
        url = client_data_urls[client]
        response = wget.download(url, dest)
    print("\ndone with prepare data")


def main():
    prog_name = "data_utils"
    parser, args = parse_args(prog_name)

    if args.prepare_data:
        prepare_data(args.dest)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
