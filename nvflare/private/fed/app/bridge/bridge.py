import argparse
import logging
import logging.config
import sys
import threading

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.net_agent import NetAgent
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.mpm import MainProcessMonitor as mpm
from nvflare.fuel.utils.config_service import ConfigService, search_file

SSL_ROOT_CERT = "rootCA.pem"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", "-w", type=str, help="WORKSPACE folder", required=True)
    parser.add_argument("--name", "-n", type=str, help="my basename", required=True)
    parser.add_argument("--parent_fqcn", "-pn", type=str, help="parent fqcn", required=False, default="")
    parser.add_argument("--parent_url", "-pu", type=str, help="parent url", required=False, default="")
    parser.add_argument("--root_url", "-r", type=str, help="root url", required=False, default="")
    args = parser.parse_args()

    if not args.parent_url and not args.root_url:
        # either parent or root must be specified
        raise ValueError("either parent or root must be specified")

    if args.parent_url and args.root_url:
        raise ValueError("either parent or root must be specified, but not both")

    if args.parent_url and not args.parent_fqcn:
        raise ValueError("parent fqcn must be specified")

    return args


def main(args):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    credentials = {}
    secure_mode = True
    stop_event = threading.Event()

    ConfigService.initialize(
        section_files={},
        config_path=[args.workspace],
    )

    if secure_mode:
        root_cert_path = search_file(SSL_ROOT_CERT, args.workspace)
        if not root_cert_path:
            raise ValueError(f"cannot find {SSL_ROOT_CERT} from config path {args.workspace}")

        credentials = {
            DriverParams.CA_CERT.value: root_cert_path,
        }

    if not args.parent_fqcn:
        my_fqcn = args.name
    else:
        my_fqcn = FQCN.join([args.parent_fqcn, args.name])

    cell = Cell(
        fqcn=my_fqcn,
        root_url=args.root_url,
        secure=secure_mode,
        credentials=credentials,
        create_internal_listener=True,
        parent_url=args.parent_url,
    )
    net_agent = NetAgent(cell)
    cell.start()

    # wait until stopped
    print(f"started bridge {args.name} ...")
    stop_event.wait()


if __name__ == "__main__":
    args = parse_arguments()
    rc = mpm.run(main_func=main, run_dir=args.workspace, args=args)
    sys.exit(rc)
