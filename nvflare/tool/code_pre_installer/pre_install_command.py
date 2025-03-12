import argparse
from pathlib import Path

from nvflare.cli_exception import CLIException
from nvflare.tool.code_pre_installer.install import install_job_structure


def define_args_parser(parser=None):
    if not parser:
        parser = argparse.ArgumentParser(description="Pre-install NVFLARE job code and shared resources")
    
    parser.add_argument(
        "--job-structure",
        required=True,
        type=Path,
        help="Path to job structure zip file"
    )
    parser.add_argument(
        "--site-name",
        required=True,
        help="Target site name (e.g., site-1, server)"
    )
    parser.add_argument(
        "--install-prefix",
        type=Path,
        default="/opt/nvflare/jobs",
        help="Installation directory for job code (default: /opt/nvflare/jobs)"
    )
    parser.add_argument(
        "--share-location",
        type=Path,
        default="/opt/nvflare/share",
        help="Shared resources location (default: /opt/nvflare/share)"
    )
    return parser


def run(args):
    try:
        install_job_structure(
            args.job_structure,
            args.install_prefix,
            args.share_location,
            args.site_name
        )
    except Exception as e:
        raise CLIException(str(e)) 