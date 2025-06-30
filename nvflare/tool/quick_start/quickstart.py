#!/usr/bin/env python3
"""
NVFlare Quick Start CLI tool for initializing federated learning jobs.
"""
import os
import typer
import click
from pathlib import Path
from typing import Optional, List
from enum import Enum
from typer.core import TyperGroup

class NaturalOrderGroup(TyperGroup):
    def list_commands(self, ctx: click.Context) -> list[str]:
        return list(self.commands)

# Set the custom class as the default group class
typer.rich_utils.STYLE_HELPTEXT = ""
typer.rich_utils.STYLE_OPTION = ""
typer.rich_utils.STYLE_SWITCH = ""
typer.rich_utils.STYLE_METAVAR = ""
typer.rich_utils.STYLE_METAVAR_APPEND = ""
typer.rich_utils.STYLE_METAVAR_SEPARATOR = ""
typer.rich_utils.ALIGN_OPTIONS = False

app = typer.Typer(
    cls=NaturalOrderGroup,
    help="NVFlare Quick Start CLI",
    no_args_is_help=True,
    add_help_option=True,
    context_settings={"help_option_names": ["-h", "--help"]}
)

class JobType(str, Enum):    
    PYTORCH = "pytorch"
    PYTORCH_LIGHTNING = "pytorch-lightning"
    TENSORFLOW = "tensorflow"
    NUMPY = "numpy"
    SCIKIT_LEARN = "scikit-learn"
    XGBOOST = "xgboost"
    HUGGINGFACE = "huggingface"
    MONAI = "monai"
    STATISTICS = "statistics"
    DOCKER = "docker"
    POC = "poc"

# Default argument values
DEFAULT_JOB_NAME = "flare-app"
JOB_NAME_ARG = typer.Argument(DEFAULT_JOB_NAME, help="Job name")
JOB_TYPE_OPTION = typer.Option(
    JobType.PYTORCH.value,
    "--type",
    "-t",
    help="Type of the FL job",
    show_default=True,
    case_sensitive=False,
)


def create_directory_structure(base_path: Path, job_type: JobType):
    """Create the directory structure for the FL job."""
    # Create main directories
    dirs = [
        base_path / "app" / "custom",
        base_path / "app" / "config",
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    return dirs


def create_fl_job_files(base_path: Path, job_name: str, job_type: JobType):
    """Create the necessary FL job files."""
    # Create client.py
    client_py = base_path / "app" / "custom" / "client.py"
    server_py = base_path / "app" / "custom" / "server.py"
    fl_py = base_path / "app" / "fl.py"
    
    # Create client config
    client_config = base_path / "app" / "config" / "fed_client_config.conf"
    server_config = base_path / "app" / "config" / "fed_server_config.conf"
    
    # Write basic client.py
    client_py.write_text(
        f'"""Client implementation for {job_name} FL job."""\n'
        'import nvflare.client as flare\n\n'
        'def main():\n'
        '    # Initialize NVFlare client\n'
        '    flare.init()\n\n'
        '    # Your federated learning client code here\n'
        '    print(f"{job_name} client started")\n\n'
        'if __name__ == "__main__":\n'
        '    main()\n'
    )
    
    # Write basic server.py
    server_py.write_text(
        f'"""Server implementation for {job_name} FL job."""\n'
        'import nvflare.client as flare\n\n'
        'def main():\n'
        '    # Initialize NVFlare server\n'
        '    flare.init()\n\n'
        '    # Your federated learning server code here\n'
        '    print(f"{job_name} server started")\n\n'
        'if __name__ == "__main__":\n'
        '    main()\n'
    )
    
    # Write basic fl.py
    fl_py.write_text(
        f'"""Main entry point for {job_name} FL job."""\n\n'
        'def main():\n'
        '    # Your federated learning main code here\n'
        '    print(f"{job_name} FL job started")\n\n'
        'if __name__ == "__main__":\n'
        '    main()\n'
    )
    
    # Write basic config files
    client_config.write_text(
        f'[client]\n'
        f'name = {job_name}_client\n'
        'server_host = localhost\n'
        'server_port = 8002\n'
    )
    
    server_config.write_text(
        f'[server]\n'
        f'name = {job_name}_server\n'
        'host = 0.0.0.0\n'
        'port = 8002\n'
    )
    
    return [client_py, server_py, fl_py, client_config, server_config]


@app.command(no_args_is_help=True)
def init(
    job_name: str = JOB_NAME_ARG,
    job_type: JobType = JOB_TYPE_OPTION,
):
    """Initialize a new federated learning job."""
    base_path = Path(job_name).absolute()
    
    # Create directory structure
    create_directory_structure(base_path, job_type)
    
    # Create FL job files
    create_fl_job_files(base_path, job_name, job_type)
    
    # Print success message
    typer.echo(f"✔ Created new federated learning job: {job_name}")
    typer.echo(f"├── app/")
    typer.echo(f"│   ├── custom/client.py")
    typer.echo(f"│   ├── custom/server.py")
    typer.echo(f"│   ├── config/fed_client_config.conf")
    typer.echo(f"│   ├── config/fed_server_config.conf")
    typer.echo(f"│   └── fl.py")
    typer.echo(f"💬 Job type: {job_type.value.capitalize()}")
    typer.echo(f"📁 Location: {base_path}")


if __name__ == "__main__":
    app()
