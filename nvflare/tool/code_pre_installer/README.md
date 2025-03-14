# NVFLARE Code Pre-Installer

A tool to pre-install NVFLARE application code and libraries.

## Overview

The code pre-installer handles:
- Installation of application code
- Installation of shared libraries
- Site-specific customizations
- Python package dependencies

## Directory Structure

Expected application code zip structure:
```
app_code.zip
folder/
├── app_code/<app_name>/
│               ├── meta.json       # Application metadata
│               ├── app-<site>/     # Site custom code
|                  |
│                  └── custom/      # Site custom code
├── app_share/                      # Shared resources
│   └── shared.py
└── requirements.txt       # Python dependencies (optional)
```

Here is an example, if we like to create a folder for pre-instalation. We can do the following

create a folder for pre-installation:
```bash
mkdir -p /tmp/nvflare/pre-install/app-code
mkdir -p /tmp/nvflare/pre-install/app-share
```
now, I an job configuration like the following:
``` 

For example, if the app name is `fedavg`, and we have the job configuration likes the following:


Tree structure of the job configuration:

/tmp/nvflare/pre-install/
├── app-code
│   └── fedavg
│       ├── app_server
│       │   ├── config
│       │   └── custom
│       ├── app_site-1
│       │   ├── config
│       │   └── custom
│       ├── app_site-2
│       │   ├── config
│       │   └── custom
│       ├── app_site-3
│       │   ├── config
│       │   └── custom
│       ├── app_site-4
│       │   ├── config
│       │   └── custom
│       ├── app_site-5
│       │   ├── config
│       │   └── custom
│       └── meta.json
└── app-share
    └── pt
        ├── learner_with_mlflow.py
        ├── learner_with_tb.py
        ├── learner_with_wandb.py
        ├── pt_constants.py
        ├── simple_network.py
        └── test_custom.py



Then we can simply copy the `fedavg` folder to the pre-install folder:

```bash
cp -r /tmp/nvflare/jobs/workdir/fedavg /tmp/nvflare/pre-install/app-code/.
```
if we have the shared code, and shared the code module is a python module folder with nested folders and files "/tmp/nvflare/jobs/workdir/pt".  we can copy the module to the pre-install folder:


to the pre-install folder:
```bash
cp /tmp/nvflare/jobs/workdir/pt /tmp/nvflare/pre-install/app-share/.
```

You should have something like the following:

```
 tree /tmp/nvflare/pre-install/ -L 3
/tmp/nvflare/pre-install/
├── app-code
│   └── fedavg
│       ├── app_server
│       ├── app_site-1
│       ├── app_site-2
│       ├── app_site-3
│       ├── app_site-4
│       ├── app_site-5
│       └── meta.json
└── app-share
    └── pt
        ├── learner_with_mlflow.py
        ├── learner_with_tb.py
        ├── learner_with_wandb.py
        ├── pt_constants.py
        ├── simple_network.py
        └── test_custom.py

```
you can then zip the pre-install folder and use it as the app-code.zip for the code pre-installer.

```bash
cd /tmp/nvflare/pre-install/

zip -r ../app-code.zip * 
```
you should have the app-code.zip file in the ```/tmp/nvflare/``` folder.


## Usage

```bash
python -m nvflare.tool.code_pre_installer.install \
    --app-code /path/to/app_code.zip \
    --install-prefix /opt/nvflare/apps \
    --site-name site-1
```

## Installation Paths

- Application code: `<install-prefix>/<app-name>/`
- Shared resources: `/local/custom/`


## Error Handling

The installer will fail if:
- Job structure zip is invalid or missing required directories
- meta.json is missing or invalid
- Site directory not found and no default apps available
- Installation directories cannot be created
- File operations fail
- Package installation fails (if requirements.txt present)

## Notes

- Existing files may be overwritten
- Python path is automatically configured for shared packages
- All file permissions are preserved during installation
- Network access needed if requirements.txt present
- Can use private PyPI server by configuring pip

## Using Pre-installed Code

### In Job Configuration (JSON)
```json
{
    "task_script": "${NVFLARE_INSTALL_PREFIX}/src/client.py",
    "other_config": "..."
}
```

### In Development
#### JSON Config
```bash
export NVFLARE_INSTALL_PREFIX=""  # Empty for development
```

#### Python Code
```python
task_script_path = "src/client.py"
```

### In Production
#### JSON Config
```bash
export NVFLARE_INSTALL_PREFIX="/opt/nvflare/jobs/fedavg"  # Set for production
```

#### Python Code
```python
# With pre-installed code (default location)
task_script_path = "/opt/nvflare/jobs/fedavg/src/client.py"
```

#### Using Environment Variables
The environment variable works for both JSON configs and Python code:

```bash
# For production
export NVFLARE_INSTALL_PREFIX="/opt/nvflare/jobs/fedavg/"

# For development
export NVFLARE_INSTALL_PREFIX=""
```

#### In JSON Config
```json
{
    "task_script": "${NVFLARE_INSTALL_PREFIX}src/client.py"
}
```

#### In Python Code
```python
import os

install_prefix = os.getenv("NVFLARE_INSTALL_PREFIX", "")
task_script_path = f"{install_prefix}src/client.py"
```
