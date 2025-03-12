# NVFLARE Code Pre-Installer

In somecase, especally in production, we need to pre-install all the application code and shared python packages. Once the packages and codes are installed. The submit job will simply submit job configurations. 

To simplify the process, we have a tool to pre-install the code and shared resources.


## Overview

The code pre-installer helps deploy:
1. Site-specific job code to a designated job directory
2. Shared resources (like common Python packages) to a shared location

## Job Structure

Expected job structure in zip file:
```
job_structure.zip
├── job_config/
├── requirements.txt        # Python package dependencies
└── job_share/             # Shared resources
    └── pt/               # Example: shared Python package
```
the job_config can directly copied from the job config from FedJob.export_job("/path/to/job_config")

"/path/to/job_config/job_name"

the job_share/ will be any python packages will be on the python path. 


## Installation

### Command Line Usage
There are two ways to run the installer:

1. Using the shell script:
```bash
./install.sh \
    --job-structure /path/to/job_structure.zip \
    --site-name site-1 \
    [--install-prefix /opt/nvflare/jobs] \
    [--share-location /opt/nvflare/share]
```

2. Using Python module directly:
```bash
python -m nvflare.tool.code_pre_installer.install \
    --job-structure /path/to/job_structure.zip \
    --site-name site-1 \
    [--install-prefix /opt/nvflare/jobs] \
    [--share-location /opt/nvflare/share]
```

### Arguments

- `--job-structure`: (Required) Path to the job structure zip file
- `--site-name`: (Required) Target site name (e.g., site-1, server)
- `--install-prefix`: Installation directory for job code (default: /opt/nvflare/jobs)
- `--share-location`: Installation directory for shared resources (default: /opt/nvflare/share)

### Installation Process

1. Package Installation:
   - If requirements.txt exists, installs required Python packages
   - Uses pip to install dependencies
   - Must have network access or local package repository

2. Job Code Installation:
   - Extracts site-specific code from `app_<site-name>/custom/`
   - Falls back to `apps/custom/` if site-specific directory not found
   - Installs to `<install-prefix>/<job-name>/`

3. Shared Resource Installation:
   - Copies all contents from `job_share/` to `<share-location>/`
   - Maintains directory structure
   - Enables Python package imports from shared location

## Examples

1. Using shell script to install server code:
```bash
./install.sh \
    --job-structure federated_training.zip \
    --site-name server
```

2. Using shell script with custom paths:
```bash
./install.sh \
    --job-structure federated_training.zip \
    --site-name site-1 \
    --install-prefix /custom/jobs \
    --share-location /custom/share
```

3. Using Python module directly:
```bash
python -m nvflare.tool.code_pre_installer.install \
    --job-structure federated_training.zip \
    --site-name server
```

## Directory Structure After Installation

```
<install-prefix>/
└── <job-name>/              # From meta.json
    └── [custom code files]  # From app_<site-name>/custom/

<share-location>/
└── [shared resources]       # From job_share/
```

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


## Submit Job with pre-installed code and shared resources:

### Job using Client APIs

When using the Client API, you need to specify the path to your training script. This path needs to be adjusted based on whether you're using pre-installed code or running in development.

#### Development Environment
```python
# During development, use relative path
task_script_path = "src/client.py"
```

#### Production Environment
```python
# With pre-installed code (default location)
task_script_path = "/opt/nvflare/jobs/fedavg/src/client.py"
```

#### Using Environment Variables
To make your code work in both environments, you can use an environment variable:

```bash
# For production
export NVFLARE_INSTALL_PREFIX="/opt/nvflare/jobs/fedavg/"

# For development
export NVFLARE_INSTALL_PREFIX=""
```

Then in your code:
```python
import os

install_prefix = os.getenv("NVFLARE_INSTALL_PREFIX", "")
task_script_path = f"{install_prefix}src/client.py"
```

### Job requires liberaries in python path.  

Some training code may rely on custom Python packages. These packages should be:
1. Listed in requirements.txt for pip installation
2. Or placed in job_share/ folder if they're custom packages
The installer will:
- Install requirements.txt packages using pip
- Add job_share location to Python path for custom packages
- Make shared packages available to all jobs

