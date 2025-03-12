# NVFLARE Code Pre-Installer

In production environments, NVFLARE applications often need pre-installed code and shared Python packages. This tool simplifies the pre-installation process by handling both application code and shared resources.

## Overview

The code pre-installer:
1. Installs site-specific job code to a designated directory
2. Installs shared Python packages to a common location
3. Sets up Python path for shared packages
4. Handles package dependencies via requirements.txt

## Job Structure

Required structure in zip file:
```
job_structure.zip
├── job_config/                # Job configuration directory
│   ├── meta.json             # Contains job name and metadata
│   ├── app/                  # Default app (optional)
│   │   └── custom/          # Default custom code
│   ├── app_server/          # Server-specific code
│   │   └── custom/         
│   └── app_site-1/          # Site-specific code
│       └── custom/         
├── requirements.txt          # Python package dependencies
└── job_share/               # Shared resources
    └── pt/                  # Example: shared Python package
```
the job_config can directly copied from the job config from FedJob.export_job("/path/to/job_config")

"/path/to/job_config/job_name"

## Usage

### Command Line
```bash
nvflare pre-install \
    --job-structure /path/to/job_structure.zip \
    --site-name site-1 \
    [--install-prefix /opt/nvflare/jobs] \
    [--share-location /opt/nvflare/share]
```

### Arguments

- `--job-structure`: (Required) Path to job structure zip file
- `--site-name`: (Required) Target site name (e.g., site-1, server)
- `--install-prefix`: Installation directory for job code (default: /opt/nvflare/jobs)
- `--share-location`: Installation directory for shared resources (default: /opt/nvflare/share)

## Examples

1. Install server code:
```bash
nvflare pre-install \
    --job-structure federated_training.zip \
    --site-name server
```

2. Install with custom paths:
```bash
nvflare pre-install \
    --job-structure federated_training.zip \
    --site-name site-1 \
    --install-prefix /custom/jobs \
    --share-location /custom/share
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

### Job requires liberaries in python path.  

Some training code may rely on custom Python packages. These packages should be:
1. Listed in requirements.txt for pip installation
2. Or placed in job_share/ folder if they're custom packages
The installer will:
- Install requirements.txt packages using pip
- Add job_share location to Python path for custom packages
- Make shared packages available to all jobs

