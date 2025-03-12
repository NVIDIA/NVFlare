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
├── app_code/
│   ├── meta.json           # Application metadata
│   ├── apps/              # Default application code
│   │   └── custom/        # Default custom code
│   └── app_site-1/        # Site-specific code (optional)
│       └── custom/        # Site custom code
├── app_share/             # Shared resources
│   └── shared.py
└── requirements.txt       # Python dependencies (optional)
```

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
