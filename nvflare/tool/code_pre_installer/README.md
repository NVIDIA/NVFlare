# NVFLARE Code Pre-Installer

This tool helps install NVFLARE application code and libraries before running federated learning jobs.

## Overview

The code pre-installer handles:
- Installation of application code
- Installation of shared libraries
- Site-specific customizations
- Python package dependencies

## Directory Structure

Expected application code zip structure:

```
application.zip
├── application/<app_name>/
│               ├── meta.json       # Application metadata
│               ├── app_<site>/     # Site custom code
│                  └── custom/      # Site custom code
├── application-share/              # Shared resources
│   └── shared.py
└── requirements.txt       # Python dependencies (optional)
```
or
```
application.zip
├── application/<app_name>/
│               ├── meta.json       # Application metadata
│               ├── app/            # Site custom code
│                  └── custom/      # Site custom code
``` 
Here is an example of creating a folder structure for pre-installation:
```bash
mkdir -p /tmp/nvflare/pre-install/application
mkdir -p /tmp/nvflare/pre-install/application-share
```

For example, if the app name is `fedavg`, the directory structure would look like this:

Tree structure of the job configuration:

/tmp/nvflare/pre-install/
├── application
│   └── fedavg
│       ├── app_server
│       │   ├── config
│       │   └── custom
│       ├── app_site-1
│       │   ├── config
│       │   └── custom
│       ├── app_site-2
│       │   ├── config
│       │   └── custom
│       ├── app_site-3
│       │   ├── config
│       │   └── custom
│       ├── app_site-4
│       │   ├── config
│       │   └── custom
│       ├── app_site-5
│       │   ├── config
│       │   └── custom
│       └── meta.json
└── application-share
    └── pt
        ├── learner_with_mlflow.py
        ├── learner_with_tb.py
        ├── learner_with_wandb.py
        ├── pt_constants.py
        ├── simple_network.py
        └── test_custom.py



Then we can simply copy the `fedavg` folder to the pre-install folder:

```bash
cp -r /tmp/nvflare/jobs/workdir/fedavg /tmp/nvflare/pre-install/application/.
```

If you have shared code (such as Python modules with nested folders and files) in "/tmp/nvflare/jobs/workdir/pt", copy it to the application-share directory:
```bash
cp -r /tmp/nvflare/jobs/workdir/pt /tmp/nvflare/pre-install/application-share/.
```

You should have something like the following:

```
 tree /tmp/nvflare/pre-install/ -L 3
/tmp/nvflare/pre-install/
├── application
│   └── fedavg
│       ├── app_server
│       ├── app_site-1
│       ├── app_site-2
│       ├── app_site-3
│       ├── app_site-4
│       ├── app_site-5
│       └── meta.json
└── application-share
    └── pt
        ├── learner_with_mlflow.py
        ├── learner_with_tb.py
        ├── learner_with_wandb.py
        ├── pt_constants.py
        ├── simple_network.py
        └── test_custom.py

Finally, create the app-code.zip file from the pre-install folder:
```bash
cd /tmp/nvflare/pre-install/
zip -r ../application.zip *
```

The application.zip file will be created in the `/tmp/nvflare/` directory.

## Usage

### Command Line Interface

```bash
nvflare pre-install -a /path/to/application.zip -p /opt/nvflare/apps -s site-1 [-ts /local/custom] [-debug]
```

Arguments:
- `-a, --application`: Path to application code zip file (required)
- `-p, --install-prefix`: Installation prefix (default: /opt/nvflare/apps)
- `-s, --site-name`: Target site name e.g., site-1, server (required)
- `-ts, --target_shared_dir`: Target shared directory path (default: /local/custom)
- `-debug, --debug`: Enable debug mode

### Example

```bash
# Install application code for site-1
nvflare pre-install -a /path/to/myapp.zip -s site-1

# Install with custom paths
nvflare pre-install -a /path/to/myapp.zip -p /custom/install/path -s site-1 -ts /custom/shared/path

# Install with debug output
nvflare pre-install -a /path/to/myapp.zip -s site-1 -debug
```

## Application Code Structure

The application zip file should have the following structure:

```
application/
├── app_name/
│   ├── meta.json
│   ├── app_site-1/
│   │   └── custom/
│   │       └── site_specific_code.py
│   └── app_site-2/
│       └── custom/
│           └── site_specific_code.py
└── application-share/
    └── shared_code.py
```

- `app_name/`: Application directory containing site-specific code
- `meta.json`: Application metadata file
- `app_site-*/custom/`: Site-specific custom code directories
- `application-share/`: Shared code directory

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
- The tool will extract site-specific code to the installation prefix
- Shared code will be installed to the target shared directory
- The application zip file will be cleaned up after installation
- Installation paths must be writable by the current user

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
