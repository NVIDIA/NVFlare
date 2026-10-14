## Overview

This is a simple example to show how to build an application Docker image for NVFlare.

### Prerequisites

Before proceeding, ensure you have Docker installed and running on your machine. Additionally, make sure you have access to the required NVFlare code and resources.

### Build the Docker Image

We have provided a `Dockerfile` to build the NVFlare Docker image.

To build the NVFlare Docker image, run the following command in your terminal:

```bash
./docker_build.sh Dockerfile nvflare-site
```

Important notes:
- Code directory: The local ``code`` folder is copied into the Docker image under ``/local/custom`` and placed on `PYTHONPATH`.
- Job configuration: The [job configuration](../jobs/hello-pt_cifar10_fedavg/app_site-1/config/config_fed_client.json) selects `CVMClientAPIExecutor`. That measured-image class fixes in-process execution and `/local/custom/client.py`; a submitted job cannot select an external command or another script path.

Make sure to adjust the paths and filenames in your configurations accordingly.
