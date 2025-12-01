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
- Code directory: The local ``code`` folder is copied into the Docker image under ``/local/custom``
- Job configuration: In the [job configuration](../jobs/hello-pt_cifar10_fedavg/app_site-1/config/config_fed_client.json) we configure the task script to be ``/local/custom/client.py``

Make sure to adjust the paths and filenames in your configurations accordingly.
