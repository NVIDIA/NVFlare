## Overview

This is a simple example to show how to build an application Docker image for NVFlare.

### Prerequisites

Before proceeding, ensure you have Docker installed and running on your machine. Additionally, make sure you have access to the required NVFlare code and resources.

### Build the Docker Image

We have provided a `Dockerfile` to build the NVFlare Docker image.
Use `Dockerfile` for the GPU example and `Dockerfile.cpu` for CPU-only
participants. Both pin the application framework dependencies; the CPU variant
uses PyTorch's explicit `+cpu` wheels and omits CUDA dependencies and TensorFlow,
which the CIFAR10 client does not use. The matching versions and wheel index
are documented by [PyTorch](https://pytorch.org/get-started/previous-versions/).

To build the NVFlare Docker image, run the following command in your terminal:

```bash
./docker_build.sh Dockerfile nvflare-site /srv/nvflare/images/application.tar
```

For the CPU-only federation instead:

```bash
./docker_build.sh Dockerfile.cpu nvflare-site-cpu /srv/nvflare/images/application.tar
```

The third argument is required. The script writes an uncompressed Docker-save
archive at that exact path, matching ``workload.source.path`` in the participant
configuration, and prints the provisioning command to run next.

"Uncompressed" describes the **outer `.tar`**, not its image layers. A valid
Docker save archive may contain compressed layers, particularly with Docker's
containerd image store. They are accepted and expanded during import. Different
Docker versions/stores can therefore produce different archive sizes for the
same installed image. Do not gzip the outer tar or compare only archive sizes.

Measure the produced inputs before choosing `vault_size_gib`:

```bash
stat --printf='%s bytes in Docker-save archive\n' /srv/nvflare/images/application.tar
docker image inspect nvflare-site-cpu --format '{{.Size}} bytes in installed image'
```

Budget for the retained archive, unpacked archive/layers, Docker's imported
image, application files, expected writes and safety margin. The GPU image is
larger because it includes CUDA-capable libraries. The builder's capacity check
estimates expanded content and rejects an undersized vault; see
[vault sizing](../../../../nvflare/lighter/cc/image_builder/BUILD_GUIDE.md#4-seal-an-application-delivery).
Record the final image digest and resolved package versions with the release;
top-level framework pins alone are not a hash-locked production dependency set.

Important notes:
- Code directory: The local ``code`` folder is copied into the Docker image under ``/local/custom`` and placed on `PYTHONPATH`.
- Job configuration: The [job configuration](../jobs/hello-pt_cifar10_fedavg/app_site-1/config/config_fed_client.json) selects `CVMClientAPIExecutor`. That measured-image class fixes in-process execution and `/local/custom/client.py`; a submitted job cannot select an external command or another script path.

Make sure to adjust the paths and filenames in your configurations accordingly.
