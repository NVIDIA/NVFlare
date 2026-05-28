# NVFlare Container Images

NVFlare uses separate images for parent processes and user jobs. This keeps the
control-plane runtime small while leaving user workloads free to use a richer
framework image.

## Parent Image

`Dockerfile.parent` builds the image used by long-running NVFlare server and
client parent processes. These processes coordinate jobs, manage communication,
and launch workload containers.

The final stage uses NVIDIA Distroless Python. Distroless images contain only
the application and runtime dependencies needed to run it. They intentionally do
not include a shell, package manager, or general-purpose OS utilities. This
reduces image size and attack surface, but it also means debugging should use
container logs and NVFlare logs instead of `docker exec sh`.

Build from the repository root:

```bash
docker build -f docker/Dockerfile.parent -t nvflare-parent:latest .
```

Use this image in `docker.yaml` or `k8s.yaml` as the parent image:

```yaml
parent:
  docker_image: nvflare-parent:latest
```

## Job Image

`Dockerfile.job` builds a starter image for submitted jobs. Job containers run
user training, evaluation, or data-processing code and normally need the ML
framework stack.

The starter job image uses `nvcr.io/nvidia/pytorch:26.04-py3` as its base. The
NGC PyTorch container is a ready-to-run GPU framework image with CUDA, NVIDIA
libraries, PyTorch, and related runtime dependencies. The Dockerfile installs
NVFlare with the `K8S` extra, but does not install the NVFlare `PT` extra so pip
does not replace the PyTorch packages supplied by the NGC image.

Build from the repository root:

```bash
docker build -f docker/Dockerfile.job -t nvflare-job:latest .
```

Use this image in submitted job metadata:

```json
{
  "launcher_spec": {
    "default": {
      "docker": {"image": "nvflare-job:latest"},
      "k8s": {"image": "nvflare-job:latest"}
    }
  }
}
```

Extend `Dockerfile.job` for project-specific dependencies, datasets, model
code, or framework packages. If you need to change packages constrained by the
NGC PyTorch image, review the base image release notes first; recent PyTorch
containers include `/etc/pip/constraint.txt` to protect the tested package set.

## Choosing an Image

- Use `Dockerfile.parent` for parent server/client processes.
- Use `Dockerfile.job` for submitted job containers.
- Do not rely on a shell being available in the parent image; it is absent by
  design.
- Add user workload dependencies to the job image, not the parent image, unless
  the parent process itself needs them.

## References

- NVIDIA Distroless documentation:
  https://developer.nvidia.com/w/distroless-oss/docs.html
- NGC PyTorch container:
  https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
- PyTorch 26.04 release notes:
  https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-04.html
- NGC Catalog user guide:
  https://docscontent.nvidia.com/dita/00000186-18ab-dad2-a9a7-5eafb5c20000/ngc/gpu-cloud/pdf/ngc-catalog-user-guide.pdf
