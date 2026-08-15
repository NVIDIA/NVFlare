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

The dashboard can use the same runtime image. If you want a dashboard-specific
image name, tag the same build with both names:

```bash
docker build -f docker/Dockerfile.parent \
  -t nvflare-parent:latest \
  -t nvflare-dashboard:latest \
  .
```

Then start the dashboard with the dashboard tag:

```bash
nvflare dashboard --start -i nvflare-dashboard:latest --port 8443
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
NVFlare without optional extras so the job image does not inherit parent-only
launcher dependencies and pip does not replace the PyTorch packages supplied by
the NGC image.

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

## Confidential Containers Parent Image

`Dockerfile.coco` builds a generic parent image for NVFlare deployments using
Confidential Containers. It adds the pinned AMD SEV-SNP, Intel TDX, and NVIDIA
NVAT attestation clients needed by the confidential-computing authorizers to
the standard parent runtime. It does not copy application code from `examples/`
or install dependencies for a particular workload.

Unlike `Dockerfile.parent`, which currently uses Python 3.14,
`Dockerfile.coco` intentionally stays on Python 3.12. Its
`PYTHON_BUILD_BASE`, `PYTHON_RUNTIME_BASE`, and `PYTHON_MINOR` values must be
updated together; do not automatically synchronize them to the parent image's
Python version.

Build it from the repository root:

```bash
docker build -f docker/Dockerfile.coco -t nvflare-coco:latest .
```

Use it as the parent image in a Confidential Containers deployment, then
provide workload code and dependencies through the selected job-launcher
mechanism. Pin, sign, and authorize the resulting image digest according to the
deployment's attestation policy.

## Choosing an Image

- Use `Dockerfile.parent` for non-CoCo parent server/client processes.
- Use `Dockerfile.parent` for the dashboard runtime; tag the same build as a
  dashboard image if you want a dashboard-specific image name.
- Use `Dockerfile.coco` for CoCo server/client parent processes that need the
  bundled attestation clients.
- Use `Dockerfile.job` for submitted job containers.
- Do not rely on a shell being available in the parent image; it is absent by
  design.
- Add user workload dependencies to the job image when using a container job
  launcher. With a process job launcher, extend and sign a deployment-specific
  derivative of the parent image instead of adding workload code to the generic
  Dockerfile.

## References

- NVIDIA Distroless documentation:
  https://developer.nvidia.com/w/distroless-oss/docs.html
- NGC PyTorch container:
  https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
- PyTorch 26.04 release notes:
  https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-04.html
- NGC Catalog user guide:
  https://docscontent.nvidia.com/dita/00000186-18ab-dad2-a9a7-5eafb5c20000/ngc/gpu-cloud/pdf/ngc-catalog-user-guide.pdf
