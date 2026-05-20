# NVIDIA FLARE with Docker (Legacy)

This notebook shows the older provision-time Docker script flow based on
`StaticFileBuilder.docker_image`, which generates `startup/docker.sh`.

For new Docker deployments, use the current Docker job launcher example in
[`examples/docker`](../../docker/README.md). That example uses
`nvflare deploy prepare`, `startup/start_docker.sh`, and job-level
`launcher_spec` settings for per-job Docker containers.
