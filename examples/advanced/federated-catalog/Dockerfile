FROM python:3.11-slim
ARG NVFL_BASE_VERSION=2.10.0
WORKDIR /opt/NVFlare
COPY setup.py setup.cfg pyproject.toml versioneer.py README.md LICENSE MANIFEST.in ./
COPY nvflare ./nvflare
RUN pip install --no-cache-dir . "SQLAlchemy>=2.0,<3" "pyarrow>=14"
COPY examples/advanced/federated-catalog /opt/example
WORKDIR /opt/example
ENV PYTHONUNBUFFERED=1
CMD ["bash", "/startup-kit/startup/sub_start.sh"]
