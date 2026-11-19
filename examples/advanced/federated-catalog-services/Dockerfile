FROM python:3.11-slim
ARG NVFLARE_PACKAGE
# Setup derives the archive's Versioneer fallback from the installed SDK.
ARG NVFL_BASE_VERSION
WORKDIR /opt/example
COPY .sdk /opt/sdk
# Match the nvflare requirement even for source revisions that default to the nvflare-nightly package name.
RUN NVFL_RELEASE=1 pip install --no-cache-dir "${NVFLARE_PACKAGE:?Run example_setup.py before building}" "pg8000>=1.31,<2"
COPY . .
ENV PYTHONUNBUFFERED=1
CMD ["bash", "/startup-kit/startup/sub_start.sh"]
