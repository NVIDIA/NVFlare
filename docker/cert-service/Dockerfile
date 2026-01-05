# Dockerfile for NVIDIA FLARE Certificate Service
#
# This is a minimal image for running the Certificate Service independently.
# It contains only the dependencies needed for certificate management and
# token-based enrollment.
#
# Build (from project root):
#   docker build -f docker/cert-service/Dockerfile -t nvflare/cert-service:latest .
#
# Run:
#   docker run -d \
#     -p 8443:8443 \
#     -v /path/to/data:/var/lib/cert_service \
#     -e NVFLARE_API_KEY=your-api-key \
#     nvflare/cert-service:latest
#
# With custom config:
#   docker run -d \
#     -p 8443:8443 \
#     -v /path/to/data:/var/lib/cert_service \
#     -v /path/to/config.yaml:/etc/cert_service/config.yaml \
#     -e NVFLARE_CERT_SERVICE_CONFIG=/etc/cert_service/config.yaml \
#     -e NVFLARE_API_KEY=your-api-key \
#     nvflare/cert-service:latest

FROM python:3.10-slim

LABEL maintainer="NVIDIA FLARE Team"
LABEL description="NVIDIA FLARE Certificate Service for token-based enrollment"

# Install minimal system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -s /bin/bash certservice

# Create directories
RUN mkdir -p /var/lib/cert_service /etc/cert_service && \
    chown -R certservice:certservice /var/lib/cert_service /etc/cert_service

# Install Python dependencies (minimal set for cert-service)
COPY docker/cert-service/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Copy only the necessary nvflare modules
COPY nvflare/__init__.py /opt/NVFlare/nvflare/__init__.py
COPY nvflare/_version.py /opt/NVFlare/nvflare/_version.py
COPY nvflare/cert_service /opt/NVFlare/nvflare/cert_service
COPY nvflare/tool/enrollment/token_service.py /opt/NVFlare/nvflare/tool/enrollment/token_service.py
COPY nvflare/tool/enrollment/__init__.py /opt/NVFlare/nvflare/tool/enrollment/__init__.py
COPY nvflare/tool/__init__.py /opt/NVFlare/nvflare/tool/__init__.py
COPY nvflare/lighter/constants.py /opt/NVFlare/nvflare/lighter/constants.py
COPY nvflare/lighter/utils.py /opt/NVFlare/nvflare/lighter/utils.py
COPY nvflare/lighter/__init__.py /opt/NVFlare/nvflare/lighter/__init__.py
COPY nvflare/security/__init__.py /opt/NVFlare/nvflare/security/__init__.py

# Set Python path
ENV PYTHONPATH=/opt/NVFlare

# Environment variables with defaults
ENV NVFLARE_CERT_SERVICE_PORT=8443
ENV NVFLARE_DATA_DIR=/var/lib/cert_service

# Switch to non-root user
USER certservice

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${NVFLARE_CERT_SERVICE_PORT}/health || exit 1

# Expose port
EXPOSE 8443

# Working directory
WORKDIR /opt/NVFlare

# Run with gunicorn for production
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8443", "--access-logfile", "-", "--error-logfile", "-", "nvflare.cert_service.wsgi:app"]

