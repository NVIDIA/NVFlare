# NVIDIA FLARE Certificate Service Docker

This directory contains Docker configuration for the NVIDIA FLARE Certificate Service.

## Files

| File | Description |
|------|-------------|
| `Dockerfile` | Multi-stage build for minimal cert-service image |
| `requirements.txt` | Python dependencies (subset of full nvflare) |
| `docker-compose.yml` | Easy deployment configuration |

## Quick Start

```bash
# From project root
cd /path/to/NVFlare

# 1. Generate an API key
export NVFLARE_API_KEY=$(nvflare cert api-key)

# 2. Build the image
docker build -f docker/cert-service/Dockerfile -t nvflare/cert-service:latest .

# 3. Run with docker-compose
cd docker/cert-service
docker-compose up -d

# 4. Check status
docker-compose logs -f
curl http://localhost:8443/health
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NVFLARE_API_KEY` | (required) | API key for admin operations |
| `NVFLARE_CERT_SERVICE_PORT` | `8443` | Service port |
| `NVFLARE_DATA_DIR` | `/var/lib/cert_service` | Data directory |
| `NVFLARE_CERT_SERVICE_CONFIG` | (none) | Path to custom config file |

### Persistent Data

Mount `/var/lib/cert_service` to persist:
- Root CA certificate and key (`rootCA.pem`, `rootCA.key`)
- SQLite database (`enrollment.db`)

## Production Deployment

For production, place behind a reverse proxy (nginx, traefik) that handles TLS termination:

```yaml
# docker-compose.yml
services:
  cert-service:
    image: nvflare/cert-service:latest
    expose:
      - "8443"
    # ... rest of config
    
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - /etc/letsencrypt:/etc/letsencrypt:ro
```

## Scaling

For high concurrency (100+ simultaneous enrollments):

```bash
# Use more gunicorn workers
docker run -e GUNICORN_WORKERS=4 ...

# Or use PostgreSQL backend
docker run -e NVFLARE_STORAGE_TYPE=postgresql \
           -e NVFLARE_DB_URL=postgresql://user:pass@db:5432/certservice ...
```

