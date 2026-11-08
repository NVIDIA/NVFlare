FROM python:3.11-slim
ARG NVFL_BASE_VERSION=2.10.0
WORKDIR /opt/NVFlare
COPY setup.py setup.cfg pyproject.toml versioneer.py README.md LICENSE MANIFEST.in ./
COPY 3rdParty/sqlalchemy.LICENSE.txt 3rdParty/pyarrow.LICENSE.txt 3rdParty/pyarrow.NOTICE.txt ./3rdParty/
COPY nvflare ./nvflare
RUN pip install --no-cache-dir ".[catalog]" "psycopg[binary]>=3,<4"
COPY examples/advanced/federated-catalog-services /opt/example
WORKDIR /opt/example
RUN python -c 'import requests; from pathlib import Path; from common import PIN; p=Path("data/playground"); p.mkdir(parents=True,exist_ok=True); [(lambda r,n: (r.raise_for_status(),(p/n).write_bytes(r.content)))(requests.get(f"https://raw.githubusercontent.com/open-lakehouse/unitycatalog-playground/{PIN}/{s}",timeout=30),n) for n,s in [("sanctuary.py","marimo-playground/notebooks/sanctuary.py"),("LICENSE","LICENSE")]]'
ENV PYTHONUNBUFFERED=1
CMD ["bash", "/startup-kit/startup/sub_start.sh"]
