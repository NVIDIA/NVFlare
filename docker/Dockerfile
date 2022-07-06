FROM python:3.8
COPY nvflare /opt/NVFlare/nvflare
COPY setup.py /opt/NVFlare/setup.py
COPY versioneer.py /opt/NVFlare/versioneer.py
COPY README.md /opt/NVFlare/README.md
COPY setup.cfg /opt/NVFlare/setup.cfg
WORKDIR /opt/NVFlare
RUN pip install -U pip
RUN pip install -e .
RUN pip install Flask-sqlalchemy Flask-JWT-extended
