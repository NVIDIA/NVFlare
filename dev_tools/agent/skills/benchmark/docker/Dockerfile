ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.19
ARG NODE_IMAGE=node:22.16.0-bookworm-slim
ARG BENCHMARK_DOCKER_AGENT=agent
ARG BENCHMARK_AGENT_HOME=/workspace/agent-home
ARG AGENT_CLI_NAME=agent
ARG AGENT_INSTALL_COMMAND
ARG AGENT_VERSION_COMMAND
ARG SDK_PACKAGE_NAME=sdk
ARG SDK_IMPORT_NAME
ARG SDK_VERSION_COMMAND
ARG SKILLS_SETUP_TYPE=command
ARG SKILLS_INSTALL_COMMAND
ARG SKILLS_LIST_COMMAND
ARG SKILLS_INSTALL_OUTPUT=skills_build_install.json
ARG SKILLS_LIST_OUTPUT=skills_list.json
ARG SKILLS_INSTALL_EXPECTED_SOURCE=local_sdk_wheel

FROM ${UV_IMAGE} AS uv_source

FROM ${NODE_IMAGE} AS base
ARG UV_IMAGE
ARG NODE_IMAGE
ARG BENCHMARK_DOCKER_AGENT
ARG BENCHMARK_AGENT_HOME
ARG AGENT_CLI_NAME
ARG AGENT_INSTALL_COMMAND
ARG AGENT_VERSION_COMMAND
ARG SDK_PACKAGE_NAME
ARG SDK_IMPORT_NAME
ARG SDK_VERSION_COMMAND
ARG SKILLS_SETUP_TYPE
ARG SKILLS_INSTALL_COMMAND
ARG SKILLS_LIST_COMMAND
ARG SKILLS_INSTALL_OUTPUT
ARG SKILLS_LIST_OUTPUT
ARG SKILLS_INSTALL_EXPECTED_SOURCE

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    bubblewrap \
    ca-certificates \
    git \
    python3 \
    findutils \
    coreutils \
    socat \
    tree \
    && rm -rf /var/lib/apt/lists/*

RUN if [ -z "${AGENT_INSTALL_COMMAND}" ]; then \
        echo "AGENT_INSTALL_COMMAND is required by the agent profile" >&2; \
        exit 2; \
    fi && \
    /bin/bash -lc "${AGENT_INSTALL_COMMAND}" && \
    if [ -n "${AGENT_VERSION_COMMAND}" ]; then /bin/bash -lc "${AGENT_VERSION_COMMAND}"; fi

COPY --from=uv_source /uv /uvx /usr/local/bin/
RUN uv --version

ENV BENCHMARK_DOCKER_AGENT=${BENCHMARK_DOCKER_AGENT}
ENV BENCHMARK_AGENT_HOME=${BENCHMARK_AGENT_HOME}
ENV AGENT_CLI_NAME="${AGENT_CLI_NAME}"
ENV AGENT_INSTALL_COMMAND="${AGENT_INSTALL_COMMAND}"
ENV AGENT_VERSION_COMMAND="${AGENT_VERSION_COMMAND}"
ENV SDK_PACKAGE_NAME="${SDK_PACKAGE_NAME}"
ENV SDK_IMPORT_NAME="${SDK_IMPORT_NAME}"
ENV SDK_VERSION_COMMAND="${SDK_VERSION_COMMAND}"
ENV SKILLS_SETUP_TYPE="${SKILLS_SETUP_TYPE}"
ENV SKILLS_INSTALL_COMMAND="${SKILLS_INSTALL_COMMAND}"
ENV SKILLS_LIST_COMMAND="${SKILLS_LIST_COMMAND}"
ENV SKILLS_INSTALL_OUTPUT="${SKILLS_INSTALL_OUTPUT}"
ENV SKILLS_LIST_OUTPUT="${SKILLS_LIST_OUTPUT}"
ENV SKILLS_INSTALL_EXPECTED_SOURCE="${SKILLS_INSTALL_EXPECTED_SOURCE}"
ENV JOB_INPUT_DIR=/workspace/input
ENV TRAINING_CODE=/workspace/input
ENV RECORDS_DIR=/tmp/agent_benchmark/records

WORKDIR /workspace
RUN mkdir -p "${BENCHMARK_AGENT_HOME}" /workspace/prompts && \
    uv venv --seed /workspace/venv

ENV VIRTUAL_ENV=/workspace/venv
ENV PATH="/workspace/venv/bin:/usr/local/bin:${PATH}"

# Agent tool commands may use `bash -lc`; Debian /etc/profile resets PATH for login shells.
RUN printf '%s\n' \
    'export VIRTUAL_ENV=/workspace/venv' \
    'case ":$PATH:" in *:/workspace/venv/bin:*) ;; *) export PATH="/workspace/venv/bin:$PATH" ;; esac' \
    > /etc/profile.d/99-agent-benchmark-venv.sh && \
    /bin/bash -lc 'command -v python && python --version && python -m pip --version && pip --version'

RUN python --version && \
    python -m pip --version && \
    pip --version && \
    tree --version && \
    python -c "import json, os, pathlib, subprocess; version_cmd=os.environ.get('AGENT_VERSION_COMMAND') or ''; version_output=subprocess.check_output(['/bin/bash', '-lc', version_cmd], text=True).strip() if version_cmd else ''; home=pathlib.Path(os.environ.get('BENCHMARK_AGENT_HOME', '/workspace/agent-home')); home.mkdir(parents=True, exist_ok=True); home.joinpath('build_metadata.json').write_text(json.dumps({'agent': os.environ.get('BENCHMARK_DOCKER_AGENT', 'agent'), 'agent_cli': os.environ.get('AGENT_CLI_NAME', 'agent'), 'agent_cli_install_command': os.environ.get('AGENT_INSTALL_COMMAND'), 'agent_cli_version_command': version_cmd, 'agent_cli_version_output': version_output, 'apt_package_policy': 'Debian apt packages are intentionally not exact-version pinned; rebuild comparability is tracked through the pinned base image/build args and recorded package/tool versions.', 'node_image': os.environ.get('NODE_IMAGE'), 'node_version': subprocess.check_output(['node', '--version'], text=True).strip(), 'sdk_import_name': os.environ.get('SDK_IMPORT_NAME'), 'sdk_package_name': os.environ.get('SDK_PACKAGE_NAME'), 'skills_install_expected_source': os.environ.get('SKILLS_INSTALL_EXPECTED_SOURCE'), 'skills_install_network_proof': 'not_enforced_by_dockerfile', 'tree_version': subprocess.check_output(['tree', '--version'], text=True).splitlines()[0], 'uv_image': os.environ.get('UV_IMAGE'), 'uv_version': subprocess.check_output(['uv', '--version'], text=True).strip(), 'virtual_env': os.environ.get('VIRTUAL_ENV')}, indent=2, sort_keys=True) + '\n')"

COPY config/ /workspace/config/
COPY skills/ /workspace/skills/

# Bake agent settings before switching to non-root. Claude Code merges settings.json
# from CLAUDE_CONFIG_DIR; placing an explicit allow list here ensures Bash and other
# required tools are approved even when launched non-interactively (--print mode).
RUN if [ -f "/workspace/config/agents/${BENCHMARK_DOCKER_AGENT}-settings.json" ]; then \
        cp "/workspace/config/agents/${BENCHMARK_DOCKER_AGENT}-settings.json" "${BENCHMARK_AGENT_HOME}/settings.json"; \
    fi

# Claude Code refuses --dangerously-skip-permissions when running as root.
# Create a non-root user that owns the workspace so the agent can run normally.
RUN useradd -m -u 9999 -s /bin/bash agent && \
    chown -R agent:agent /workspace
USER agent

CMD ["bash"]

FROM base AS skills

# Install the configured SDK from the staged local wheel that packages skills.
COPY --chown=agent:agent dist/skills/*.whl dist/skills/sdk_wheel_metadata.json /tmp/wheels/
RUN python -c "import os, sys; from pathlib import Path; package=os.environ.get('SDK_PACKAGE_NAME', 'SDK'); wheels=sorted(Path('/tmp/wheels').glob('*.whl')); len(wheels) == 1 or sys.exit(f'Expected exactly one staged {package} skills wheel in /tmp/wheels, found {len(wheels)}: {[p.name for p in wheels]}'); Path('/tmp/selected_sdk_wheel').write_text(str(wheels[0]))" && \
    wheel="$(cat /tmp/selected_sdk_wheel)" && \
    uv pip install --python /workspace/venv/bin/python "${wheel}" && \
    cp /tmp/wheels/sdk_wheel_metadata.json "${BENCHMARK_AGENT_HOME}/sdk_wheel_metadata.json" && \
    rm -rf /tmp/wheels /tmp/selected_sdk_wheel && \
    if [ -n "${SDK_VERSION_COMMAND}" ]; then /bin/bash -lc "${SDK_VERSION_COMMAND}"; fi && \
    python -c "import importlib, os; name=os.environ.get('SDK_IMPORT_NAME'); importlib.import_module(name) if name else None" && \
    /bin/bash -lc 'command -v python && python -m pip --version'

# Preinstall skills so benchmark runs measure agent behavior, not skill setup time.
COPY --chown=agent:agent sdk_skills/ /tmp/sdk_skills/
RUN python -m skills.harness.container.sdk_skills_setup

FROM base AS baseline

# Baseline image: install the configured SDK from the staged local wheel built without packaged skills.
COPY --chown=agent:agent dist/baseline/*.whl dist/baseline/sdk_wheel_metadata.json /tmp/wheels/
RUN python -c "import os, sys; from pathlib import Path; package=os.environ.get('SDK_PACKAGE_NAME', 'SDK'); wheels=sorted(Path('/tmp/wheels').glob('*.whl')); len(wheels) == 1 or sys.exit(f'Expected exactly one staged {package} baseline wheel in /tmp/wheels, found {len(wheels)}: {[p.name for p in wheels]}'); Path('/tmp/selected_sdk_wheel').write_text(str(wheels[0]))" && \
    wheel="$(cat /tmp/selected_sdk_wheel)" && \
    uv pip install --python /workspace/venv/bin/python "${wheel}" && \
    cp /tmp/wheels/sdk_wheel_metadata.json "${BENCHMARK_AGENT_HOME}/sdk_wheel_metadata.json" && \
    rm -rf /tmp/wheels /tmp/selected_sdk_wheel && \
    if [ -n "${SDK_VERSION_COMMAND}" ]; then /bin/bash -lc "${SDK_VERSION_COMMAND}"; fi && \
    python -c "import importlib, os; name=os.environ.get('SDK_IMPORT_NAME'); importlib.import_module(name) if name else None" && \
    /bin/bash -lc 'command -v python && python -m pip --version' && \
    python -c "import json, os, pathlib; root=pathlib.Path(os.environ['BENCHMARK_AGENT_HOME']); agent=os.environ['BENCHMARK_DOCKER_AGENT']; reason=f'local baseline wheel does not package {agent} skills'; (root / os.environ['SKILLS_INSTALL_OUTPUT']).write_text(json.dumps({'status': 'skipped', 'agent': agent, 'reason': reason}, indent=2, sort_keys=True) + '\n'); (root / os.environ['SKILLS_LIST_OUTPUT']).write_text(json.dumps({'status': 'skipped', 'agent': agent, 'installed': [], 'reason': reason}, indent=2, sort_keys=True) + '\n')"
