# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NVIDIA FLARE (NVFlare) is a domain-agnostic, open-source Python SDK for federated learning. It allows ML/DL workflows to be adapted to a federated paradigm, enabling secure, privacy-preserving distributed multi-party collaboration.

**Key capabilities:** PyTorch/TensorFlow/Scikit-learn/XGBoost support, horizontal and vertical FL, built-in FL algorithms (FedAvg, FedProx, FedOpt, Scaffold, Ditto), privacy preservation (differential privacy, homomorphic encryption, PSI).

## Common Commands

### Development Setup
```bash
# Install development dependencies (macOS)
python3 -m pip install -e .[dev_mac]

# Install development dependencies (Linux)
python3 -m pip install -e .[dev]
```

### Testing
```bash
# Run all checks (license, style, tests with coverage)
./runtest.sh

# Run unit tests only
./runtest.sh -u

# Run unit tests with coverage report
./runtest.sh -u -c

# Run a specific test file
python3 -m pytest tests/unit_test/path/to/test_file.py -v

# Run a specific test
python3 -m pytest tests/unit_test/path/to/test_file.py::TestClass::test_method -v

# Run tests in parallel (8 processes)
python3 -m pytest --numprocesses=8 -v tests/unit_test
```

### Code Style
```bash
# Check code style (flake8, black, isort)
./runtest.sh -s

# Auto-fix code style
./runtest.sh -f

# Check license headers
./runtest.sh -l
```

### Documentation
```bash
# Build documentation
./build_doc.sh --html

# Clean documentation
./build_doc.sh --clean
```

### Clean Build Artifacts
```bash
./runtest.sh --clean
```

## Architecture Overview

### Core Concepts

NVFlare uses a **server-client federated architecture**:
- **Server**: Runs a **Controller** that schedules **Tasks** and aggregates results
- **Clients**: Run **Executors** that execute tasks on local data
- **Shareable**: The dict-based data structure for communication between server and clients
- **FLContext**: Context object passing data between FL components (supports private/public and sticky/non-sticky properties)

### Key Components (nvflare/apis/)

| Component | Purpose |
|-----------|---------|
| `Controller` | Server-side workflow orchestrator that schedules tasks to clients |
| `Executor` | Client-side task processor that runs on federated client nodes |
| `Shareable` | Data container for server-client communication |
| `FLContext` | Context for passing data between FL components |
| `Task` | Work unit assigned by Controller to client workers |
| `Filter` | Transforms Shareable data in/out of components |
| `FLComponent` | Base class for all FL components |

### Controller Task Distribution Methods
- `broadcast()` / `broadcast_and_wait()`: Send task to multiple clients
- `send()` / `send_and_wait()`: Send task to a single client
- `relay()` / `relay_and_wait()`: Sequential task execution across clients

### Package Structure

```
nvflare/
├── apis/           # Core API specifications and interfaces
├── app_common/     # Common application utilities and FL algorithms
├── app_opt/        # Optional dependencies (HE, PSI, ML frameworks)
├── client/         # Client-side FLARE Client API
├── job_config/     # Job definition and configuration (FedJob API)
├── private/        # Internal implementations
├── fuel/           # Core infrastructure utilities
├── tool/           # CLI tools
├── dashboard/      # Web dashboard
└── lighter/        # Provisioning tools
```

### Running Modes
- **Simulator**: Local development (`nvflare simulator`)
- **POC**: Proof-of-concept deployment
- **Production**: Full distributed deployment with provisioning

## Code Style Requirements

- **Formatter**: black (line length: 120)
- **Linter**: flake8
- **Import sorter**: isort (black profile)
- **Python**: 3.9, 3.10, 3.11, 3.12

All Python files must include Apache 2.0 license header:
```python
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
```

## Test Conventions

- Test files: `tests/unit_test/` following `[module_name]_test.py` pattern
- Integration tests: `tests/integration_test/`
- Framework: pytest with pytest-xdist (parallel) and pytest-cov (coverage)

## CLI Entry Point

The `nvflare` CLI is the main entry point:
```bash
nvflare --help
nvflare simulator --help
```

## Personal Configuration

Create a `CLAUDE.local.md` file (git-ignored) for personal preferences such as:
- Preferred coding patterns
- Local environment specifics
- Custom shortcuts or workflows

