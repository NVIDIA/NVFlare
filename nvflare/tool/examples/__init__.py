# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

EXAMPLE_CATALOG = {
    "hello-pt": {
        "source_path": "examples/hello-world/hello-pt",
        "required_extra": "PT",
        "setup_commands": (),
        "next_command": ("python", "job.py"),
        "files": (
            "README.md",
            "client.py",
            "hello-pt.ipynb",
            "job.py",
            "model.py",
            "prepare_data.py",
            "requirements.txt",
        ),
    },
    "hello-numpy": {
        "source_path": "examples/hello-world/hello-numpy",
        "required_extra": None,
        "setup_commands": (("python", "-m", "pip", "install", "-r", "requirements.txt"),),
        "next_command": ("python", "job.py"),
        "files": ("README.md", "client.py", "job.py", "requirements.txt"),
    },
    "hello-tf": {
        "source_path": "examples/hello-world/hello-tf",
        "required_extra": None,
        "setup_commands": (("python", "-m", "pip", "install", "-r", "requirements.txt"),),
        "next_command": ("env", "TF_FORCE_GPU_ALLOW_GROWTH=true", "python", "job.py"),
        "files": ("README.md", "client.py", "job.py", "model.py", "requirements.txt"),
    },
    "hello-jax": {
        "source_path": "examples/hello-world/hello-jax",
        "required_extra": None,
        "setup_commands": (
            ("python", "-m", "pip", "install", "-r", "requirements.txt"),
            ("python", "prepare_model.py"),
            ("python", "prepare_data.py"),
        ),
        "next_command": ("python", "job.py"),
        "files": (
            "README.md",
            "client.py",
            "job.py",
            "model.py",
            "prepare_data.py",
            "prepare_model.py",
            "requirements.txt",
        ),
    },
    "hello-lightning": {
        "source_path": "examples/hello-world/hello-lightning",
        "required_extra": "PT",
        "setup_commands": (("python", "-m", "pip", "install", "-r", "requirements.txt"),),
        "next_command": ("python", "job.py", "--synthetic_data"),
        "files": (
            "README.md",
            "client.py",
            "hello_lightning.ipynb",
            "job.py",
            "model.py",
            "prepare_data.py",
            "requirements.txt",
        ),
    },
    "hello-huggingface": {
        "source_path": "examples/hello-world/hello-huggingface",
        "required_extra": "PT",
        "setup_commands": (
            ("python", "-m", "pip", "install", "-r", "requirements.txt"),
            ("python", "prepare_data.py"),
        ),
        "next_command": ("python", "job.py"),
        "files": ("README.md", "client.py", "job.py", "model.py", "prepare_data.py", "requirements.txt"),
    },
    "hello-flower": {
        "source_path": "examples/hello-world/hello-flower",
        "required_extra": "PT",
        "setup_commands": (("python", "-m", "pip", "install", "-r", "requirements.txt"),),
        "next_command": ("python", "job.py", "--job_name", "flwr-pt", "--content_dir", "./flwr-pt"),
        "files": (
            "README.md",
            "flwr-pt/flwr_pt/__init__.py",
            "flwr-pt/flwr_pt/client.py",
            "flwr-pt/flwr_pt/server.py",
            "flwr-pt/flwr_pt/task.py",
            "flwr-pt/pyproject.toml",
            "flwr-pt-tb/flwr_pt_tb/__init__.py",
            "flwr-pt-tb/flwr_pt_tb/client.py",
            "flwr-pt-tb/flwr_pt_tb/server.py",
            "flwr-pt-tb/flwr_pt_tb/task.py",
            "flwr-pt-tb/pyproject.toml",
            "job.py",
            "requirements.txt",
            "train.png",
        ),
    },
}
