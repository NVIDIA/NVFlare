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

"""Run the Wilson-based MemCo workflow with NVFlare's released Collab API."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from nvflare.collab import CollabRecipe, collab
from nvflare.recipe import SimEnv

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memco.core import aggregate_local_payloads  # noqa: E402

CANONICAL_DOMAINS = {
    "pddl": ["gripper", "blockworld", "barman", "tyreworld"],
    "fever": ["A_film_tv", "B_music"],
    "alfworld": ["bathroom", "bedroom", "kitchen", "living"],
}
PDDL_TRAIN_FILES = {
    "gripper": "pddl_domain_gripper.jsonl",
    "blockworld": "pddl_domain_blockworld.jsonl",
    "barman": "pddl_domain_barman.jsonl",
    "tyreworld": "pddl_domain_tyreworld.jsonl",
}
FEVER_TRAIN_FILES = {
    "A_film_tv": "fever_ab_train_A_v3.jsonl",
    "B_music": "fever_ab_train_B_v3.jsonl",
}


def _empty_global_memory() -> dict:
    return {
        "promoted_batches": [],
        "candidate_count": 0,
        "rule_count": 0,
        "artifact_count": 0,
        "candidates": [],
        "rules": [],
        "artifacts": [],
    }


def _site_index(site_name: str) -> int:
    prefix, separator, suffix = site_name.rpartition("-")
    if not separator or prefix != "site" or not suffix.isdigit():
        raise ValueError(f"Expected Collab client name site-N, got {site_name!r}")
    return int(suffix) - 1


def _domain_for_site(config: dict, site_name: str) -> str:
    domains = config["domains"]
    index = _site_index(site_name)
    if index < 0 or index >= len(domains):
        raise ValueError(f"No MemCo domain configured for {site_name}")
    return domains[index]


def _data_path(config: dict, domain: str, mode: str) -> Path:
    data_root = Path(config["data_root"])
    family = config["dataset_family"]
    if family == "pddl":
        name = PDDL_TRAIN_FILES[domain] if mode == "train" else "test.jsonl"
        return data_root / "data" / "pddl" / name
    if family == "fever":
        name = FEVER_TRAIN_FILES[domain] if mode == "train" else "fever_ab_test_v3.jsonl"
        return data_root / "data" / "fever" / name
    return data_root / "data" / "alfworld" / "collab_subsets" / "v3_s"


def _worker_command(config: dict, domain: str, mode: str, work_dir: Path) -> list[str]:
    family = config["dataset_family"]
    output_root = work_dir / config.get("output_root", f"artifacts/{family}") / domain
    worker_path = PROJECT_ROOT / "scripts" / "run_domain.py"
    if not worker_path.is_file():
        worker_path = PROJECT_ROOT / "run_domain.py"
    if not worker_path.is_file():
        raise FileNotFoundError(f"MemCo worker run_domain.py is missing from {PROJECT_ROOT}")
    command = [
        sys.executable,
        str(worker_path),
        mode,
        "--family",
        family,
        "--domain",
        domain,
        "--data",
        str(_data_path(config, domain, mode)),
        "--memory-dir",
        str(output_root / "memory"),
        "--shared-global-dir",
        str(output_root / "shared_global"),
        "--local-memory-path",
        str(output_root / "local_memory.json"),
        "--metrics-path",
        str(output_root / f"{mode}_metrics.json"),
        "--log-dir",
        str(output_root / f"{mode}_logs"),
        "--memory-settings",
        "local_plus_global",
        "--promotion-threshold",
        str(config.get("promotion_threshold", 0.35)),
        "--promotion-policy",
        config.get("promotion_policy", "wilson"),
        "--wilson-alpha",
        str(config.get("wilson_alpha", 0.05)),
        "--wilson-threshold",
        str(config.get("wilson_threshold", 0.35)),
        "--model",
        config.get("model", "gpt-4o-mini"),
        "--max-tasks",
        str(config.get(f"max_{mode}_tasks", 0)),
        "--max-trials",
        str(config.get("max_trials", 30)),
    ]
    if family == "alfworld":
        command.extend(
            [
                "--alfworld-game-root",
                config["alfworld_game_root"],
                "--alfworld-eval-split",
                "valid_seen,valid_unseen",
                "--alfworld-domains",
                ",".join(config["domains"]),
            ]
        )
    elif family == "fever":
        command.extend(["--fever-wiki-cache", str(Path(config["data_root"]) / ".cache" / "fever_wiki_cache.json")])
    return command


def _paths(config: dict, domain: str, work_dir: Path) -> dict[str, Path]:
    root = work_dir / config.get("output_root", f"artifacts/{config['dataset_family']}") / domain
    return {
        "local": root / "local_memory.json",
        "global": root / "shared_global" / "memco" / "global_memory.json",
        "metrics": root / "evaluate_metrics.json",
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _work_dir() -> Path:
    return Path(collab.workspace.get_run_dir(collab.fl_ctx.get_job_id()))


def _run_worker(config: dict, domain: str, mode: str, global_memory: dict) -> dict:
    work_dir = _work_dir()
    paths = _paths(config, domain, work_dir)
    _write_json(paths["global"], global_memory)
    command = _worker_command(config, domain, mode, work_dir)
    print(f"[{collab.site_name}] {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=work_dir, env=os.environ.copy(), check=True)
    result = {"site_name": collab.site_name, "domain": domain}
    if mode == "train":
        result["local_memory"] = json.loads(paths["local"].read_text(encoding="utf-8"))
        result["global_memory"] = json.loads(paths["global"].read_text(encoding="utf-8"))
    else:
        result["metrics"] = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    return result


class MemCoClient:
    """Run one domain worker on each NVFlare client site."""

    def __init__(self, config: dict):
        self.config = config

    @collab.publish
    def train(self, global_memory: dict) -> dict:
        """Train this site's domain and return local and cumulative global memory."""
        domain = _domain_for_site(self.config, collab.site_name)
        return _run_worker(self.config, domain, "train", global_memory)

    @collab.publish
    def evaluate(self, final_global_memory: dict) -> dict:
        """Evaluate local memory with the frozen final global memory."""
        domain = _domain_for_site(self.config, collab.site_name)
        return _run_worker(self.config, domain, "evaluate", final_global_memory)


def _average_metrics(metrics_by_client: dict[str, dict]) -> dict:
    rows = list(metrics_by_client.values())
    completed = sum(int(row.get("num_completed", 0)) for row in rows)
    success = sum(int(row.get("num_success", 0)) for row in rows)
    weighted_reward = sum(float(row.get("avg_reward", 0.0)) * int(row.get("num_completed", 0)) for row in rows)
    return {
        "accuracy": success / completed if completed else 0.0,
        "avg_reward": weighted_reward / completed if completed else 0.0,
        "num_clients": len(rows),
        "num_completed": completed,
        "num_success": success,
        "num_tasks": sum(int(row.get("num_tasks", 0)) for row in rows),
    }


class MemCoServer:
    """Coordinate sequential memory construction and parallel evaluation."""

    def __init__(self, config: dict):
        self.config = config

    @collab.main
    def run(self) -> dict:
        clients = sorted(collab.clients, key=lambda client: _site_index(client.name))
        global_memory = _empty_global_memory()
        local_memories = []
        timeout = float(self.config.get("call_timeout", 86400))

        for client in clients:
            result = client(timeout=timeout).train(global_memory)
            if isinstance(result, Exception):
                raise result
            local_memories.append(result["local_memory"])
            global_memory = result["global_memory"]

        final_global = aggregate_local_payloads(
            local_memories,
            promotion_threshold=float(self.config.get("promotion_threshold", 0.35)),
            promotion_policy=self.config.get("promotion_policy", "wilson"),
            wilson_alpha=float(self.config.get("wilson_alpha", 0.05)),
            wilson_threshold=float(self.config.get("wilson_threshold", 0.35)),
            batch_name="collab_dynamic_train",
        )
        server_dir = _work_dir() / self.config.get(
            "server_output_directory", f"artifacts/{self.config['dataset_family']}/server"
        )
        _write_json(server_dir / "global_memory.json", final_global)

        call_results = collab.clients(timeout=timeout, parallel=len(clients)).evaluate(final_global)
        if call_results.failures:
            failures = ", ".join(f"{site}: {error}" for site, error in call_results.failures.items())
            raise RuntimeError(f"MemCo evaluation failed: {failures}")
        metrics_by_client = {site_name: result["metrics"] for site_name, result in call_results}
        metrics_by_client["average"] = _average_metrics(metrics_by_client)
        _write_json(server_dir / "metrics_by_client.json", metrics_by_client)
        return metrics_by_client


def _load_config(path: Path, data_root: Path | None, alfworld_game_root: Path | None) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    family = config.get("dataset_family")
    if family not in CANONICAL_DOMAINS:
        raise ValueError("dataset_family must be pddl, fever, or alfworld")
    if data_root:
        config["data_root"] = str(data_root.expanduser().resolve())
    else:
        config["data_root"] = str(Path(config["data_root"]).expanduser().resolve())
    if alfworld_game_root:
        config["alfworld_game_root"] = str(alfworld_game_root.expanduser().resolve())
    elif family == "alfworld":
        config["alfworld_game_root"] = str(Path(config["alfworld_game_root"]).expanduser().resolve())
    domains = config.get("domains", CANONICAL_DOMAINS[family])
    if (
        not isinstance(domains, list)
        or not domains
        or any(domain not in CANONICAL_DOMAINS[family] for domain in domains)
        or len(set(domains)) != len(domains)
    ):
        raise ValueError(f"domains must be a nonempty, unique subset of {CANONICAL_DOMAINS[family]}")
    config["domains"] = list(domains)
    config["model"] = os.environ.get("MEMCO_LLM_MODEL") or config.get("model", "gpt-4o-mini")
    policy = str(config.get("promotion_policy", "wilson")).strip().lower()
    if policy not in {"legacy", "shadow", "wilson"}:
        raise ValueError("promotion_policy must be legacy, shadow, or wilson")
    config["promotion_policy"] = policy
    alpha = float(config.get("wilson_alpha", 0.05))
    threshold = float(config.get("wilson_threshold", 0.35))
    if not 0.0 < alpha < 0.5:
        raise ValueError("wilson_alpha must lie strictly between 0 and 0.5")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("wilson_threshold must lie between 0 and 1")
    config["wilson_alpha"] = alpha
    config["wilson_threshold"] = threshold
    return config


def make_recipe(config: dict) -> CollabRecipe:
    """Build a released Collab API recipe and bundle MemCo runtime dependencies."""
    recipe = CollabRecipe(
        job_name=config.get("job_name", "memco-collab"),
        server=MemCoServer(config),
        client=MemCoClient(config),
        min_clients=len(config["domains"]),
        sync_task_timeout=float(config.get("call_timeout", 86400)),
    )
    recipe.add_server_file(str(PROJECT_ROOT / "src"))
    recipe.add_client_file(str(PROJECT_ROOT / "src"))
    recipe.add_client_file(str(PROJECT_ROOT / "runtime"))
    recipe.add_client_file(str(PROJECT_ROOT / "scripts"))
    return recipe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path("/tmp/nvflare/collab_simulation"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--alfworld-game-root", type=Path)
    args = parser.parse_args()

    config = _load_config(args.config, args.data_root, args.alfworld_game_root)
    recipe = make_recipe(config)
    environment = SimEnv(
        num_clients=len(config["domains"]),
        workspace_root=str(args.workspace.expanduser().resolve()),
    )
    run = recipe.execute(environment)
    print(f"Results at: {run.get_result()}")


if __name__ == "__main__":
    main()
