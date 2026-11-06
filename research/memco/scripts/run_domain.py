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

"""Run one MemCo client domain with the migrated legacy execution semantics."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RUNTIME_ROOT = PROJECT_ROOT / "runtime"
SOURCE_ROOT = PROJECT_ROOT / "src"
if not RUNTIME_ROOT.exists():
    if (PROJECT_ROOT / "mas").is_dir() and (PROJECT_ROOT / "tasks").is_dir():
        RUNTIME_ROOT = PROJECT_ROOT
        SOURCE_ROOT = PROJECT_ROOT
    else:
        PROJECT_ROOT = SCRIPT_DIR
        RUNTIME_ROOT = SCRIPT_DIR
        SOURCE_ROOT = SCRIPT_DIR
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(RUNTIME_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

from mas.llm import GPTChat  # noqa: E402
from mas.memory.mas_memory.memco import MemCoMASMemory  # noqa: E402
from mas.reasoning import ReasoningIO  # noqa: E402
from mas.utils import EmbeddingFunc  # noqa: E402
from memco.domains import (  # noqa: E402
    build_fever_task,
    load_jsonl_rows,
    load_subset_file,
    normalize_pddl_rows,
    summarize_task_results,
)
from tasks.envs import AlfworldEnv, AlfworldRecorder, FeverEnv, FeverRecorder, PDDLEnv, PDDLRecorder  # noqa: E402
from tasks.mas_workflow import AutoGen  # noqa: E402
from tasks.prompts import (  # noqa: E402
    alfworld_few_shots,
    alfworld_solver_system_prompt,
    fever_few_shots,
    fever_solver_system_prompt,
    pddl_prompts,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)


def _limit_tasks(tasks: list[dict], maximum: int) -> list[dict]:
    return tasks[:maximum] if maximum > 0 else tasks


def _dedupe_tasks(tasks: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for task in tasks:
        gamefile = str((task.get("env_kwargs") or {}).get("gamefile", "")).strip()
        key = gamefile or json.dumps(task, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(task)
    return unique


def _assert_replay_consumed(llm: GPTChat) -> None:
    replay_tape = getattr(llm, "replay_tape", None)
    if replay_tape:
        replay_tape.assert_consumed()


def _run_fever(args: argparse.Namespace) -> None:
    rows = [build_fever_task(row) for row in load_jsonl_rows(args.data)]
    tasks = [row for row in rows if row]
    if args.mode == "train":
        tasks = [row for row in tasks if not args.domain or row.get("ab_domain") == args.domain]
    else:
        tasks = _dedupe_tasks(tasks)
    tasks = _limit_tasks(tasks, args.max_tasks)

    env = FeverEnv(env_config={}, max_trials=args.max_trials)
    llm = GPTChat(model_name=args.model)
    memory = MemCoMASMemory(
        namespace="memco",
        global_config={
            "working_dir": str(args.memory_dir),
            "viz_output_dir": str(args.log_dir),
            "task_name": "fever",
            "memco_dynamic_graph": True,
            "memco_router": "textloss",
            "memco_owner_scene": args.domain,
            "memco_settings": args.memory_settings,
            "memco_freeze_memory": args.mode == "evaluate",
            "memco_shared_global_dir": str(args.shared_global_dir),
            "memco_promotion_threshold": args.promotion_threshold,
            "memco_promotion_policy": args.promotion_policy,
            "memco_wilson_alpha": args.wilson_alpha,
            "memco_wilson_threshold": args.wilson_threshold,
        },
        llm_model=llm,
        embedding_func=EmbeddingFunc("sentence-transformers/all-MiniLM-L6-v2"),
    )
    workflow = AutoGen()
    recorder = FeverRecorder(working_dir=str(args.log_dir), namespace="total_task")
    recorder.set_quiet_console(True)
    workflow.add_observer(recorder)
    workflow.build_system(
        ReasoningIO(llm_model=llm), memory, env, {"autogen": True, "silent_mas": True, "insights_topk": 1}
    )

    task_rows = []
    recorder.dataset_begin()
    for task_id, task in enumerate(tasks):
        recorder.task_begin(task_id, task)
        task_main, task_description = env.set_env(task)
        task.update(
            task_main=task_main,
            task_description=task_description,
            few_shots=fever_few_shots[:3],
            env_ref=env,
            task_index=task_id,
            task_index_1based=task_id + 1,
            mas_memory_type="memco",
            model_type="",
        )
        for agent in workflow.agents_team.values():
            recorder.log(f"------------ MAS Agent: {agent.name} ------------")
            recorder.log(agent.add_task_instruction(fever_solver_system_prompt))
        reward, done = workflow.schedule(task)
        recorder.task_end(reward, done)
        task_rows.append({"task": task["task"], "reward": float(reward), "done": bool(done)})
    recorder.dataset_end()

    persist = getattr(memory, "persist_entity_graph", None)
    if callable(persist):
        persist()
    _assert_replay_consumed(llm)
    metrics = summarize_task_results(task_rows)
    _write_json(args.metrics_path, metrics)

    if args.mode == "train":
        candidates = sorted((args.memory_dir / "memco").glob("local_*.json"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one local memory artifact, found: {candidates}")
        args.local_memory_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[0], args.local_memory_path)


def _run_pddl(args: argparse.Namespace) -> None:
    tasks = normalize_pddl_rows(load_jsonl_rows(args.data))
    if args.mode == "train":
        tasks = [row for row in tasks if row.get("game_name") == args.domain]
    else:
        tasks = _dedupe_tasks(tasks)
    tasks = _limit_tasks(tasks, args.max_tasks)

    env = PDDLEnv(env_config={}, max_trials=args.max_trials)
    llm = GPTChat(model_name=args.model)
    memory = MemCoMASMemory(
        namespace="memco",
        global_config={
            "working_dir": str(args.memory_dir),
            "viz_output_dir": str(args.log_dir),
            "task_name": "pddl",
            "memco_dynamic_graph": True,
            "memco_router": "textloss",
            "memco_owner_scene": args.domain,
            "memco_settings": args.memory_settings,
            "memco_freeze_memory": args.mode == "evaluate",
            "memco_shared_global_dir": str(args.shared_global_dir),
            "memco_promotion_threshold": args.promotion_threshold,
            "memco_promotion_policy": args.promotion_policy,
            "memco_wilson_alpha": args.wilson_alpha,
            "memco_wilson_threshold": args.wilson_threshold,
        },
        llm_model=llm,
        embedding_func=EmbeddingFunc("sentence-transformers/all-MiniLM-L6-v2"),
    )
    workflow = AutoGen()
    recorder = PDDLRecorder(working_dir=str(args.log_dir), namespace="total_task")
    recorder.set_quiet_console(True)
    workflow.add_observer(recorder)
    workflow.build_system(
        ReasoningIO(llm_model=llm), memory, env, {"autogen": True, "silent_mas": True, "insights_topk": 1}
    )

    task_rows = []
    recorder.dataset_begin()
    for task_id, task in enumerate(tasks):
        recorder.task_begin(task_id, task)
        task_main, task_description = env.set_env(task)
        prompt = pddl_prompts[task["game_name"]]
        task.update(
            task_main=task_main,
            task_description=task_description,
            few_shots=prompt["examples"][:1],
            env_ref=env,
            task_index=task_id,
            task_index_1based=task_id + 1,
            mas_memory_type="memco",
            model_type="",
        )
        for agent in workflow.agents_team.values():
            recorder.log(f"------------ MAS Agent: {agent.name} ------------")
            recorder.log(agent.add_task_instruction(prompt["instruction"]))
        reward, done = workflow.schedule(task)
        recorder.task_end(reward, done)
        task_rows.append(
            {
                "game_name": task["game_name"],
                "problem_index": task["problem_index"],
                "reward": float(reward),
                "done": bool(done),
            }
        )
    recorder.dataset_end()

    persist = getattr(memory, "persist_entity_graph", None)
    if callable(persist):
        persist()
    _assert_replay_consumed(llm)
    metrics = summarize_task_results(task_rows)
    _write_json(args.metrics_path, metrics)

    if args.mode == "train":
        candidates = sorted((args.memory_dir / "memco").glob("local_*.json"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one local memory artifact, found: {candidates}")
        args.local_memory_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[0], args.local_memory_path)


def _run_alfworld(args: argparse.Namespace) -> None:
    split_names = ["train"] if args.mode == "train" else args.alfworld_eval_split.split(",")
    tasks = []
    for split_name in split_names:
        domains = [args.domain] if args.mode == "train" else args.alfworld_domains.split(",")
        for domain in domains:
            tasks.extend(load_subset_file(args.data, domain.strip(), split_name.strip()))
    if args.mode == "evaluate":
        tasks = _dedupe_tasks(tasks)
    tasks = _limit_tasks(tasks, args.max_tasks)

    config_path = RUNTIME_ROOT / "tasks" / "env_configs" / "alfworld_config.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        env_config = yaml.safe_load(stream)
    env_config["external_game_root"] = str(args.alfworld_game_root)
    env = AlfworldEnv(env_config=env_config, max_trials=args.max_trials)
    llm = GPTChat(model_name=args.model)
    memory = MemCoMASMemory(
        namespace="memco",
        global_config={
            "working_dir": str(args.memory_dir),
            "viz_output_dir": str(args.log_dir),
            "task_name": "alfworld",
            "memco_dynamic_graph": True,
            "memco_router": "textloss",
            "memco_owner_scene": args.domain,
            "memco_settings": args.memory_settings,
            "memco_freeze_memory": args.mode == "evaluate",
            "memco_shared_global_dir": str(args.shared_global_dir),
            "memco_promotion_threshold": args.promotion_threshold,
            "memco_promotion_policy": args.promotion_policy,
            "memco_wilson_alpha": args.wilson_alpha,
            "memco_wilson_threshold": args.wilson_threshold,
        },
        llm_model=llm,
        embedding_func=EmbeddingFunc("sentence-transformers/all-MiniLM-L6-v2"),
    )
    workflow = AutoGen()
    recorder = AlfworldRecorder(working_dir=str(args.log_dir), namespace="total_task")
    recorder.set_quiet_console(True)
    workflow.add_observer(recorder)
    workflow.build_system(
        ReasoningIO(llm_model=llm), memory, env, {"autogen": True, "silent_mas": True, "insights_topk": 1}
    )

    task_rows = []
    recorder.dataset_begin()
    for task_id, task in enumerate(tasks):
        recorder.task_begin(task_id, task)
        task_main, task_description = env.set_env(task)
        task_type = task["task_type"]
        task.update(
            task_main=task_main,
            task_description=task_description,
            few_shots=[alfworld_few_shots[f"react_{task_type}_2"]],
            env_ref=env,
            task_index=task_id,
            task_index_1based=task_id + 1,
            mas_memory_type="memco",
            model_type="",
        )
        for agent in workflow.agents_team.values():
            recorder.log(f"------------ MAS Agent: {agent.name} ------------")
            recorder.log(agent.add_task_instruction(alfworld_solver_system_prompt))
        reward, done = workflow.schedule(task)
        recorder.task_end(reward, done)
        task_rows.append(
            {
                "scene_domain": task["scene_domain"],
                "source_split": task["source_split"],
                "trial_id": task["trial_id"],
                "reward": float(reward),
                "done": bool(done),
            }
        )
    recorder.dataset_end()

    persist = getattr(memory, "persist_entity_graph", None)
    if callable(persist):
        persist()
    _assert_replay_consumed(llm)
    metrics = summarize_task_results(task_rows)
    _write_json(args.metrics_path, metrics)

    if args.mode == "train":
        candidates = sorted((args.memory_dir / "memco").glob("local_*.json"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one local memory artifact, found: {candidates}")
        args.local_memory_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[0], args.local_memory_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "evaluate"])
    parser.add_argument("--family", choices=["alfworld", "fever", "pddl"], required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--shared-global-dir", type=Path, required=True)
    parser.add_argument("--local-memory-path", type=Path, required=True)
    parser.add_argument("--metrics-path", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--memory-settings", default="local_plus_global")
    parser.add_argument("--promotion-threshold", type=float, default=0.35)
    parser.add_argument("--promotion-policy", choices=["legacy", "shadow", "wilson"], default="wilson")
    parser.add_argument("--wilson-alpha", type=float, default=0.05)
    parser.add_argument("--wilson-threshold", type=float, default=0.35)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument("--max-trials", type=int, default=7)
    parser.add_argument("--alfworld-game-root", type=Path)
    parser.add_argument("--alfworld-eval-split", default="valid_seen,valid_unseen")
    parser.add_argument("--alfworld-domains", default="bathroom,bedroom,kitchen,living")
    parser.add_argument("--fever-wiki-cache", type=Path)
    args = parser.parse_args()

    os.environ["NV_DATASET_FAMILY"] = args.family
    if args.family == "fever":
        os.environ["NV_MEMCO_FEVER_POLICY"] = "adaptive_cache"
        if args.fever_wiki_cache:
            os.environ.setdefault("FEVER_WIKI_CACHE", str(args.fever_wiki_cache))
    elif args.family == "pddl":
        os.environ["NV_MEMCO_PDDL_POLICY"] = "action_guard"
    elif not args.alfworld_game_root:
        parser.error("--alfworld-game-root is required for ALFWorld")
    args.memory_dir.mkdir(parents=True, exist_ok=True)
    args.shared_global_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    if args.family == "alfworld":
        _run_alfworld(args)
    elif args.family == "fever":
        _run_fever(args)
    else:
        _run_pddl(args)


if __name__ == "__main__":
    main()
