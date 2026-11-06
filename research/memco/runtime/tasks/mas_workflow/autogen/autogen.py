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

import json
import os
import sys
import time
from dataclasses import dataclass

from mas.agents import Agent, Env
from mas.mas import MetaMAS
from mas.memory import GMemory, MASMemoryBase
from mas.memory.common import AgentMessage
from mas.reasoning import ReasoningBase, ReasoningConfig
from mas.timing_profile import append as _timing_append
from mas.timing_profile import enabled as _timing_enabled

from ..format import format_memory_augmentation_parts, format_prompt_payload
from .autogen_prompt import AUTOGEN_PROMPT


def _append_memory_augment_jsonl(
    log_dir: str,
    task_index: int,
    *,
    task_main: str,
    step: int,
    max_trials: int,
    mas_memory: str,
    prompt_role: str,
    parts: dict[str, str],
) -> None:
    path = os.path.join(log_dir, f"task_{task_index:04d}.jsonl")
    rec = {
        "task_index": task_index,
        "task_index_1based": task_index + 1,
        "task_main": task_main,
        "step": step,
        "max_trials": max_trials,
        "mas_memory": mas_memory,
        "prompt_role": prompt_role,
        "past_successes": parts.get("past_successes", ""),
        "insights": parts.get("insights", ""),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


@dataclass
class AutoGen(MetaMAS):

    def __post_init__(self):

        self.solver_name: str = "solver"
        self.ground_truth_name: str = "ground_truth"
        self.observers = []

        self.reasoning_config = ReasoningConfig(temperature=0, stop_strs=["\n"])

    def build_system(self, reasoning: ReasoningBase, mas_memory: MASMemoryBase, env: Env, config: dict):

        self._quiet: bool = bool(config.get("quiet", False))
        self._successful_topk: int = config.get("successful_topk", 1)
        self._failed_topk: int = config.get("failed_topk", 1)
        self._insights_topk: int = config.get("insights_topk", 3)
        self._threshold: float = config.get("threshold", 0)
        self._use_projector: bool = config.get("use_projector", False)
        self.notify_observers(f"Successful Topk   : {self._successful_topk}")
        self.notify_observers(f"Failed Topk       : {self._failed_topk}")
        self.notify_observers(f"Insights Topk     : {self._insights_topk}")
        self.notify_observers(f"Retrieve Threshold: {self._threshold}")
        self.notify_observers(f"Use Role Projector: {self._use_projector}")

        if not isinstance(reasoning, ReasoningBase):
            raise TypeError("reasoning module must be an instance of ReasoningBase")
        if not isinstance(mas_memory, MASMemoryBase):
            raise TypeError("mas_memory module must be an instance of MASMemoryBase")

        solver_agent: Agent = Agent(
            name=self.solver_name,
            role="solver",
            system_instruction=AUTOGEN_PROMPT.solver_system_prompt,
            reasoning_module=reasoning,
            memory_module=None,
        )

        ground_truth_agent: Agent = Agent(
            name=self.ground_truth_name,
            role="ground truth agent",
            system_instruction=AUTOGEN_PROMPT.ground_truth_system_prompt,
            reasoning_module=reasoning,
            memory_module=None,
        )

        env_executor = env

        self.hire([solver_agent, ground_truth_agent])
        self.set_env(env_executor)
        self.meta_memory = mas_memory

    def add_observer(self, observer):
        self.observers.append(observer)

    def notify_observers(self, message: str):
        for observer in self.observers:
            observer.log(message)

    def _timing_autogen_log_dir(self) -> str:
        gc = getattr(self.meta_memory, "global_config", None) or {}
        return str(gc.get("viz_output_dir") or gc.get("working_dir") or ".")

    def schedule(self, task_config: dict) -> tuple[float, bool]:
        """
        Schedules and executes a task according to the given task configuration.
        This function initializes the task context based on the configuration, retrieves relevant memories and insights,
        and then executes the task by interacting with the environment and agents. It also handles memory updates and feedback.

        Parameters:
        - task_config (dict): A dictionary containing the task configuration, including the main task and description.

        Returns:
        - tuple[float, bool]: Returns the final reward and whether the task was successfully completed.
        """
        if task_config.get("task_main") is None:
            raise ValueError("Missing required keys `task_main` in task_config")
        if task_config.get("task_description") is None:
            raise ValueError("Missing required keys `task_description` in task_config")

        task_main: str = task_config.get("task_main")
        task_description: str = task_config.get("task_description")
        few_shots: list[str] = task_config.get("few_shots", [])
        _intermediate_env: str | None = (task_config.get("env_name") or "").strip() or None
        model_name = str(getattr(getattr(self.meta_memory, "llm_model", None), "model_name", "") or "").strip().lower()
        env_kwargs = task_config.get("env_kwargs") if isinstance(task_config, dict) else {}
        gamefile = ""
        if isinstance(env_kwargs, dict):
            gamefile = str(env_kwargs.get("gamefile", "") or "")
        task_blob = " ".join(
            [
                str(task_main or ""),
                str(task_description or ""),
                gamefile,
                str(task_config.get("env_name", "") or "") if isinstance(task_config, dict) else "",
            ]
        ).lower()
        is_alfworld = (
            ("alfworld" in task_blob)
            or ("alfworld_official" in task_blob)
            or ("game.tw-pddl" in task_blob)
            or ("textworld" in task_blob and "alfred" in task_blob)
        )
        prof = _timing_enabled(
            task_config=task_config,
            global_config=getattr(self.meta_memory, "global_config", None),
        )
        tlog = self._timing_autogen_log_dir() if prof else ""

        # Initialize environment and agents
        env: Env = self.env
        solver: Agent = self.get_agent(self.solver_name)
        ground_truth: Agent = self.get_agent(self.ground_truth_name)
        env.reset()

        self.meta_memory.init_task_context(task_main, task_description)

        t0 = time.perf_counter()
        payload = self.meta_memory.retrieve_prompt_payload(
            query_task=task_main,
            successful_topk=self._successful_topk,
            failed_topk=self._failed_topk,
            insight_topk=self._insights_topk,
            threshold=self._threshold,
            env_ref=env,
            task_config=task_config,
            step_index=0,
        )
        retrieve_s = time.perf_counter() - t0
        payload = payload or {}
        successful_shots: list[str] = [str(item) for item in payload.get("execution_patterns", []) if str(item).strip()]
        raw_rules: list[str] = [str(item) for item in payload.get("insights", []) if str(item).strip()]
        planner_notes: list[str] = [str(item) for item in payload.get("planner_notes", []) if str(item).strip()]
        action_constraints: list[str] = [
            str(item) for item in payload.get("action_constraints", []) if str(item).strip()
        ]
        repair_hints: list[str] = [str(item) for item in payload.get("repair_hints", []) if str(item).strip()]
        t0 = time.perf_counter()
        roles_rules: dict[str, list[str]] = self._project_insights(raw_rules)
        project_insights_s = time.perf_counter() - t0

        # Log the full prompt content once per task (includes trajectories, key steps, insights)
        t0 = time.perf_counter()
        prompt_for_log: str = format_prompt_payload(
            {
                "reference_cases": few_shots,
                "execution_patterns": successful_shots,
                "insights": roles_rules.get(solver.profile, raw_rules),
                "planner_notes": planner_notes,
                "action_constraints": action_constraints,
                "repair_hints": repair_hints,
            },
            self.meta_memory.summarize(),
            intermediate_findings_env=_intermediate_env,
        )
        self.notify_observers("=== Prompt With Memory/Insights ===")
        self.notify_observers(prompt_for_log)
        first_prompt_format_notify_s = time.perf_counter() - t0

        if prof:
            gc = getattr(self.meta_memory, "global_config", None) or {}
            _timing_append(
                tlog,
                "timing_autogen_task.jsonl",
                {
                    "kind": "autogen_preflight",
                    "retrieve_memory_s": retrieve_s,
                    "project_insights_s": project_insights_s,
                    "first_prompt_format_notify_s": first_prompt_format_notify_s,
                    "task_main_preview": (task_main or "")[:160],
                    "episode_index_1based": gc.get("episode_index_1based"),
                },
            )

        # Optional: run first Search automatically and inject result into trajectory (force_search style)
        if task_config.get("auto_first_search"):
            t_afs = time.perf_counter()
            first_search_display = "Search[（已自动执行当前题目检索）]"
            observation, _r, _d = env.step("Search[.]")
            self.meta_memory.move_memory_state(first_search_display, observation)
            self.notify_observers(
                f"Act 1: {first_search_display}\nObs 1: {observation[:500]}{'...' if len(observation) > 500 else ''}"
            )
            if prof:
                _timing_append(
                    tlog,
                    "timing_autogen_task.jsonl",
                    {
                        "kind": "autogen_auto_first_search",
                        "env_step_move_s": time.perf_counter() - t_afs,
                        "episode_index_1based": (getattr(self.meta_memory, "global_config", None) or {}).get(
                            "episode_index_1based"
                        ),
                    },
                )

        # Main loop for task execution
        action_history: list = []
        step_rows: list[dict] = []
        for i in range(env.max_trials):
            if not self._quiet:
                print(f"  [MAS] step {i + 1}/{env.max_trials} ...", flush=True)
                sys.stdout.flush()

            row: dict = {"step": i + 1}
            t0 = time.perf_counter()
            if bool(getattr(self.meta_memory, "refresh_each_step", False)):
                step_payload = (
                    self.meta_memory.retrieve_prompt_payload(
                        query_task=task_main,
                        successful_topk=self._successful_topk,
                        failed_topk=self._failed_topk,
                        insight_topk=self._insights_topk,
                        threshold=self._threshold,
                        env_ref=env,
                        task_config=task_config,
                        step_index=i + 1,
                    )
                    or {}
                )
                successful_shots = [
                    str(item) for item in step_payload.get("execution_patterns", []) if str(item).strip()
                ]
                raw_rules = [str(item) for item in step_payload.get("insights", []) if str(item).strip()]
                planner_notes = [str(item) for item in step_payload.get("planner_notes", []) if str(item).strip()]
                action_constraints = [
                    str(item) for item in step_payload.get("action_constraints", []) if str(item).strip()
                ]
                repair_hints = [str(item) for item in step_payload.get("repair_hints", []) if str(item).strip()]
                roles_rules = self._project_insights(raw_rules)
            user_prompt: str = format_prompt_payload(
                {
                    "reference_cases": few_shots,
                    "execution_patterns": successful_shots,
                    "insights": roles_rules.get(solver.profile, raw_rules),
                    "planner_notes": planner_notes,
                    "action_constraints": action_constraints,
                    "repair_hints": repair_hints,
                },
                self.meta_memory.summarize(),
                intermediate_findings_env=_intermediate_env,
            )
            if prof:
                row["format_prompt_s"] = time.perf_counter() - t0
            _ma_dir = task_config.get("memory_augment_log_dir")
            if _ma_dir:
                _ma_parts = format_memory_augmentation_parts(
                    successful_shots,
                    roles_rules.get(solver.profile, raw_rules),
                )
                _append_memory_augment_jsonl(
                    _ma_dir,
                    int(task_config.get("task_index", 0)),
                    task_main=task_main,
                    step=i + 1,
                    max_trials=int(env.max_trials),
                    mas_memory=str(task_config.get("mas_memory_type") or getattr(self.meta_memory, "namespace", "")),
                    prompt_role="solver",
                    parts=_ma_parts,
                )
            tries = 0
            action_raw = ""
            action = ""
            self.notify_observers(f"[detail][step {i + 1}] solver.input:\n{user_prompt}")

            t0 = time.perf_counter()
            while tries < 3:
                try:
                    action_raw: str = solver.response(user_prompt, self.reasoning_config)
                    self.notify_observers(f"[detail][step {i + 1}] solver.output.raw.try_{tries + 1}: {action_raw}")
                    action: str = action_raw
                    if action == "":
                        self.notify_observers(f"[detail][step {i + 1}] solver.empty_output.try_{tries + 1}")
                        tries += 1
                        continue
                    action = env.process_action(action)
                    self.notify_observers(f"[detail][step {i + 1}] solver.output.processed.try_{tries + 1}: {action}")
                    break
                except Exception as e:
                    self.notify_observers(
                        f"[detail][step {i + 1}] solver.error.try_{tries + 1}: {type(e).__name__}: {e}"
                    )
                    if not self._quiet:
                        print(f"Error during execution of solver agent: {e}")
                tries += 1
            if not action:
                raise RuntimeError(f"solver failed to produce an action after {tries} tries at step {i + 1}")
            if prof:
                row["solver_try_loop_s"] = time.perf_counter() - t0

            name: str = solver.name
            system_instruction = solver.system_instruction

            if self._solver_stuck(action, action_history):
                self.notify_observers(
                    f"[detail][step {i + 1}] ground_truth.triggered: True (repeated action: {action})"
                )
                user_prompt: str = format_prompt_payload(
                    {
                        "reference_cases": few_shots,
                        "execution_patterns": successful_shots,
                        "insights": roles_rules.get(ground_truth.profile, raw_rules),
                        "planner_notes": planner_notes,
                        "action_constraints": action_constraints,
                        "repair_hints": repair_hints,
                    },
                    self.meta_memory.summarize(),
                    intermediate_findings_env=_intermediate_env,
                )
                _ma_dir_gt = task_config.get("memory_augment_log_dir")
                if _ma_dir_gt:
                    _ma_parts_gt = format_memory_augmentation_parts(
                        successful_shots,
                        roles_rules.get(ground_truth.profile, raw_rules),
                    )
                    _append_memory_augment_jsonl(
                        _ma_dir_gt,
                        int(task_config.get("task_index", 0)),
                        task_main=task_main,
                        step=i + 1,
                        max_trials=int(env.max_trials),
                        mas_memory=str(
                            task_config.get("mas_memory_type") or getattr(self.meta_memory, "namespace", "")
                        ),
                        prompt_role="ground_truth",
                        parts=_ma_parts_gt,
                    )
                tries = 0
                action_raw = ""
                action = ""
                self.notify_observers(f"[detail][step {i + 1}] ground_truth.input:\n{user_prompt}")
                t0_gt = time.perf_counter()
                while tries < 3:
                    try:
                        action_raw: str = ground_truth.response(user_prompt, self.reasoning_config)
                        self.notify_observers(
                            f"[detail][step {i + 1}] ground_truth.output.raw.try_{tries + 1}: {action_raw}"
                        )
                        action: str = action_raw
                        if action == "":
                            self.notify_observers(f"[detail][step {i + 1}] ground_truth.empty_output.try_{tries + 1}")
                            tries += 1
                            continue
                        action = env.process_action(action)
                        self.notify_observers(
                            f"[detail][step {i + 1}] ground_truth.output.processed.try_{tries + 1}: {action}"
                        )
                        break
                    except Exception as e:
                        self.notify_observers(
                            f"[detail][step {i + 1}] ground_truth.error.try_{tries + 1}: {type(e).__name__}: {e}"
                        )
                        if not self._quiet:
                            print(f"Error during execution of ground truth agent: {e}")
                    tries += 1
                if not action:
                    raise RuntimeError(f"ground_truth failed to produce an action after {tries} tries at step {i + 1}")
                if prof:
                    row["ground_truth_try_loop_s"] = time.perf_counter() - t0_gt
                name: str = ground_truth.name
                system_instruction = ground_truth.system_instruction
            else:
                self.notify_observers(f"[detail][step {i + 1}] ground_truth.triggered: False")
                if prof:
                    row["ground_truth_try_loop_s"] = 0.0

            repair_fn = getattr(self.meta_memory, "repair_action", None)
            if callable(repair_fn):
                before_repair = action
                try:
                    action = repair_fn(
                        raw_response=action_raw,
                        processed_action=action,
                        env_ref=env,
                        task_config=task_config,
                        step_index=i + 1,
                    )
                    if action != before_repair:
                        self.notify_observers(
                            f"[detail][step {i + 1}] memory.repair_action: {before_repair} -> {action}"
                        )
                except Exception as e:
                    action = before_repair
                    self.notify_observers(f"[detail][step {i + 1}] memory.repair_action.error: {type(e).__name__}: {e}")

            agent_message: AgentMessage = AgentMessage(
                agent_name=name,
                system_instruction=system_instruction,
                user_instruction=user_prompt,
                message=action,
            )
            self.meta_memory.add_agent_node(agent_message, upstream_agent_ids=[])

            t0 = time.perf_counter()
            observation, reward, done = env.step(action)
            if prof:
                row["env_step_s"] = time.perf_counter() - t0
            action_history.append(action)
            action_preview = (action[:60] + "..") if len(action) > 60 else action
            if not self._quiet:
                print(f"  [MAS] step {i + 1} done: {action_preview}", flush=True)
                sys.stdout.flush()

            step_message: str = f"Act {i + 1}: {action}\nObs {i + 1}: {observation}"
            self.notify_observers(step_message)
            self.notify_observers(
                f"[detail][step {i + 1}] env.done={done}, env.won={getattr(env, 'won', None)}, step_reward={reward}"
            )

            t0 = time.perf_counter()
            self.meta_memory.move_memory_state(action, observation, reward=reward)
            if prof:
                row["move_memory_state_s"] = time.perf_counter() - t0
                step_rows.append(row)

            if is_alfworld and "gpt-4o-mini" in model_name and ((i + 1) % 5 == 0) and (not done):
                try:
                    admissible_all = [
                        str(cmd).strip()
                        for cmd in (getattr(env, "last_admissible_commands", []) or [])
                        if str(cmd).strip()
                    ]
                except Exception:
                    admissible_all = []
                admissible_hint = "\n".join(f"- {a}" for a in admissible_all[:20])
                reflect_prompt = (
                    user_prompt + "\n\n" + "Meta-check (does NOT consume an environment step):\n"
                    "Write EXACTLY ONE line starting with `think:` that states:\n"
                    "- what is already done,\n"
                    "- what is still missing,\n"
                    "- and what the next concrete action should be.\n"
                    "Keep it short. Do NOT output an environment action now.\n"
                    "Admissible actions (for choosing your next step):\n"
                    + (admissible_hint or "- (unavailable)")
                    + "\n"
                )
                try:
                    reflection_raw = solver.response(reflect_prompt, self.reasoning_config)
                except Exception:
                    reflection_raw = ""
                reflection = str(reflection_raw or "").strip()
                if reflection:
                    self.notify_observers(f"[detail][step {i + 1}] alfworld_reflect.raw: {reflection}")
                if reflection.lower().startswith("think"):
                    try:
                        agent_message_reflect: AgentMessage = AgentMessage(
                            agent_name=solver.name,
                            system_instruction=solver.system_instruction,
                            user_instruction=reflect_prompt,
                            message=reflection,
                        )
                        self.meta_memory.add_agent_node(agent_message_reflect, upstream_agent_ids=[])
                        self.meta_memory.move_memory_state(reflection, "OK.", reward=0)
                    except Exception:
                        pass

            if done:
                break

        # Final feedback and memory update
        t0 = time.perf_counter()
        final_reward, final_done, final_feedback = self.env.feedback()
        feedback_block_s = time.perf_counter() - t0
        self.notify_observers(final_feedback)
        t0 = time.perf_counter()
        memory_label = bool(final_done)
        self.meta_memory.save_task_context(
            label=memory_label,
            feedback=final_feedback,
            env_ref=env,
            task_config=task_config,
        )
        self.meta_memory.backward(final_done)
        save_backward_s = time.perf_counter() - t0

        if prof:
            gc = getattr(self.meta_memory, "global_config", None) or {}
            _timing_append(
                tlog,
                "timing_autogen_task.jsonl",
                {
                    "kind": "autogen_task_footer",
                    "n_steps_logged": len(step_rows),
                    "feedback_block_s": feedback_block_s,
                    "save_task_context_backward_s": save_backward_s,
                    "final_done": bool(final_done),
                    "episode_index_1based": gc.get("episode_index_1based"),
                    "step_timings": step_rows,
                },
            )

        return final_reward, final_done

    def _solver_stuck(self, current_action: str, action_history: list[str]) -> bool:
        """
        Determines whether the agent is stuck by repeating the same action.

        If the current action is identical to the last two actions in the history,
        the agent is considered to be stuck in a loop.

        Args:
            current_action (str): The action currently being executed by the agent.
            action_history (list[str]): A chronological list of previously taken actions.

        Returns:
            bool: True if the last two actions are the same as the current action; False otherwise.
        """
        return (
            len(action_history) >= 2 and current_action == action_history[-1] and current_action == action_history[-2]
        )

    def _project_insights(self, insights: list[str]) -> dict[str, list[str]]:
        """
        Process insights to generate a dictionary matching roles to insights, based on whether a projector is used.

        Args:
            insights (list[str]): A list of insight strings.

        Returns:
            dict[str, list[str]]: A dictionary with roles as keys and lists of insights as values.
        """

        roles_rules: dict[str, list[str]] = {}
        roles = set([agent.profile for agent in self.agents_team.values()])

        if not self._use_projector or not isinstance(self.meta_memory, GMemory):
            for role in roles:
                roles_rules[role] = insights
        else:
            for role in roles:
                roles_rules[role] = self.meta_memory.project_insights(insights, role)

        # Limit the number of insights per role to self._insights_topk
        for role, insights in roles_rules.items():
            roles_rules[role] = insights[: self._insights_topk]
        return roles_rules
