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

"""Minimal PDDLGym registration for the four MemCo planning domains."""

import os

import gym
from gym.envs.registration import register


def make(*args, **kwargs):
    return gym.make(*args, disable_env_checker=True, **kwargs)


def _register(name: str) -> None:
    pddl_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pddl")
    register(
        id=f"PDDLEnv{name.capitalize()}-v0",
        entry_point="pddlgym.core:PDDLEnv",
        kwargs={
            "domain_file": os.path.join(pddl_dir, f"{name}.pddl"),
            "problem_dir": os.path.join(pddl_dir, name),
            "operators_as_actions": True,
            "dynamic_action_space": True,
            "raise_error_on_invalid_action": False,
        },
    )


for _name in ("gripper", "blockworld", "barman", "tyreworld"):
    _register(_name)
