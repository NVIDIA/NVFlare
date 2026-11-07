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

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from mas.agents import Env


class BaseEnv(Env, ABC):

    def __init__(self, env_config: dict, max_trials: int):
        pass

    @abstractmethod
    def set_env(self, task_config: dict) -> tuple[str, str]:
        pass

    @abstractmethod
    def step(self, action: str) -> tuple[str, float, bool]:
        pass

    @classmethod
    @abstractmethod
    def process_action(cls, action: str) -> str:
        pass

    @abstractmethod
    def feedback(self) -> tuple[float, bool, str]:
        pass


@dataclass
class BaseRecorder:

    working_dir: str = None
    namespace: str = None
    task: str = None

    def __post_init__(self):

        self.file_path = os.path.join(self.working_dir, self.namespace + ".log")
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        self.logger = logging.getLogger(self.namespace)
        self.logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(self.file_path)
        formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self._console_handler = logging.StreamHandler()
        self._console_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(self._console_handler)

        self.current_task_id: int = None
        self.current_task_config: dict = field(default_factory=dict)

    def set_quiet_console(self, quiet: bool) -> None:
        """
        When ``quiet`` is True, detach the console StreamHandler so ``log()`` only writes to the file.
        Progress lines printed via ``print()`` in ``tasks/run.py`` are unaffected.
        """
        if not quiet:
            return
        ch = getattr(self, "_console_handler", None)
        if ch is not None and ch in self.logger.handlers:
            self.logger.removeHandler(ch)

    def task_begin(self, task_id: int, task_config: dict) -> None:

        self.current_task_id = task_id
        self.current_task_config = task_config

    def task_end(self, reward: float, done: bool) -> None:

        if self.current_task_id is None or self.current_task_config is None:
            raise RuntimeError("The task id or the task config should not be None.")

    def dataset_begin(self) -> None:

        self.start_time = time.time()

        message: str = "=============== Task Begin ==============="
        self.log(message)

    def dataset_end(self) -> None:
        message: str = "=============== Task End ==============="

        end_time = time.time()
        total_time = end_time - self.start_time

        # 格式化时间显示
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60

        if hours > 0:
            time_message = f"Total execution time: {hours}小时{minutes}分钟{seconds:.1f}秒 ({total_time:.2f}秒)"
        elif minutes > 0:
            time_message = f"Total execution time: {minutes}分钟{seconds:.1f}秒 ({total_time:.2f}秒)"
        else:
            time_message = f"Total execution time: {seconds:.1f}秒 ({total_time:.2f}秒)"

        self.log(message)
        self.log(time_message)

    def log(self, message: str) -> None:

        if hasattr(self, "logger") and self.logger:
            self.logger.info(message)
        else:
            print("Logger is not initialized.")
