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

"""Standalone PyTorch distillation helpers.

Basic KD uses losses, teachers and trainer. Optional data-free helpers are
imported explicitly from synthesis, pool, classification and features;
importing basic KD does not load those modules or construct any models. The
synthesis helper does not choose a task model, objective weights, augmentation,
or optimizer policy; applications supply them explicitly.

These helpers target FP32 models/inputs. FP16/AMP execution, automatic precision
conversion and overflow recovery are outside the current scope. Models execute
on the configured device; stored snapshots/pools and built-in random draws use
CPU explicitly, independent of the process default device.
"""

from .losses import kl_divergence
from .teachers import TeacherBuffer, TeacherSnapshot
from .trainer import DistillationTrainer

__all__ = ["kl_divergence", "TeacherBuffer", "TeacherSnapshot", "DistillationTrainer"]
