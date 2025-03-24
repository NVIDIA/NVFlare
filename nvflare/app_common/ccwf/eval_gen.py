# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from enum import Enum
from typing import List


class EvalInclusionRC(Enum):
    CAN_INCLUDE = 0
    EVALUATOR_CONFLICT = 1
    ENOUGH_ACTIONS_FOR_EVALUATEE = 2
    ENOUGH_ACTIONS_FOR_EVALUATOR = 3


class EvalGenerator:
    def __init__(self, evaluators: List[str], evaluatees: List[str], max_parallel_actions: int):
        """Constructor of EvalGenerator.

        Args:
            evaluators: names of evaluators
            evaluatees: names of evaluatees
            max_parallel_actions: max parallel actions per site (evaluator or evaluatee)
        """
        self._evaluatee_states = [(e, list(evaluators)) for e in evaluatees]
        self.max_parallel_actions = max_parallel_actions
        self.evaluators = evaluators
        self.evaluatees = evaluatees

    def is_empty(self):
        """Determine whether the generator has any remaining evaluations to be processed.

        Returns: True if the generator has no remaining evaluations; False otherwise.

        """
        return False if self._evaluatee_states else True

    def _can_be_included(self, evals, target) -> EvalInclusionRC:
        evaluator_actions = 0
        evaluatee_actions = 0
        evaluator_t, evaluatee_t = target
        for p in evals:
            evaluator_p, evaluatee_p = p
            if evaluator_t == evaluator_p:
                # the evaluator is already in the eval - we allow only once for the same evaluator
                return EvalInclusionRC.EVALUATOR_CONFLICT

            if evaluator_t == evaluatee_p:
                evaluator_actions += 1

            if evaluatee_t == evaluator_p:
                evaluatee_actions += 1

            if evaluatee_t == evaluatee_p and evaluator_p != evaluatee_p:
                evaluatee_actions += 1

        if evaluatee_actions > self.max_parallel_actions:
            return EvalInclusionRC.ENOUGH_ACTIONS_FOR_EVALUATEE

        if evaluator_actions > self.max_parallel_actions:
            return EvalInclusionRC.ENOUGH_ACTIONS_FOR_EVALUATOR

        return EvalInclusionRC.CAN_INCLUDE

    def get_parallel_evals(self):
        """Determine next set of evaluations that can be done in parallel.

        Returns: a list of evals that can be done in parallel.

        """
        result = []
        empty_evaluatees = []
        for ee in self._evaluatee_states:
            e, evaluators = ee
            accepted_evaluators = []
            for t in evaluators:
                target = (t, e)
                rc = self._can_be_included(result, target)
                if rc == EvalInclusionRC.CAN_INCLUDE:
                    result.append(target)
                    accepted_evaluators.append(t)
                elif rc == EvalInclusionRC.ENOUGH_ACTIONS_FOR_EVALUATEE:
                    # no need to try other evaluators with this evaluatee
                    break
                else:
                    # this evaluator cannot be included into the result either because its inclusion
                    # will conflict with another eval in the result, or because it already has enough actions.
                    # we'll try next evaluator.
                    continue

            if accepted_evaluators:
                for t in accepted_evaluators:
                    evaluators.remove(t)

                if not evaluators:
                    empty_evaluatees.append(ee)

        for ee in empty_evaluatees:
            self._evaluatee_states.remove(ee)

        return result
