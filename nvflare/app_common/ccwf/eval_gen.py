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

from nvflare.fuel.utils.validation_utils import check_non_negative_int


class EvalInclusionRC(Enum):
    CAN_INCLUDE = 0
    EVALUATOR_CONFLICT = 1
    ENOUGH_ACTIONS_FOR_EVALUATEE = 2
    ENOUGH_ACTIONS_FOR_EVALUATOR = 3


def _check_names(arg_name, names_to_check):
    if not names_to_check:
        raise ValueError(f"no {arg_name}")

    if not isinstance(names_to_check, list):
        raise ValueError(f"expect {arg_name} to be a list of str but got {type(names_to_check)}")

    if not all(isinstance(e, str) for e in names_to_check):
        raise ValueError(f"expect {arg_name} to be a list of str but some items are not str")


def parallel_eval_generator(evaluators: List[str], evaluatees: List[str], max_parallel_actions: int):
    """Generates parallel evaluations to be performed.

    Args:
        evaluators: names of evaluators
        evaluatees: names of evaluatees
        max_parallel_actions: max parallel actions per site (evaluator or evaluatee)

    Each time iterated, it generates a list of evaluations that can be performed in parallel.
    An evaluation is expressed as a tuple of (evaluator name, evaluatee name).
    """
    _check_names("evaluators", evaluators)
    _check_names("evaluatees", evaluatees)
    check_non_negative_int("max_parallel_actions", max_parallel_actions)

    evaluatee_states = [(e, list(evaluators)) for e in evaluatees]
    while evaluatee_states:
        result = []
        empty_evaluatees = []
        for ee in evaluatee_states:
            e, evaluators = ee
            accepted_evaluators = []
            for t in evaluators:
                target = (t, e)
                rc = _can_be_included(result, target, max_parallel_actions)
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
            evaluatee_states.remove(ee)

        yield result


def _can_be_included(evals, target, max_parallel_actions) -> EvalInclusionRC:
    """Determine whether the target evaluation can be included into the set of evals without violating
    parallel evaluation rules.

    Args:
        evals: the set of evaluations already included
        target: the evaluation in question, expressed as a tuple (evaluator name, evaluatee name)
        max_parallel_actions: max parallel actions allowed per actor (evaluator or evaluatee).

    Returns: an EvalInclusionRC

    """
    evaluator_actions = 0
    evaluatee_actions = 0
    evaluator_t, evaluatee_t = target
    for p in evals:
        evaluator_p, evaluatee_p = p
        if evaluator_t == evaluator_p:
            # the evaluator is already in the evals - we allow only once for the same evaluator
            return EvalInclusionRC.EVALUATOR_CONFLICT

        if evaluator_t == evaluatee_p:
            # the evaluator of the target is already an evaluatee of another evaluation
            evaluator_actions += 1

        if evaluatee_t == evaluator_p:
            # the evaluatee of the target is already an evaluator of another evaluation
            evaluatee_actions += 1

        if evaluatee_t == evaluatee_p and evaluator_p != evaluatee_p:
            # the evaluatee of the target is already an evaluatee of another evaluation
            evaluatee_actions += 1

    if evaluatee_actions > max_parallel_actions:
        # if the target is included, its evaluatee_actions would be too much
        return EvalInclusionRC.ENOUGH_ACTIONS_FOR_EVALUATEE

    if evaluator_actions > max_parallel_actions:
        # if the target is included, its evaluator_actions would be too much
        return EvalInclusionRC.ENOUGH_ACTIONS_FOR_EVALUATOR

    # the target can be included!
    return EvalInclusionRC.CAN_INCLUDE
