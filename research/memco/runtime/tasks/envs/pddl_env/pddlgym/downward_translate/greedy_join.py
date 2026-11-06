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

import sys

from . import pddl, pddl_to_prolog


class OccurrencesTracker:
    """Keeps track of the number of times each variable appears
    in a list of symbolic atoms."""

    def __init__(self, rule):
        self.occurrences = {}
        self.update(rule.effect, +1)
        for cond in rule.conditions:
            self.update(cond, +1)

    def update(self, symatom, delta):
        for var in symatom.args:
            if var[0] == "?":
                if var not in self.occurrences:
                    self.occurrences[var] = 0
                self.occurrences[var] += delta
                assert self.occurrences[var] >= 0
                if not self.occurrences[var]:
                    del self.occurrences[var]

    def variables(self):
        return set(self.occurrences)


class CostMatrix:
    def __init__(self, joinees):
        self.joinees = []
        self.cost_matrix = []
        for joinee in joinees:
            self.add_entry(joinee)

    def add_entry(self, joinee):
        new_row = [self.compute_join_cost(joinee, other) for other in self.joinees]
        self.cost_matrix.append(new_row)
        self.joinees.append(joinee)

    def delete_entry(self, index):
        for row in self.cost_matrix[index + 1 :]:
            del row[index]
        del self.cost_matrix[index]
        del self.joinees[index]

    def find_min_pair(self):
        assert len(self.joinees) >= 2
        min_cost = (sys.maxsize, sys.maxsize)
        for i, row in enumerate(self.cost_matrix):
            for j, entry in enumerate(row):
                if entry < min_cost:
                    min_cost = entry
                    left_index, right_index = i, j
        return left_index, right_index

    def remove_min_pair(self):
        left_index, right_index = self.find_min_pair()
        left, right = self.joinees[left_index], self.joinees[right_index]
        assert left_index > right_index
        self.delete_entry(left_index)
        self.delete_entry(right_index)
        return (left, right)

    def compute_join_cost(self, left_joinee, right_joinee):
        left_vars = pddl_to_prolog.get_variables([left_joinee])
        right_vars = pddl_to_prolog.get_variables([right_joinee])
        if len(left_vars) > len(right_vars):
            left_vars, right_vars = right_vars, left_vars
        common_vars = left_vars & right_vars
        return (len(left_vars) - len(common_vars), len(right_vars) - len(common_vars), -len(common_vars))

    def can_join(self):
        return len(self.joinees) >= 2


class ResultList:
    def __init__(self, rule, name_generator):
        self.final_effect = rule.effect
        self.result = []
        self.name_generator = name_generator

    def get_result(self):
        self.result[-1].effect = self.final_effect
        return self.result

    def add_rule(self, type, conditions, effect_vars):
        effect = pddl.Atom(next(self.name_generator), effect_vars)
        rule = pddl_to_prolog.Rule(conditions, effect)
        rule.type = type
        self.result.append(rule)
        return rule.effect


def greedy_join(rule, name_generator):
    assert len(rule.conditions) >= 2
    cost_matrix = CostMatrix(rule.conditions)
    occurrences = OccurrencesTracker(rule)
    result = ResultList(rule, name_generator)

    while cost_matrix.can_join():
        joinees = list(cost_matrix.remove_min_pair())
        for joinee in joinees:
            occurrences.update(joinee, -1)

        common_vars = set(joinees[0].args) & set(joinees[1].args)
        condition_vars = set(joinees[0].args) | set(joinees[1].args)
        effect_vars = occurrences.variables() & condition_vars
        for i, joinee in enumerate(joinees):
            joinee_vars = set(joinee.args)
            retained_vars = joinee_vars & (effect_vars | common_vars)
            if retained_vars != joinee_vars:
                joinees[i] = result.add_rule("project", [joinee], sorted(retained_vars))
        joint_condition = result.add_rule("join", joinees, sorted(effect_vars))
        cost_matrix.add_entry(joint_condition)
        occurrences.update(joint_condition, +1)

    # assert occurrences.variables() == set(rule.effect.args)
    # for var in set(rule.effect.args):
    #     assert occurrences.occurrences[var] == 2 * rule.effect.args.count(var)
    return result.get_result()
