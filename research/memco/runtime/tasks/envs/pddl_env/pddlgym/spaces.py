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

"""Gym spaces involving Literals.

Unlike typical spaces, Literal spaces may change with
each episode, since objects, and therefore possible
groundings, may change with each new PDDL problem.
"""
import itertools
import os
import tempfile
from collections import defaultdict

from gym.spaces import Space
from pddlgym.downward_translate.instantiate import explore as downward_explore
from pddlgym.downward_translate.pddl_parser import open as downward_open
from pddlgym.parser import PDDLProblemParser
from pddlgym.structs import Literal, LiteralConjunction, ground_literal
from pddlgym.utils import nostdout

TMP_PDDL_DIR = "/dev/shm" if os.path.exists("/dev/shm") else None


class LiteralSpace(Space):

    def __init__(
        self, predicates, lit_valid_test=lambda state, lit: True, type_hierarchy=None, type_to_parent_types=None
    ):
        self.predicates = sorted(predicates)
        self.num_predicates = len(predicates)
        self._objects = None
        self._lit_valid_test = lit_valid_test
        self.type_hierarchy = type_hierarchy
        self._type_to_parent_types = type_to_parent_types
        super().__init__()

    def reset_initial_state(self, initial_state):
        self._objects = None

    def _update_objects_from_state(self, state):
        """Given a state, extract the objects and if they have changed,
        recompute all ground literals
        """
        # Check whether the objects have changed
        # If so, we need to recompute the ground literals
        if state.objects == self._objects:
            return

        # Organize objects by type
        self._type_to_objs = defaultdict(list)

        for obj in sorted(state.objects):
            if self._type_to_parent_types is None:
                self._type_to_objs[obj.var_type].append(obj)
            else:
                for t in self._type_to_parent_types[obj.var_type]:
                    self._type_to_objs[t].append(obj)

        self._objects = state.objects
        self._all_ground_literals = sorted(self._compute_all_ground_literals(state))

    def sample_literal(self, state):
        while True:
            num_lits = len(self._all_ground_literals)
            idx = self.np_random.choice(num_lits)
            lit = self._all_ground_literals[idx]
            if self._lit_valid_test(state, lit):
                break
        return lit

    def sample(self, state):
        self._update_objects_from_state(state)
        return self.sample_literal(state)

    def all_ground_literals(self, state, valid_only=True):
        self._update_objects_from_state(state)
        if not valid_only:
            return set(self._all_ground_literals)
        return set(l for l in self._all_ground_literals if self._lit_valid_test(state, l))

    def _compute_all_ground_literals(self, state):
        all_ground_literals = set()
        for predicate in self.predicates:
            choices = [self._type_to_objs[vt] for vt in predicate.var_types]
            for choice in itertools.product(*choices):
                if len(set(choice)) != len(choice):
                    continue
                lit = predicate(*choice)
                all_ground_literals.add(lit)
        return all_ground_literals


class LiteralActionSpace(LiteralSpace):
    """Literal space with more efficient valid action generation.

    For now, assumes operators_as_actions.
    """

    def __init__(self, domain, predicates, type_hierarchy=None, type_to_parent_types=None):
        self.domain = domain
        self._initial_state = None
        if not domain.operators_as_actions:
            raise NotImplementedError()

        # Validate and organize operators
        action_predicate_to_operators = {}
        for operator_name, operator in domain.operators.items():
            assert len([p for p in predicates if p.name == operator_name]) == 1
            action_predicate = [p for p in predicates if p.name == operator_name][0]
            action_predicate_to_operators[action_predicate] = operator
            if isinstance(operator.preconds, LiteralConjunction):
                assert all([isinstance(l, Literal) for l in operator.preconds.literals])
            else:
                assert isinstance(operator.preconds, Literal)
        self._action_predicate_to_operators = action_predicate_to_operators

        super().__init__(predicates, type_hierarchy=type_hierarchy, type_to_parent_types=type_to_parent_types)

    def reset_initial_state(self, initial_state):
        super().reset_initial_state(initial_state)
        self._initial_state = initial_state

    def _update_objects_from_state(self, state):
        # Check whether the objects have changed
        # If so, we need to recompute things
        if state.objects == self._objects:
            return

        # Parent class update
        super()._update_objects_from_state(state)

        # Recompute all ground operators
        # Associate each ground action literal with ground preconditions
        self._ground_action_to_pos_preconds = {}
        self._ground_action_to_neg_preconds = {}
        for ground_action in self._all_ground_literals:
            operator = self._action_predicate_to_operators[ground_action.predicate]
            if isinstance(operator.preconds, LiteralConjunction):
                lifted_preconds = operator.preconds.literals
            else:
                assert isinstance(operator.preconds, Literal)
                lifted_preconds = [operator.preconds]
            subs = dict(zip(operator.params, ground_action.variables))
            subs.update(zip(self.domain.constants, self.domain.constants))
            preconds = [ground_literal(lit, subs) for lit in lifted_preconds]
            pos_preconds, neg_preconds = set(), set()
            for p in preconds:
                if p.is_negative:
                    neg_preconds.add(p.positive)
                else:
                    pos_preconds.add(p)
            self._ground_action_to_pos_preconds[ground_action] = pos_preconds
            self._ground_action_to_neg_preconds[ground_action] = neg_preconds

    def sample_literal(self, state):
        valid_literals = self.all_ground_literals(state)
        valid_literals = list(sorted(valid_literals))
        return valid_literals[self.np_random.choice(len(valid_literals))]

    def sample(self, state):
        return self.sample_literal(state)

    def all_ground_literals(self, state, valid_only=True):
        self._update_objects_from_state(state)
        assert valid_only, "The point of this class is to avoid the cross product!"
        valid_literals = set()
        for ground_action in self._all_ground_literals:
            pos_preconds = self._ground_action_to_pos_preconds[ground_action]
            if not pos_preconds.issubset(state.literals):
                continue
            neg_preconds = self._ground_action_to_neg_preconds[ground_action]
            if len(neg_preconds & state.literals) > 0:
                continue
            valid_literals.add(ground_action)
        return valid_literals

    def _compute_all_ground_literals(self, state):
        """Call FastDownward's instantiator."""
        # Generate temporary files to hand over to instantiator.
        assert state.objects == self._initial_state.objects
        d_desc, domain_fname = tempfile.mkstemp(dir=TMP_PDDL_DIR, text=True)
        self.domain.write(domain_fname)
        p_desc, problem_fname = tempfile.mkstemp(dir=TMP_PDDL_DIR, text=True)
        with os.fdopen(p_desc, "w") as f:
            PDDLProblemParser.create_pddl_file(
                file_or_filepath=f,
                objects=state.objects - set(self.domain.constants),
                initial_state=self._initial_state.literals,
                problem_name="myproblem",
                domain_name=self.domain.domain_name,
                goal=state.goal,
                fast_downward_order=True,
            )
        # Call instantiator.
        task = downward_open(domain_fname, problem_fname)
        with nostdout():
            _, _, actions, _, _ = downward_explore(task)
        # Post-process to our representation.
        obj_name_to_obj = {obj.name: obj for obj in state.objects}
        all_ground_literals = set()
        for action in actions:
            name = action.name.strip().strip("()").split()
            pred_name, obj_names = name[0], name[1:]
            if len(set(obj_names)) != len(obj_names):
                continue
            pred = None
            for p in self.predicates:
                if p.name == pred_name:
                    assert pred is None
                    pred = p
                    break
            assert pred is not None
            objs = [obj_name_to_obj[obj_name] for obj_name in obj_names]
            all_ground_literals.add(pred(*objs))
        os.close(d_desc)
        return all_ground_literals


class LiteralSetSpace(LiteralSpace):

    def sample(self):
        raise NotImplementedError()
