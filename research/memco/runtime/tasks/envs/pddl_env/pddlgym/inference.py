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

"""Search for assignments of vars to objects to satisfy a conjunction of Literals"""

import functools
from collections import defaultdict

from pddlgym.prolog_interface import PrologInterface
from pddlgym.structs import Literal, LiteralConjunction, ground_literal
from pddlgym.utils import get_object_combinations


def find_satisfying_assignments(
    kb,
    conds,
    variable_sort_fn=None,
    verbose=False,
    max_assignment_count=2,
    type_to_parent_types=None,
    allow_redundant_variables=True,
    constants=None,
    mode="csp",
    init_assignments=None,
):
    if mode == "csp":
        return ProofSearchTree(
            kb,
            allow_redundant_variables=allow_redundant_variables,
            type_to_parent_types=type_to_parent_types,
            constants=constants,
            initial_assignments=init_assignments,
        ).prove(
            list(conds), max_assignment_count=max_assignment_count, variable_sort_fn=variable_sort_fn, verbose=verbose
        )
    if mode == "ground":
        return run_ground_inference(
            kb,
            conds,
            constants=constants,
            allow_redundant_variables=allow_redundant_variables,
            type_to_parent_types=type_to_parent_types,
        )
    assert mode == "prolog"
    assert all(
        len(v) == 1 for v in type_to_parent_types.values()
    ), "TODO: implement support for hierarchical types in prolog inference"
    prolog_interface = PrologInterface(
        kb,
        conds,
        max_assignment_count=max_assignment_count,
        allow_redundant_variables=allow_redundant_variables,
        constants=constants,
    )
    return prolog_interface.run()


def check_goal(state, goal):
    if isinstance(goal, Literal):
        if goal.is_negative and goal.positive in state.literals:
            return False
        if not goal.is_negative and goal not in state.literals:
            return False
        return True
    if isinstance(goal, LiteralConjunction):
        return all(check_goal(state, lit) for lit in goal.literals)
    prolog_interface = PrologInterface(state.literals, goal, max_assignment_count=2, allow_redundant_variables=True)
    assignments = prolog_interface.run()
    return len(assignments) > 0


def run_ground_inference(kb, conds, constants=None, allow_redundant_variables=True, type_to_parent_types=None):
    objs = tuple(sorted({o for lit in kb for o in lit.variables}))
    if constants is not None:
        objs += tuple(sorted(constants))
    conds = tuple(sorted(conds))
    if type_to_parent_types:
        type_to_parent_types_as_tuple = tuple(sorted([(k, tuple(sorted(v))) for k, v in type_to_parent_types.items()]))
    else:
        type_to_parent_types_as_tuple = None
    assignments = []
    for assignment, ground_conds in get_ground_conds(
        conds, objs, type_to_parent_types=type_to_parent_types_as_tuple, allow_duplicates=allow_redundant_variables
    ):
        if ground_conds.issubset(kb):
            assignments.append(assignment)
    return assignments


@functools.lru_cache(maxsize=None)
def get_ground_conds(conds, objs, type_to_parent_types=None, allow_duplicates=True):
    if type_to_parent_types:
        type_to_parent_types = dict(type_to_parent_types)
    vrs = sorted({v for lit in conds for v in lit.variables})
    var_types = [v.var_type for v in vrs]
    for choice in get_object_combinations(
        objs,
        len(var_types),
        var_types=var_types,
        allow_duplicates=allow_duplicates,
        type_to_parent_types=type_to_parent_types,
    ):
        assignment = dict(zip(vrs, choice))
        ground_conds = {ground_literal(lit, assignment) for lit in conds}
        yield (assignment, ground_conds)


def unify(lits1, lits2):
    """Return a tuple of (whether the given frozensets lits1 and lits2 can be
    unified, the mapping if the first return value is True).
    Also returns the mapping.
    """
    sorted_lits1 = sorted(lits1)
    sorted_lits2 = sorted(lits2)

    # Terminate quickly if there is a mismatch between lits
    predicate_order1 = [lit.predicate for lit in sorted_lits1]
    predicate_order2 = [lit.predicate for lit in sorted_lits2]
    if predicate_order1 != predicate_order2:
        return False, None

    # Terminate quickly if there is a mismatch between num objs
    num_objs1 = len({o for lit in lits1 for o in lit.variables})
    num_objs2 = len({o for lit in lits2 for o in lit.variables})
    if num_objs1 != num_objs2:
        return False, None

    # Try to get lucky with a one-to-one mapping
    subs12 = {}
    subs21 = {}
    success = True
    for lit1, lit2 in zip(sorted_lits1, sorted_lits2):
        if not success:
            break
        for v1, v2 in zip(lit1.variables, lit2.variables):
            if v1 in subs12 and subs12[v1] != v2:
                success = False
                break
            elif v2 in subs21:
                success = False
                break
            else:
                subs12[v1] = v2
                subs21[v2] = v1
    if success:
        return True, subs12

    assignments = find_satisfying_assignments(lits2, lits1, allow_redundant_variables=False)
    if not assignments:
        return False, None
    return True, assignments[0]


def can_unify(lits1, lits2):
    """Shorthand for unify that only returns whether the lits can be
    unified, not the mapping
    """
    return unify(lits1, lits2)[0]


class CommitGoalError(Exception):
    pass


class ProofSearchTree(object):
    def __init__(
        self,
        knowledge_base,
        allow_redundant_variables=True,
        initial_assignments=None,
        allow_commit_exception=True,
        type_to_parent_types=None,
        constants=None,
    ):
        self.knowledge_base = self.initialize_kb(knowledge_base)
        self.allow_redundant_variables = allow_redundant_variables
        self.goal_literals = []
        self.initial_assignments = initial_assignments
        self.allow_commit_exception = allow_commit_exception
        self.type_to_parent_types = type_to_parent_types
        self.constants = constants or []

    def initialize_kb(self, knowledge_base):
        self.all_atoms = set()
        d = defaultdict(list)  # predicate to literals
        for literal in knowledge_base:
            d[literal.predicate].append(literal)
            for atom in literal.variables:
                self.all_atoms.add(atom)
        return d

    def prove(self, goal_literal, verbose=False, commit_if_true=False, max_assignment_count=1, variable_sort_fn=None):
        if not isinstance(goal_literal, list):
            goal_literals = [goal_literal]
        else:
            goal_literals = goal_literal
        for goal_lit in goal_literals:
            assert len(goal_lit.variables) == len(
                set(goal_lit.variables)
            ), "Duplicate variables in predicates not supported."
        goal_literals = self.goal_literals + goal_literals

        self.root = {"variable_assignments": {c: c for c in self.constants}}
        self.queue = [self.root]

        all_assignments = []

        if verbose:
            print("Trying to prove goals", goal_literals)

        # Handle zero-arity goals first, separately
        for goal in goal_literals:
            if goal.predicate.arity == 0:
                if goal.is_negative and len(self.knowledge_base[goal.positive.predicate]) > 0:
                    return []
                if not goal.is_negative and len(self.knowledge_base[goal.predicate]) == 0:
                    return []

        variables = set()
        for lit in goal_literals:
            variables.update(set(lit.variables))
        variables = sorted(list(variables), key=variable_sort_fn)

        if verbose:
            print("variables:", variables)

        while len(self.queue) > 0:
            node = self.queue.pop()

            if verbose:
                print("parent:", node["variable_assignments"])

            if set(variables) <= set(node["variable_assignments"]):
                if verbose:
                    print("Done:", set(variables), set(node["variable_assignments"]))

                all_assignments.append(node["variable_assignments"].copy())

                if len(all_assignments) >= max_assignment_count:
                    if commit_if_true:
                        self.commit_goal(goal_lit)
                    return all_assignments

            for child in self.get_children(node, variables, goal_literals, verbose=verbose):
                if verbose:
                    print(" child:", child["variable_assignments"])
                # Forward checking.
                if any(
                    not self.get_possible_assignments(var, child["variable_assignments"], goal_literals)
                    for var in variables
                    if var not in child["variable_assignments"]
                ):
                    continue
                self.queue.append(child)

        return all_assignments

    def commit_goal(self, goal_literal):
        if self.allow_commit_exception:
            if not self.prove(goal_literal, verbose=False):
                raise CommitGoalError("Tried to commit a goal literal that cannot be proven!")

        self.goal_literals.append(goal_literal)

    def remove_goal(self, goal_literal):
        self.goal_literals.remove(goal_literal)

    def get_children(self, node, variables, goal_literals, verbose=False):
        next_variable = None
        for variable in variables:
            if variable not in node["variable_assignments"]:
                next_variable = variable
                break
        if next_variable is None:
            return

        for possible_assignment in self.get_possible_assignments(
            next_variable, node["variable_assignments"], goal_literals, verbose=verbose
        ):
            yield self.create_child_node(next_variable, possible_assignment, node, goal_literals)

    def get_possible_assignments(self, variable, established_assignments, goal_literals, verbose=False):
        possible_assignments = None
        impossible_assignments = None

        already_assigned_atoms = set([v for k, v in established_assignments.items()])

        variable_involved_in_positive_goal = False

        for goal_literal in goal_literals:
            if variable not in goal_literal.variables:
                continue

            if not goal_literal.is_negative:
                variable_involved_in_positive_goal = True

            possible_atoms = set()
            inevitable_atoms = set()

            for kb_literal in self.knowledge_base[goal_literal.predicate.positive]:

                literal_may_hold = True  # depending on other vars, which are not yet bound
                literal_definitely_holds = True  # if all vars are bound
                variable_atom = None

                for v, atom in zip(goal_literal.variables, kb_literal.variables):
                    if v == variable:
                        if not (self.allow_redundant_variables) and (atom in already_assigned_atoms):
                            literal_may_hold = False
                            literal_definitely_holds = False
                            break
                        elif not self.type_is_of_type(atom.var_type, v.var_type):
                            literal_may_hold = False
                            literal_definitely_holds = False
                            break
                        else:
                            variable_atom = atom
                    elif v in established_assignments and established_assignments[v] != atom:
                        literal_may_hold = False
                        literal_definitely_holds = False
                        break
                    elif v not in established_assignments:
                        literal_definitely_holds = False

                if literal_may_hold:
                    possible_atoms.add(variable_atom)

                if literal_definitely_holds:
                    inevitable_atoms.add(variable_atom)

            if goal_literal.is_negative:
                if verbose:
                    import ipdb

                    ipdb.set_trace()
                if impossible_assignments is None:
                    impossible_assignments = inevitable_atoms
                else:
                    impossible_assignments |= inevitable_atoms

            else:
                if possible_assignments is None:
                    possible_assignments = possible_atoms
                else:
                    possible_assignments &= possible_atoms

        if possible_assignments is None:
            possible_assignments = set()

        if impossible_assignments is None:
            impossible_assignments = set()

        if self.initial_assignments is not None and variable in self.initial_assignments:
            initial_assignment = self.initial_assignments[variable]
            if initial_assignment in possible_assignments:
                possible_assignments = {initial_assignment}
            else:
                possible_assignments = set()

        if not variable_involved_in_positive_goal:
            assert len(possible_assignments) == 0
            possible_assignments = self.get_atoms_of_type(variable.var_type)
            possible_assignments -= already_assigned_atoms

        return possible_assignments - impossible_assignments

    def type_is_of_type(self, type1, type2):
        if self.type_to_parent_types is None:
            return type1 == type2
        return type2 in self.type_to_parent_types[type1]

    def get_atoms_of_type(self, var_type):
        return {o for o in self.all_atoms if self.type_is_of_type(o.var_type, var_type)}

    def create_child_node(self, variable, assignment, parent_node, goal_literals):
        variable_assignments = parent_node["variable_assignments"].copy()
        variable_assignments[variable] = assignment
        return {"variable_assignments": variable_assignments}
