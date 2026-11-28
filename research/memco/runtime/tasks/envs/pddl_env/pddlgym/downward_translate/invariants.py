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

import itertools
from collections import defaultdict

from . import constraints, pddl, tools

# Notes:
# All parts of an invariant always use all non-counted variables
# -> the arity of all predicates covered by an invariant is either the
# number of the invariant variables or this value + 1
#
# we currently keep the assumption that each predicate occurs at most once
# in every invariant.


def invert_list(alist):
    result = defaultdict(list)
    for pos, arg in enumerate(alist):
        result[arg].append(pos)
    return result


def instantiate_factored_mapping(pairs):
    part_mappings = [
        [list(zip(preimg, perm_img)) for perm_img in itertools.permutations(img)] for (preimg, img) in pairs
    ]
    return tools.cartesian_product(part_mappings)


def find_unique_variables(action, invariant):
    # find unique names for invariant variables
    params = {p.name for p in action.parameters}
    for eff in action.effects:
        params.update([p.name for p in eff.parameters])
    inv_vars = []
    counter = itertools.count()
    for _ in range(invariant.arity()):
        while True:
            new_name = "?v%i" % next(counter)
            if new_name not in params:
                inv_vars.append(new_name)
                break
    return inv_vars


def get_literals(condition):
    if isinstance(condition, pddl.Literal):
        yield condition
    elif isinstance(condition, pddl.Conjunction):
        yield from condition.parts


def ensure_conjunction_sat(system, *parts):
    """Modifies the constraint system such that it is only solvable if the
    conjunction of all parts is satisfiable.

    Each part must be an iterator, generator, or an iterable over
    literals."""
    pos = defaultdict(set)
    neg = defaultdict(set)
    for literal in itertools.chain(*parts):
        if literal.predicate == "=":  # use (in)equalities in conditions
            if literal.negated:
                n = constraints.NegativeClause([literal.args])
                system.add_negative_clause(n)
            else:
                a = constraints.Assignment([literal.args])
                system.add_assignment_disjunction([a])
        else:
            if literal.negated:
                neg[literal.predicate].add(literal)
            else:
                pos[literal.predicate].add(literal)

    for pred, posatoms in pos.items():
        if pred in neg:
            for posatom in posatoms:
                for negatom in neg[pred]:
                    parts = list(zip(negatom.args, posatom.args))
                    if parts:
                        negative_clause = constraints.NegativeClause(parts)
                        system.add_negative_clause(negative_clause)


def ensure_cover(system, literal, invariant, inv_vars):
    """Modifies the constraint system such that it is only solvable if the
    invariant covers the literal"""
    a = invariant.get_covering_assignments(inv_vars, literal)
    assert len(a) == 1
    # if invariants could contain several parts of one predicate, this would
    # not be true but the depending code in parts relies on this assumption
    system.add_assignment_disjunction(a)


def ensure_inequality(system, literal1, literal2):
    """Modifies the constraint system such that it is only solvable if the
    literal instantiations are not equal (ignoring whether one is negated and
    the other is not)"""
    if literal1.predicate == literal2.predicate and literal1.args:
        parts = list(zip(literal1.args, literal2.args))
        system.add_negative_clause(constraints.NegativeClause(parts))


class InvariantPart:
    def __init__(self, predicate, order, omitted_pos=-1):
        self.predicate = predicate
        self.order = order
        self.omitted_pos = omitted_pos

    def __eq__(self, other):
        # This implies equality of the omitted_pos component.
        return self.predicate == other.predicate and self.order == other.order

    def __ne__(self, other):
        return self.predicate != other.predicate or self.order != other.order

    def __le__(self, other):
        return self.predicate <= other.predicate or self.order <= other.order

    def __lt__(self, other):
        return self.predicate < other.predicate or self.order < other.order

    def __hash__(self):
        return hash((self.predicate, tuple(self.order)))

    def __str__(self):
        var_string = " ".join(map(str, self.order))
        omitted_string = ""
        if self.omitted_pos != -1:
            omitted_string = " [%d]" % self.omitted_pos
        return "%s %s%s" % (self.predicate, var_string, omitted_string)

    def arity(self):
        return len(self.order)

    def get_assignment(self, parameters, literal):
        equalities = [(arg, literal.args[argpos]) for arg, argpos in zip(parameters, self.order)]
        return constraints.Assignment(equalities)

    def get_parameters(self, literal):
        return [literal.args[pos] for pos in self.order]

    def instantiate(self, parameters):
        args = ["?X"] * (len(self.order) + (self.omitted_pos != -1))
        for arg, argpos in zip(parameters, self.order):
            args[argpos] = arg
        return pddl.Atom(self.predicate, args)

    def possible_mappings(self, own_literal, other_literal):
        allowed_omissions = len(other_literal.args) - len(self.order)
        if allowed_omissions not in (0, 1):
            return []
        own_parameters = self.get_parameters(own_literal)
        arg_to_ordered_pos = invert_list(own_parameters)
        other_arg_to_pos = invert_list(other_literal.args)
        factored_mapping = []

        for key, other_positions in other_arg_to_pos.items():
            own_positions = arg_to_ordered_pos.get(key, [])
            len_diff = len(own_positions) - len(other_positions)
            if len_diff >= 1 or len_diff <= -2 or len_diff == -1 and not allowed_omissions:
                return []
            if len_diff:
                own_positions.append(-1)
                allowed_omissions = 0
            factored_mapping.append((other_positions, own_positions))
        return instantiate_factored_mapping(factored_mapping)

    def possible_matches(self, own_literal, other_literal):
        assert self.predicate == own_literal.predicate
        result = []
        for mapping in self.possible_mappings(own_literal, other_literal):
            new_order = [None] * len(self.order)
            omitted = -1
            for key, value in mapping:
                if value == -1:
                    omitted = key
                else:
                    new_order[value] = key
            result.append(InvariantPart(other_literal.predicate, new_order, omitted))
        return result

    def matches(self, other, own_literal, other_literal):
        return self.get_parameters(own_literal) == other.get_parameters(other_literal)


class Invariant:
    # An invariant is a logical expression of the type
    #   forall V1...Vk: sum_(part in parts) weight(part, V1, ..., Vk) <= 1.
    # k is called the arity of the invariant.
    # A "part" is a symbolic fact only variable symbols in {V1, ..., Vk, X};
    # the symbol X may occur at most once.

    def __init__(self, parts):
        self.parts = frozenset(parts)
        self.predicates = {part.predicate for part in parts}
        self.predicate_to_part = {part.predicate: part for part in parts}
        assert len(self.parts) == len(self.predicates)

    def __eq__(self, other):
        return self.parts == other.parts

    def __ne__(self, other):
        return self.parts != other.parts

    def __lt__(self, other):
        return self.parts < other.parts

    def __le__(self, other):
        return self.parts <= other.parts

    def __hash__(self):
        return hash(self.parts)

    def __str__(self):
        return "{%s}" % ", ".join(str(part) for part in self.parts)

    def __repr__(self):
        return "<Invariant %s>" % self

    def arity(self):
        return next(iter(self.parts)).arity()

    def get_parameters(self, atom):
        return self.predicate_to_part[atom.predicate].get_parameters(atom)

    def instantiate(self, parameters):
        return [part.instantiate(parameters) for part in self.parts]

    def get_covering_assignments(self, parameters, atom):
        part = self.predicate_to_part[atom.predicate]
        return [part.get_assignment(parameters, atom)]
        # if there were more parts for the same predicate the list
        # contained more than one element

    def check_balance(self, balance_checker, enqueue_func):
        # Check balance for this hypothesis.
        actions_to_check = set()
        for part in self.parts:
            actions_to_check |= balance_checker.get_threats(part.predicate)
        for action in actions_to_check:
            heavy_action = balance_checker.get_heavy_action(action)
            if self.operator_too_heavy(heavy_action):
                return False
            if self.operator_unbalanced(action, enqueue_func):
                return False
        return True

    def operator_too_heavy(self, h_action):
        add_effects = [
            eff
            for eff in h_action.effects
            if not eff.literal.negated and self.predicate_to_part.get(eff.literal.predicate)
        ]
        inv_vars = find_unique_variables(h_action, self)

        if len(add_effects) <= 1:
            return False

        for eff1, eff2 in itertools.combinations(add_effects, 2):
            system = constraints.ConstraintSystem()
            ensure_inequality(system, eff1.literal, eff2.literal)
            ensure_cover(system, eff1.literal, self, inv_vars)
            ensure_cover(system, eff2.literal, self, inv_vars)
            ensure_conjunction_sat(
                system,
                get_literals(h_action.precondition),
                get_literals(eff1.condition),
                get_literals(eff2.condition),
                [eff1.literal.negate()],
                [eff2.literal.negate()],
            )
            if system.is_solvable():
                return True
        return False

    def operator_unbalanced(self, action, enqueue_func):
        inv_vars = find_unique_variables(action, self)
        relevant_effs = [eff for eff in action.effects if self.predicate_to_part.get(eff.literal.predicate)]
        add_effects = [eff for eff in relevant_effs if not eff.literal.negated]
        del_effects = [eff for eff in relevant_effs if eff.literal.negated]
        for eff in add_effects:
            if self.add_effect_unbalanced(action, eff, del_effects, inv_vars, enqueue_func):
                return True
        return False

    def minimal_covering_renamings(self, action, add_effect, inv_vars):
        """computes the minimal renamings of the action parameters such
        that the add effect is covered by the action.
        Each renaming is an constraint system"""

        # add_effect must be covered
        assigs = self.get_covering_assignments(inv_vars, add_effect.literal)

        # renaming of operator parameters must be minimal
        minimal_renamings = []
        params = [p.name for p in action.parameters]
        for assignment in assigs:
            system = constraints.ConstraintSystem()
            system.add_assignment(assignment)
            mapping = assignment.get_mapping()
            if len(params) > 1:
                for n1, n2 in itertools.combinations(params, 2):
                    if mapping.get(n1, n1) != mapping.get(n2, n2):
                        negative_clause = constraints.NegativeClause([(n1, n2)])
                        system.add_negative_clause(negative_clause)
            minimal_renamings.append(system)
        return minimal_renamings

    def add_effect_unbalanced(self, action, add_effect, del_effects, inv_vars, enqueue_func):

        minimal_renamings = self.minimal_covering_renamings(action, add_effect, inv_vars)

        lhs_by_pred = defaultdict(list)
        for lit in itertools.chain(
            get_literals(action.precondition),
            get_literals(add_effect.condition),
            get_literals(add_effect.literal.negate()),
        ):
            lhs_by_pred[lit.predicate].append(lit)

        for del_effect in del_effects:
            minimal_renamings = self.unbalanced_renamings(
                del_effect, add_effect, inv_vars, lhs_by_pred, minimal_renamings
            )
            if not minimal_renamings:
                return False

        # Otherwise, the balance check fails => Generate new candidates.
        self.refine_candidate(add_effect, action, enqueue_func)
        return True

    def refine_candidate(self, add_effect, action, enqueue_func):
        """refines the candidate for an add effect that is unbalanced in the
        action and adds the refined one to the queue"""
        part = self.predicate_to_part[add_effect.literal.predicate]
        for del_eff in [eff for eff in action.effects if eff.literal.negated]:
            if del_eff.literal.predicate not in self.predicate_to_part:
                for match in part.possible_matches(add_effect.literal, del_eff.literal):
                    enqueue_func(Invariant(self.parts.union((match,))))

    def unbalanced_renamings(self, del_effect, add_effect, inv_vars, lhs_by_pred, unbalanced_renamings):
        """returns the renamings from unbalanced renamings for which
        the del_effect does not balance the add_effect."""

        system = constraints.ConstraintSystem()
        ensure_cover(system, del_effect.literal, self, inv_vars)

        # Since we may only rename the quantified variables of the delete effect
        # we need to check that "renamings" of constants are already implied by
        # the unbalanced_renaming (of the of the operator parameters). The
        # following system is used as a helper for this. It builds a conjunction
        # that formulates that the constants are NOT renamed accordingly. We
        # below check that this is impossible with each unbalanced renaming.
        check_constants = False
        constant_test_system = constraints.ConstraintSystem()
        for a, b in system.combinatorial_assignments[0][0].equalities:
            # first 0 because the system was empty before we called ensure_cover
            # second 0 because ensure_cover only adds assignments with one entry
            if b[0] != "?":
                check_constants = True
                neg_clause = constraints.NegativeClause([(a, b)])
                constant_test_system.add_negative_clause(neg_clause)

        ensure_inequality(system, add_effect.literal, del_effect.literal)

        still_unbalanced = []
        for renaming in unbalanced_renamings:
            if check_constants:
                new_sys = constant_test_system.combine(renaming)
                if new_sys.is_solvable():
                    # it is possible that the operator arguments are not
                    # mapped to constants as required for covering the delete
                    # effect
                    still_unbalanced.append(renaming)
                    continue

            new_sys = system.combine(renaming)
            if self.lhs_satisfiable(renaming, lhs_by_pred):
                implies_system = self.imply_del_effect(del_effect, lhs_by_pred)
                if not implies_system:
                    still_unbalanced.append(renaming)
                    continue
                new_sys = new_sys.combine(implies_system)
            if not new_sys.is_solvable():
                still_unbalanced.append(renaming)
        return still_unbalanced

    def lhs_satisfiable(self, renaming, lhs_by_pred):
        system = renaming.copy()
        ensure_conjunction_sat(system, *itertools.chain(lhs_by_pred.values()))
        return system.is_solvable()

    def imply_del_effect(self, del_effect, lhs_by_pred):
        """returns a constraint system that is solvable if lhs implies
        the del effect (only if lhs is satisfiable). If a solvable
        lhs never implies the del effect, return None."""
        # del_effect.cond and del_effect.atom must be implied by lhs
        implies_system = constraints.ConstraintSystem()
        for literal in itertools.chain(get_literals(del_effect.condition), [del_effect.literal.negate()]):
            poss_assignments = []
            for match in lhs_by_pred[literal.predicate]:
                if match.negated != literal.negated:
                    continue
                else:
                    a = constraints.Assignment(list(zip(literal.args, match.args)))
                    poss_assignments.append(a)
            if not poss_assignments:
                return None
            implies_system.add_assignment_disjunction(poss_assignments)
        return implies_system
