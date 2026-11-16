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


class NegativeClause:
    # disjunction of inequalities
    def __init__(self, parts):
        self.parts = parts
        assert len(parts)

    def __str__(self):
        disj = " or ".join(["(%s != %s)" % (v1, v2) for (v1, v2) in self.parts])
        return "(%s)" % disj

    def is_satisfiable(self):
        for part in self.parts:
            if part[0] != part[1]:
                return True
        return False

    def apply_mapping(self, m):
        new_parts = [(m.get(v1, v1), m.get(v2, v2)) for (v1, v2) in self.parts]
        return NegativeClause(new_parts)


class Assignment:
    def __init__(self, equalities):
        self.equalities = tuple(equalities)
        # represents a conjunction of expressions ?x = ?y or ?x = d
        # with ?x, ?y being variables and d being a domain value

        self.consistent = None
        self.mapping = None
        self.eq_classes = None

    def __str__(self):
        conj = " and ".join(["(%s = %s)" % (v1, v2) for (v1, v2) in self.equalities])
        return "(%s)" % conj

    def _compute_equivalence_classes(self):
        eq_classes = {}
        for v1, v2 in self.equalities:
            c1 = eq_classes.setdefault(v1, {v1})
            c2 = eq_classes.setdefault(v2, {v2})
            if c1 is not c2:
                if len(c2) > len(c1):
                    v1, c1, v2, c2 = v2, c2, v1, c1
                c1.update(c2)
                for elem in c2:
                    eq_classes[elem] = c1
        self.eq_classes = eq_classes

    def _compute_mapping(self):
        if not self.eq_classes:
            self._compute_equivalence_classes()

        # create mapping: each key is mapped to the smallest
        # element in its equivalence class (with objects being
        # smaller than variables)
        mapping = {}
        for eq_class in self.eq_classes.values():
            variables = [item for item in eq_class if item.startswith("?")]
            constants = [item for item in eq_class if not item.startswith("?")]
            if len(constants) >= 2:
                self.consistent = False
                self.mapping = None
                return
            if constants:
                set_val = constants[0]
            else:
                set_val = min(variables)
            for entry in eq_class:
                mapping[entry] = set_val
        self.consistent = True
        self.mapping = mapping

    def is_consistent(self):
        if self.consistent is None:
            self._compute_mapping()
        return self.consistent

    def get_mapping(self):
        if self.consistent is None:
            self._compute_mapping()
        return self.mapping


class ConstraintSystem:
    def __init__(self):
        self.combinatorial_assignments = []
        self.neg_clauses = []

    def __str__(self):
        combinatorial_assignments = []
        for comb_assignment in self.combinatorial_assignments:
            disj = " or ".join([str(assig) for assig in comb_assignment])
            disj = "(%s)" % disj
            combinatorial_assignments.append(disj)
        assigs = " and\n".join(combinatorial_assignments)

        neg_clauses = [str(clause) for clause in self.neg_clauses]
        neg_clauses = " and ".join(neg_clauses)
        return assigs + "(" + neg_clauses + ")"

    def _all_clauses_satisfiable(self, assignment):
        mapping = assignment.get_mapping()
        for neg_clause in self.neg_clauses:
            clause = neg_clause.apply_mapping(mapping)
            if not clause.is_satisfiable():
                return False
        return True

    def _combine_assignments(self, assignments):
        new_equalities = []
        for a in assignments:
            new_equalities.extend(a.equalities)
        return Assignment(new_equalities)

    def add_assignment(self, assignment):
        self.add_assignment_disjunction([assignment])

    def add_assignment_disjunction(self, assignments):
        self.combinatorial_assignments.append(assignments)

    def add_negative_clause(self, negative_clause):
        self.neg_clauses.append(negative_clause)

    def combine(self, other):
        """Combines two constraint systems to a new system"""
        combined = ConstraintSystem()
        combined.combinatorial_assignments = self.combinatorial_assignments + other.combinatorial_assignments
        combined.neg_clauses = self.neg_clauses + other.neg_clauses
        return combined

    def copy(self):
        other = ConstraintSystem()
        other.combinatorial_assignments = list(self.combinatorial_assignments)
        other.neg_clauses = list(self.neg_clauses)
        return other

    def dump(self):
        print("AssignmentSystem:")
        for comb_assignment in self.combinatorial_assignments:
            disj = " or ".join([str(assig) for assig in comb_assignment])
            print("  ASS: ", disj)
        for neg_clause in self.neg_clauses:
            print("  NEG: ", str(neg_clause))

    def is_solvable(self):
        """Check whether the combinatorial assignments include at least
        one consistent assignment under which the negative clauses
        are satisfiable"""
        for assignments in itertools.product(*self.combinatorial_assignments):
            combined = self._combine_assignments(assignments)
            if not combined.is_consistent():
                continue
            if self._all_clauses_satisfiable(combined):
                return True
        return False
