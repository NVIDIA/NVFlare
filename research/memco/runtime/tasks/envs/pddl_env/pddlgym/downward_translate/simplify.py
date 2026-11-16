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

"""This module contains a function for simplifying tasks in
finite-domain representation (SASTask). Usage:

    simplify.filter_unreachable_propositions(sas_task)

simplifies `sas_task` in-place. If simplification detects that the
task is unsolvable, the function raises `simplify.Impossible`. If it
detects that is has an empty goal, the function raises
`simplify.TriviallySolvable`.

The simplification procedure generates DTGs for the task and then
removes facts that are unreachable from the initial state in a DTG.
Note that such unreachable facts can exist even though we perform a
relaxed reachability analysis before grounding (and DTG reachability
is weaker than relaxed reachability) because the previous relaxed
reachability does not take into account any mutex information, while
PDDL-to-SAS conversion gets rid of certain operators that cannot be
applicable given the mutex information.

Despite the name, the method touches more than the set of facts. For
example, operators that have preconditions on pruned facts are
removed, too. (See also the docstring of
filter_unreachable_propositions.)
"""

from collections import defaultdict
from itertools import count

from . import sas_tasks

DEBUG = False

# TODO:
# This is all quite hackish and would be easier if the translator were
# restructured so that more information is immediately available for
# the propositions, and if propositions had more structure. Directly
# working with int pairs is awkward.


class DomainTransitionGraph:
    """Domain transition graphs.

    Attributes:
    - init (int): the initial state value of the DTG variable
    - size (int): the number of values in the domain
    - arcs (defaultdict: int -> set(int)): the DTG arcs (unlabeled)

    There are no transition labels or goal values.

    The intention is that nodes are represented as ints in {1, ...,
    domain_size}, but this is not enforced.

    For derived variables, the "fallback value" that is produced by
    negation by failure should be used for `init`, so that it is
    always considered reachable.
    """

    def __init__(self, init, size):
        """Create a DTG with no arcs."""
        self.init = init
        self.size = size
        self.arcs = defaultdict(set)

    def add_arc(self, u, v):
        """Add an arc from u to v."""
        self.arcs[u].add(v)

    def reachable(self):
        """Return the values reachable from the initial value.
        Represented as a set(int)."""
        queue = [self.init]
        reachable = set(queue)
        while queue:
            node = queue.pop()
            new_neighbors = self.arcs.get(node, set()) - reachable
            reachable |= new_neighbors
            queue.extend(new_neighbors)
        return reachable

    def dump(self):
        """Dump the DTG."""
        print("DTG size:", self.size)
        print("DTG init value:", self.init)
        print("DTG arcs:")
        for source, destinations in sorted(self.arcs.items()):
            for destination in sorted(destinations):
                print("  %d => %d" % (source, destination))


def build_dtgs(task):
    """Build DTGs for all variables of the SASTask `task`.
    Return a list(DomainTransitionGraph), one for each variable.

    For derived variables, we do not consider the axiom bodies, i.e.,
    we treat each axiom as if it were an operator with no
    preconditions. In the case where the only derived variables used
    are binary and all rules change the value from the default value
    to the non-default value, this results in the correct DTG.
    Otherwise, at worst it results in an overapproximation, which
    would not threaten correctness."""

    init_vals = task.init.values
    sizes = task.variables.ranges
    dtgs = [DomainTransitionGraph(init, size) for (init, size) in zip(init_vals, sizes)]

    def add_arc(var_no, pre_spec, post):
        """Add a DTG arc for var_no induced by transition pre_spec -> post.
        pre_spec may be -1, in which case arcs from every value
        other than post are added."""
        if pre_spec == -1:
            pre_values = set(range(sizes[var_no])).difference([post])
        else:
            pre_values = [pre_spec]
        for pre in pre_values:
            dtgs[var_no].add_arc(pre, post)

    def get_effective_pre(var_no, conditions, effect_conditions):
        """Return combined information on the conditions on `var_no`
        from operator conditions and effect conditions.

        - conditions: dict(int -> int) containing the combined
          operator prevail and preconditions
        - effect_conditions: list(pair(int, int)) containing the
          effect conditions

        Result:
        - -1   if there is no condition on var_no
        - val  if there is a unique condition var_no=val
        - None if there are contradictory conditions on var_no"""

        result = conditions.get(var_no, -1)
        for cond_var_no, cond_val in effect_conditions:
            if cond_var_no == var_no:
                if result == -1:
                    # This is the first condition on var_no.
                    result = cond_val
                elif cond_val != result:
                    # We have contradictory conditions on var_no.
                    return None
        return result

    for op in task.operators:
        conditions = dict(op.get_applicability_conditions())
        for var_no, _, post, cond in op.pre_post:
            effective_pre = get_effective_pre(var_no, conditions, cond)
            if effective_pre is not None:
                add_arc(var_no, effective_pre, post)
    for axiom in task.axioms:
        var_no, val = axiom.effect
        add_arc(var_no, -1, val)

    return dtgs


always_false = object()
always_true = object()


class Impossible(Exception):
    pass


class TriviallySolvable(Exception):
    pass


class DoesNothing(Exception):
    pass


class VarValueRenaming:
    def __init__(self):
        self.new_var_nos = []  # indexed by old var_no
        self.new_values = []  # indexed by old var_no and old value
        self.new_sizes = []  # indexed by new var_no
        self.new_var_count = 0
        self.num_removed_values = 0

    def dump(self):
        old_var_count = len(self.new_var_nos)
        print("variable count: %d => %d" % (old_var_count, self.new_var_count))
        print("number of removed values: %d" % self.num_removed_values)
        print("variable conversions:")
        for old_var_no, (new_var_no, new_values) in enumerate(zip(self.new_var_nos, self.new_values)):
            old_size = len(new_values)
            if new_var_no is None:
                print("variable %d [size %d] => removed" % (old_var_no, old_size))
            else:
                new_size = self.new_sizes[new_var_no]
                print("variable %d [size %d] => %d [size %d]" % (old_var_no, old_size, new_var_no, new_size))
            for old_value, new_value in enumerate(new_values):
                if new_value is always_false:
                    new_value = "always false"
                elif new_value is always_true:
                    new_value = "always true"
                print("    value %d => %s" % (old_value, new_value))

    def register_variable(self, old_domain_size, init_value, new_domain):
        assert 1 <= len(new_domain) <= old_domain_size
        assert init_value in new_domain
        if len(new_domain) == 1:
            # Remove this variable completely.
            new_values_for_var = [always_false] * old_domain_size
            new_values_for_var[init_value] = always_true
            self.new_var_nos.append(None)
            self.new_values.append(new_values_for_var)
            self.num_removed_values += old_domain_size
        else:
            new_value_counter = count()
            new_values_for_var = []
            for value in range(old_domain_size):
                if value in new_domain:
                    new_values_for_var.append(next(new_value_counter))
                else:
                    self.num_removed_values += 1
                    new_values_for_var.append(always_false)
            new_size = next(new_value_counter)
            assert new_size == len(new_domain)

            self.new_var_nos.append(self.new_var_count)
            self.new_values.append(new_values_for_var)
            self.new_sizes.append(new_size)
            self.new_var_count += 1

    def apply_to_task(self, task):
        if DEBUG:
            self.dump()
        self.apply_to_variables(task.variables)
        self.apply_to_mutexes(task.mutexes)
        self.apply_to_init(task.init)
        self.apply_to_goals(task.goal.pairs)
        self.apply_to_operators(task.operators)
        self.apply_to_axioms(task.axioms)

    def apply_to_variables(self, variables):
        variables.ranges = self.new_sizes
        new_axiom_layers = [None] * self.new_var_count
        for old_no, new_no in enumerate(self.new_var_nos):
            if new_no is not None:
                new_axiom_layers[new_no] = variables.axiom_layers[old_no]
        assert None not in new_axiom_layers
        variables.axiom_layers = new_axiom_layers
        self.apply_to_value_names(variables.value_names)

    def apply_to_value_names(self, value_names):
        new_value_names = [[None] * size for size in self.new_sizes]
        for var_no, values in enumerate(value_names):
            for value, value_name in enumerate(values):
                new_var_no, new_value = self.translate_pair((var_no, value))
                if new_value is always_true:
                    if DEBUG:
                        print("Removed true proposition: %s" % value_name)
                elif new_value is always_false:
                    if DEBUG:
                        print("Removed false proposition: %s" % value_name)
                else:
                    new_value_names[new_var_no][new_value] = value_name
        assert all((None not in value_names) for value_names in new_value_names)
        value_names[:] = new_value_names

    def apply_to_mutexes(self, mutexes):
        new_mutexes = []
        for mutex in mutexes:
            new_facts = []
            for var, val in mutex.facts:
                new_var_no, new_value = self.translate_pair((var, val))
                if new_value is not always_true and new_value is not always_false:
                    new_facts.append((new_var_no, new_value))
            if len(new_facts) >= 2:
                mutex.facts = new_facts
                new_mutexes.append(mutex)
        mutexes[:] = new_mutexes

    def apply_to_init(self, init):
        init_pairs = list(enumerate(init.values))
        try:
            self.convert_pairs(init_pairs)
        except Impossible:
            assert False, "Initial state impossible? Inconceivable!"
        new_values = [None] * self.new_var_count
        for new_var_no, new_value in init_pairs:
            new_values[new_var_no] = new_value
        assert None not in new_values
        init.values = new_values

    def apply_to_goals(self, goals):
        # This may propagate Impossible up.
        self.convert_pairs(goals)
        if not goals:
            # We raise an exception because we do not consider a SAS+
            # task without goals well-formed. Our callers are supposed
            # to catch this and replace the task with a well-formed
            # trivially solvable task.
            raise TriviallySolvable

    def apply_to_operators(self, operators):
        new_operators = []
        num_removed = 0
        for op in operators:
            new_op = self.translate_operator(op)
            if new_op is None:
                num_removed += 1
                if DEBUG:
                    print("Removed operator: %s" % op.name)
            else:
                new_operators.append(new_op)
        print("%d operators removed" % num_removed)
        operators[:] = new_operators

    def apply_to_axioms(self, axioms):
        new_axioms = []
        num_removed = 0
        for axiom in axioms:
            try:
                self.apply_to_axiom(axiom)
            except (Impossible, DoesNothing):
                num_removed += 1
                if DEBUG:
                    print("Removed axiom:")
                    axiom.dump()
            else:
                new_axioms.append(axiom)
        print("%d axioms removed" % num_removed)
        axioms[:] = new_axioms

    def translate_operator(self, op):
        """Compute a new operator from op where the var/value renaming has
        been applied. Return None if op should be pruned (because it
        is always inapplicable or has no effect.)"""

        # We do not call this apply_to_operator, breaking the analogy
        # with the other methods, because it creates a new operator
        # rather than transforming in-place. The reason for this is
        # that it would be quite difficult to generate the operator
        # in-place.

        # This method is trickier than it may at first appear. For
        # example, pre_post values should be fully sorted (see
        # documentation in the sas_tasks module), and pruning effect
        # conditions from a conditional effects can break this sort
        # order. Recreating the operator from scratch solves this
        # because the pre_post entries are sorted by
        # SASOperator.__init__.

        # Also, when we detect a pre_post pair where the effect part
        # can never trigger, the precondition part is still important,
        # but may be demoted to a prevail condition. Whether or not
        # this happens depends on the presence of other pre_post
        # entries for the same variable. We solve this by computing
        # the sorting into prevail vs. preconditions from scratch, too.

        applicability_conditions = op.get_applicability_conditions()
        try:
            self.convert_pairs(applicability_conditions)
        except Impossible:
            # The operator is never applicable.
            return None
        conditions_dict = dict(applicability_conditions)
        new_prevail_vars = set(conditions_dict)

        new_pre_post = []
        for entry in op.pre_post:
            new_entry = self.translate_pre_post(entry, conditions_dict)
            if new_entry is not None:
                new_pre_post.append(new_entry)
                # Mark the variable in the entry as not prevailed.
                new_var = new_entry[0]
                new_prevail_vars.discard(new_var)

        if not new_pre_post:
            # The operator has no effect.
            return None
        new_prevail = sorted((var, value) for (var, value) in conditions_dict.items() if var in new_prevail_vars)
        return sas_tasks.SASOperator(name=op.name, prevail=new_prevail, pre_post=new_pre_post, cost=op.cost)

    def apply_to_axiom(self, axiom):
        # The following line may generate an Impossible exception,
        # which is propagated up.
        self.convert_pairs(axiom.condition)
        new_var, new_value = self.translate_pair(axiom.effect)
        # If the new_value is always false, then the condition must
        # have been impossible.
        assert new_value is not always_false
        if new_value is always_true:
            raise DoesNothing
        axiom.effect = new_var, new_value

    def translate_pre_post(self, pre_post_entry, conditions_dict):
        """Return a translated version of a pre_post entry.
        If the entry never causes a value change, return None.

        (It might seem that a possible precondition part of pre_post
        gets lost in this case, but pre_post entries that become
        prevail conditions are handled elsewhere.)

        conditions_dict contains all applicability conditions
        (prevail/pre) of the operator, already converted. This is
        used to detect effect conditions that can never fire.

        The method may assume that the operator remains reachable,
        i.e., that it does not have impossible preconditions, as these
        are already checked elsewhere.

        Possible cases:
        - effect is always_true => return None
        - effect equals prevailed value => return None
        - effect condition is impossible given operator applicability
          condition => return None
        - otherwise => return converted pre_post tuple
        """

        var_no, pre, post, cond = pre_post_entry
        new_var_no, new_post = self.translate_pair((var_no, post))

        if new_post is always_true:
            return None

        if pre == -1:
            new_pre = -1
        else:
            _, new_pre = self.translate_pair((var_no, pre))
        assert new_pre is not always_false, (
            "This function should only be called for operators " "whose applicability conditions are deemed possible."
        )

        if new_post == new_pre:
            return None

        new_cond = list(cond)
        try:
            self.convert_pairs(new_cond)
        except Impossible:
            # The effect conditions can never be satisfied.
            return None

        for cond_var, cond_value in new_cond:
            if cond_var in conditions_dict and conditions_dict[cond_var] != cond_value:
                # This effect condition is not compatible with
                # the applicability conditions.
                return None

        assert new_post is not always_false, (
            "if we survived so far, this effect can trigger "
            "(as far as our analysis can determine this), "
            "and then new_post cannot be always_false"
        )

        assert new_pre is not always_true, (
            "if this pre_post changes the value and can fire, " "new_pre cannot be always_true"
        )

        return new_var_no, new_pre, new_post, new_cond

    def translate_pair(self, fact_pair):
        (var_no, value) = fact_pair
        new_var_no = self.new_var_nos[var_no]
        new_value = self.new_values[var_no][value]
        return new_var_no, new_value

    def convert_pairs(self, pairs):
        # We call this convert_... because it is an in-place method.
        new_pairs = []
        for pair in pairs:
            new_var_no, new_value = self.translate_pair(pair)
            if new_value is always_false:
                raise Impossible
            elif new_value is not always_true:
                assert new_var_no is not None
                new_pairs.append((new_var_no, new_value))
        pairs[:] = new_pairs


def build_renaming(dtgs):
    renaming = VarValueRenaming()
    for dtg in dtgs:
        renaming.register_variable(dtg.size, dtg.init, dtg.reachable())
    return renaming


def filter_unreachable_propositions(sas_task):
    """We remove unreachable propositions and then prune variables
    with only one value.

    Examples of things that are pruned:
    - Constant propositions that are not detected in instantiate.py
      because instantiate.py only reasons at the predicate level, and some
      predicates such as "at" in Depot are constant for some objects
      (hoists), but not others (trucks).

      Example: "at(hoist1, distributor0)" and the associated variable
      in depots-01.

    - "none of those" values that are unreachable.

      Example: at(truck1, ?x) = <none of those> in depots-01.

    - Certain values that are relaxed reachable but detected as
      unreachable after SAS instantiation because the only operators
      that set them have inconsistent preconditions.

      Example: on(crate0, crate0) in depots-01.
    """

    if DEBUG:
        sas_task.validate()
    dtgs = build_dtgs(sas_task)
    renaming = build_renaming(dtgs)
    # apply_to_task may raise Impossible if the goal is detected as
    # unreachable or TriviallySolvable if it has no goal. We let the
    # exceptions propagate to the caller.
    renaming.apply_to_task(sas_task)
    print("%d propositions removed" % renaming.num_removed_values)
    if DEBUG:
        sas_task.validate()
