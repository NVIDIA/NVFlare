#! /usr/bin/env python3
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
import sys
from functools import reduce

from . import pddl, timers


def convert_rules(prog):
    RULE_TYPES = {
        "join": JoinRule,
        "product": ProductRule,
        "project": ProjectRule,
    }
    result = []
    for rule in prog.rules:
        RuleType = RULE_TYPES[rule.type]
        new_effect, new_conditions = variables_to_numbers(rule.effect, rule.conditions)
        rule = RuleType(new_effect, new_conditions)
        rule.validate()
        result.append(rule)
    return result


def variables_to_numbers(effect, conditions):
    new_effect_args = list(effect.args)
    rename_map = {}
    for i, arg in enumerate(effect.args):
        if arg[0] == "?":
            rename_map[arg] = i
            new_effect_args[i] = i
    new_effect = pddl.Atom(effect.predicate, new_effect_args)

    # There are three possibilities for arguments in conditions:
    # 1. They are variables that occur in the effect. In that case,
    #    they are replaced by the corresponding position in the
    #    effect, as indicated by the rename_map.
    # 2. They are constants. In that case, the unifier must guarantee
    #    that they are matched appropriately. In that case, they are
    #    not modified (remain strings denoting objects).
    # 3. They are variables that don't occur in the effect (are
    #    projected away). This is only allowed in projection rules.
    #    Such arguments are also not modified (remain "?x" strings).

    new_conditions = []
    for cond in conditions:
        new_cond_args = [rename_map.get(arg, arg) for arg in cond.args]
        new_conditions.append(pddl.Atom(cond.predicate, new_cond_args))
    return new_effect, new_conditions


class BuildRule:
    def prepare_effect(self, new_atom, cond_index):
        effect_args = list(self.effect.args)
        cond = self.conditions[cond_index]
        for var_no, obj in zip(cond.args, new_atom.args):
            if isinstance(var_no, int):
                effect_args[var_no] = obj
        return effect_args

    def __str__(self):
        return "%s :- %s" % (self.effect, ", ".join(map(str, self.conditions)))

    def __repr__(self):
        return "<%s %s>" % (self.__class__.__name__, self)


class JoinRule(BuildRule):
    def __init__(self, effect, conditions):
        self.effect = effect
        self.conditions = conditions
        left_args = conditions[0].args
        right_args = conditions[1].args
        left_vars = {var for var in left_args if isinstance(var, int)}
        right_vars = {var for var in right_args if isinstance(var, int)}
        common_vars = sorted(left_vars & right_vars)
        self.common_var_positions = [
            [args.index(var) for var in common_vars] for args in (list(left_args), list(right_args))
        ]
        self.atoms_by_key = ({}, {})

    def validate(self):
        assert len(self.conditions) == 2, self
        left_args = self.conditions[0].args
        right_args = self.conditions[1].args
        eff_args = self.effect.args
        left_vars = {v for v in left_args if isinstance(v, int) or v[0] == "?"}
        right_vars = {v for v in right_args if isinstance(v, int) or v[0] == "?"}
        eff_vars = {v for v in eff_args if isinstance(v, int) or v[0] == "?"}
        assert left_vars & right_vars, self
        assert (left_vars | right_vars) == (left_vars & right_vars) | eff_vars, self

    def update_index(self, new_atom, cond_index):
        ordered_common_args = [new_atom.args[position] for position in self.common_var_positions[cond_index]]
        key = tuple(ordered_common_args)
        self.atoms_by_key[cond_index].setdefault(key, []).append(new_atom)

    def fire(self, new_atom, cond_index, enqueue_func):
        effect_args = self.prepare_effect(new_atom, cond_index)
        ordered_common_args = [new_atom.args[position] for position in self.common_var_positions[cond_index]]
        key = tuple(ordered_common_args)
        other_cond_index = 1 - cond_index
        other_cond = self.conditions[other_cond_index]
        for atom in self.atoms_by_key[other_cond_index].get(key, []):
            for var_no, obj in zip(other_cond.args, atom.args):
                if isinstance(var_no, int):
                    effect_args[var_no] = obj
            enqueue_func(self.effect.predicate, effect_args)


class ProductRule(BuildRule):
    def __init__(self, effect, conditions):
        self.effect = effect
        self.conditions = conditions
        self.atoms_by_index = [[] for c in self.conditions]
        self.empty_atom_list_no = len(self.conditions)

    def validate(self):
        assert len(self.conditions) >= 2, self
        cond_vars = [{v for v in cond.args if isinstance(v, int) or v[0] == "?"} for cond in self.conditions]
        all_cond_vars = reduce(set.union, cond_vars)
        eff_vars = {v for v in self.effect.args if isinstance(v, int) or v[0] == "?"}
        assert len(all_cond_vars) == len(eff_vars), self
        assert len(all_cond_vars) == sum([len(c) for c in cond_vars])

    def update_index(self, new_atom, cond_index):
        atom_list = self.atoms_by_index[cond_index]
        if not atom_list:
            self.empty_atom_list_no -= 1
        atom_list.append(new_atom)

    def _get_bindings(self, atom, cond):
        return [(var_no, obj) for var_no, obj in zip(cond.args, atom.args) if isinstance(var_no, int)]

    def fire(self, new_atom, cond_index, enqueue_func):
        if self.empty_atom_list_no:
            return

        # Binding: a (var_no, object) pair
        # Bindings: List-of(Binding)
        # BindingsFactor: List-of(Bindings)
        # BindingsFactors: List-of(BindingsFactor)
        bindings_factors = []
        for pos, cond in enumerate(self.conditions):
            if pos == cond_index:
                continue
            atoms = self.atoms_by_index[pos]
            assert atoms, "if we have no atoms, this should never be called"
            factor = [self._get_bindings(atom, cond) for atom in atoms]
            bindings_factors.append(factor)

        eff_args = self.prepare_effect(new_atom, cond_index)

        for bindings_list in itertools.product(*bindings_factors):
            bindings = itertools.chain(*bindings_list)
            for var_no, obj in bindings:
                eff_args[var_no] = obj
            enqueue_func(self.effect.predicate, eff_args)


class ProjectRule(BuildRule):
    def __init__(self, effect, conditions):
        self.effect = effect
        self.conditions = conditions

    def validate(self):
        assert len(self.conditions) == 1

    def update_index(self, new_atom, cond_index):
        pass

    def fire(self, new_atom, cond_index, enqueue_func):
        effect_args = self.prepare_effect(new_atom, cond_index)
        enqueue_func(self.effect.predicate, effect_args)


class Unifier:
    def __init__(self, rules):
        self.predicate_to_rule_generator = {}
        for rule in rules:
            for i, cond in enumerate(rule.conditions):
                self._insert_condition(rule, i)

    def unify(self, atom):
        result = []
        generator = self.predicate_to_rule_generator.get(atom.predicate)
        if generator:
            generator.generate(atom, result)
        return result

    def _insert_condition(self, rule, cond_index):
        condition = rule.conditions[cond_index]
        root = self.predicate_to_rule_generator.get(condition.predicate)
        if not root:
            root = LeafGenerator()
        constant_arguments = [
            (arg_index, arg)
            for (arg_index, arg) in enumerate(condition.args)
            if not isinstance(arg, int) and arg[0] != "?"
        ]
        newroot = root._insert(constant_arguments, (rule, cond_index))
        self.predicate_to_rule_generator[condition.predicate] = newroot

    def dump(self):
        predicates = sorted(self.predicate_to_rule_generator)
        print("Unifier:")
        for pred in predicates:
            print("    %s:" % pred)
            rule_gen = self.predicate_to_rule_generator[pred]
            rule_gen.dump("    " * 2)


class LeafGenerator:
    index = sys.maxsize

    def __init__(self):
        self.matches = []

    def empty(self):
        return not self.matches

    def generate(self, atom, result):
        result += self.matches

    def _insert(self, args, value):
        if not args:
            self.matches.append(value)
            return self
        else:
            root = LeafGenerator()
            root.matches.append(value)
            for arg_index, arg in args[::-1]:
                new_root = MatchGenerator(arg_index, LeafGenerator())
                new_root.match_generator[arg] = root
                root = new_root
            root.matches = self.matches  # can be swapped in C++
            return root

    def dump(self, indent):
        for match in self.matches:
            print("%s%s" % (indent, match))


class MatchGenerator:
    def __init__(self, index, next):
        self.index = index
        self.matches = []
        self.match_generator = {}
        self.next = next

    def empty(self):
        return False

    def generate(self, atom, result):
        result += self.matches
        generator = self.match_generator.get(atom.args[self.index])
        if generator:
            generator.generate(atom, result)
        self.next.generate(atom, result)

    def _insert(self, args, value):
        if not args:
            self.matches.append(value)
            return self
        else:
            arg_index, arg = args[0]
            if self.index < arg_index:
                self.next = self.next._insert(args, value)
                return self
            elif self.index > arg_index:
                new_parent = MatchGenerator(arg_index, self)
                new_branch = LeafGenerator()._insert(args[1:], value)
                new_parent.match_generator[arg] = new_branch
                return new_parent
            else:
                branch_generator = self.match_generator.get(arg)
                if not branch_generator:
                    branch_generator = LeafGenerator()
                self.match_generator[arg] = branch_generator._insert(args[1:], value)
                return self

    def dump(self, indent):
        for match in self.matches:
            print("%s%s" % (indent, match))
        for key in sorted(self.match_generator.keys()):
            print("%sargs[%s] == %s:" % (indent, self.index, key))
            self.match_generator[key].dump(indent + "    ")
        if not self.next.empty():
            assert isinstance(self.next, MatchGenerator)
            print("%s[*]" % indent)
            self.next.dump(indent + "    ")


class Queue:
    def __init__(self, atoms):
        self.queue = atoms
        self.queue_pos = 0
        self.enqueued = {(atom.predicate,) + tuple(atom.args) for atom in self.queue}
        self.num_pushes = len(atoms)

    def __bool__(self):
        return self.queue_pos < len(self.queue)

    __nonzero__ = __bool__

    def push(self, predicate, args):
        self.num_pushes += 1
        eff_tuple = (predicate,) + tuple(args)
        if eff_tuple not in self.enqueued:
            self.enqueued.add(eff_tuple)
            self.queue.append(pddl.Atom(predicate, list(args)))

    def pop(self):
        result = self.queue[self.queue_pos]
        self.queue_pos += 1
        return result


def compute_model(prog):
    with timers.timing("Preparing model"):
        rules = convert_rules(prog)
        unifier = Unifier(rules)
        # unifier.dump()
        fact_atoms = sorted(fact.atom for fact in prog.facts)
        queue = Queue(fact_atoms)

    print("Generated %d rules." % len(rules))
    with timers.timing("Computing model"):
        relevant_atoms = 0
        auxiliary_atoms = 0
        while queue:
            next_atom = queue.pop()
            pred = next_atom.predicate
            if isinstance(pred, str) and "$" in pred:
                auxiliary_atoms += 1
            else:
                relevant_atoms += 1
            matches = unifier.unify(next_atom)
            for rule, cond_index in matches:
                rule.update_index(next_atom, cond_index)
                rule.fire(next_atom, cond_index, queue.push)
    print("%d relevant atoms" % relevant_atoms)
    print("%d auxiliary atoms" % auxiliary_atoms)
    print("%d final queue length" % len(queue.queue))
    print("%d total queue pushes" % queue.num_pushes)
    return queue.queue


if __name__ == "__main__":
    import normalize
    import pddl_parser
    import pddl_to_prolog

    print("Parsing...")
    task = pddl_parser.open()
    print("Normalizing...")
    normalize.normalize(task)
    print("Writing rules...")
    prog = pddl_to_prolog.translate(task)

    model = compute_model(prog)
    for atom in model:
        print(atom)
    print("%d atoms" % len(model))
