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

import random
import subprocess
import sys
import tempfile
from collections import defaultdict

from pddlgym.structs import Exists, ForAll, Literal, LiteralConjunction, LiteralDisjunction, Predicate
from pddlgym.utils import get_object_combinations


class PrologInterface:
    """ """

    def __init__(self, kb, conds, max_assignment_count=2, timeout=2, allow_redundant_variables=True, constants=None):
        if not isinstance(conds, list):
            conds = [conds]
        # Preprocess negative literals into renamed positive literals
        kb, conds = self._preprocess_negative_literals(kb, conds)
        self._kb = kb
        self._conds = conds
        self._cond_lits = self._get_lits_from_conds(conds)
        self._max_assignment_count = max_assignment_count
        self._allow_redundant_variables = allow_redundant_variables
        self._timeout = timeout
        self._varnames_to_var = self._create_varname_to_var(
            self._cond_lits, lambda x: self._clean_variable_name(x).lower()
        )
        self._atomname_to_atom = self._create_varname_to_var(self._kb, self._clean_atom_name)
        self._type_to_atomnames = defaultdict(list)
        for atom_name, atom in self._atomname_to_atom.items():
            self._type_to_atomnames[atom.var_type].append(atom_name)
        self._prolog_str = self._create_prolog_str()
        self._constants = constants  # unused now because variables begin with ? by convention
        # print(self._prolog_str)
        # import ipdb; ipdb.set_trace()

    @classmethod
    def _preprocess_negative_literals(cls, kb, conds):
        # Check for negated quantifiers, which we do not handle
        if any((isinstance(c, Exists) or isinstance(c, ForAll)) and c.is_negative for c in conds):
            raise NotImplementedError("We do not yet handle negated quantifiers")
        # Find all predicates with a negated literal in the conds
        negated_predicates = set()
        for cond in cls._get_lits_from_conds(conds):
            if cond.is_negative:
                negated_predicates.add(cond.predicate)
        if len(negated_predicates) == 0:
            return kb, conds
        # Start the new kb and conds
        kb = [lit for lit in kb]
        conds = [c for c in conds]
        # Sanity check
        assert all(str(p).startswith("Not") for p in negated_predicates)
        # Create positive predicates for the negated predicates
        negated_pred_to_pos_pred = {}
        for p in negated_predicates:
            # Prolog hands = specially
            if p.name == "=":
                pos_pred = Predicate("neg-eq", p.arity, p.var_types)
            else:
                pos_pred = Predicate(f"neg-{p.name}", p.arity, p.var_types)
            negated_pred_to_pos_pred[p] = pos_pred
        # TODO pass in objects separately
        objects = {o for lit in kb for o in lit.variables}
        # Get all instantiations of the new positive predicates
        for negated_pred, pos_pred in negated_pred_to_pos_pred.items():
            original_positive_pred = negated_pred.positive
            # Get all combinations of objects
            for objs in get_object_combinations(
                objects, arity=pos_pred.arity, var_types=pos_pred.var_types, allow_duplicates=True
            ):
                # Check whether the positive version is in the kb
                if original_positive_pred(*objs) in kb:
                    continue
                # Add the new positive literal to the kb
                kb.append(pos_pred(*objs))
            # Update the conds to include the positive pred instead of the negative pred
            conds = cls._replace_predicate(conds, negated_pred, pos_pred)
        return kb, conds

    @classmethod
    def _replace_predicate(cls, conds, from_pred, to_pred):
        if isinstance(conds, list):
            return [cls._replace_predicate(c, from_pred, to_pred) for c in conds]
        if isinstance(conds, Literal):
            if conds.predicate == from_pred:
                return to_pred(*conds.variables)
            return conds
        if isinstance(conds, LiteralConjunction):
            return LiteralConjunction(cls._replace_predicate(conds.literals, from_pred, to_pred))
        if isinstance(conds, LiteralDisjunction):
            return LiteralDisjunction(cls._replace_predicate(conds.literals, from_pred, to_pred))
        if isinstance(conds, ForAll):
            assert not conds.is_negative, "Negative universal quantification not implemented (use Exists instead)"
            return ForAll(
                cls._replace_predicate(conds.body, from_pred, to_pred), conds.variables, is_negative=conds.is_negative
            )
        if isinstance(conds, Exists):
            assert not conds.is_negative, "Negative exisential quantification not implemented (use ForAll instead)"
            return Exists(
                conds.variables, cls._replace_predicate(conds.body, from_pred, to_pred), is_negative=conds.is_negative
            )
        import ipdb

        ipdb.set_trace()
        raise NotImplementedError()

    @staticmethod
    def _get_lits_from_conds(conds):
        if isinstance(conds, list):
            return [lit for c in conds for lit in PrologInterface._get_lits_from_conds(c)]
        if isinstance(conds, Literal):
            return [conds]
        if hasattr(conds, "literals"):
            return PrologInterface._get_lits_from_conds(conds.literals)
        if hasattr(conds, "body"):
            return PrologInterface._get_lits_from_conds(conds.body)
        import ipdb

        ipdb.set_trace()
        raise NotImplementedError()

    @classmethod
    def _clean_atom_name(cls, atom_name):
        return atom_name.lower().replace("-", "_")

    @classmethod
    def _clean_variable_name(cls, var_name):
        var_name = var_name.replace("-", "_")
        if var_name.startswith("?"):
            return var_name.replace("?", "").capitalize()
        return var_name

    @classmethod
    def _clean_predicate_name(cls, predicate_name):
        if predicate_name == "=":
            return "predeq"
        return "pred" + predicate_name.lower().replace("-", "_")

    @staticmethod
    def _create_varname_to_var(lits, transformer):
        """ """
        vname_to_v = {}
        for lit in lits:
            for v in lit.variables:
                vname = transformer(v.name)
                if vname in vname_to_v:
                    assert vname_to_v[vname] == v
                else:
                    vname_to_v[vname] = v
        return vname_to_v

    def _create_prolog_str(self):
        """ """
        preamble = self._prolog_preamble(self._conds)
        type_str = self._prolog_type_str(self._kb)
        self._kb_str = self._prolog_kb_str(self._kb)  # can be changed by prolog_goal
        goal_str, variables = self._prolog_goal(self._conds, self._allow_redundant_variables)
        end = self._prolog_end(variables, self._max_assignment_count)
        return "\n".join([preamble, self._kb_str, type_str, goal_str, end])

    @classmethod
    def _prolog_kb_str(cls, kb):
        """ """
        kb_str = ""
        for lit in sorted(kb, key=lambda l: l.predicate.name):
            pred_name = cls._clean_predicate_name(lit.predicate.name)
            atoms = ",".join([cls._clean_atom_name(a) for a in lit.variables])
            kb_str += "\n{}({}).".format(pred_name, atoms)
        return kb_str

    @classmethod
    def _prolog_type_str(cls, kb):
        """ """
        all_atoms = sorted({v for lit in kb for v in lit.variables})
        type_str = ""
        for v in sorted(all_atoms, key=lambda v: v.var_type):
            vname = cls._clean_atom_name(v.name)
            type_str += "\nistype{}({}).".format(v.var_type, vname)
        return type_str

    def _prolog_goal(self, conds, allow_redundant_variables):
        """ """
        all_vars = sorted({v for lit in conds for v in self._get_variables(lit, set()) if v.startswith("?")})
        all_vars_cleaned = [self._clean_variable_name(v) for v in all_vars]
        main_cond_str = ""
        for lit in conds:
            pred_str = "\n\t" + self._prolog_goal_line(lit) + ","
            main_cond_str += pred_str
        type_cond_str = ""
        for v in sorted(all_vars, key=lambda v: v.var_type):
            type_cond_str += "\n\tistype{}({}),".format(v.var_type, self._clean_variable_name(v.name))
        if not allow_redundant_variables:
            all_different_str = "\n\tall_different([{}]).".format(",".join(all_vars_cleaned))
        else:
            type_cond_str = type_cond_str[:-1]
            all_different_str = "."
        head_str = "\ngoal({}) :-".format(",".join(all_vars_cleaned))
        final_str = head_str + main_cond_str + type_cond_str + all_different_str
        if final_str.endswith(",."):
            final_str = final_str[:-2] + "."
        return final_str, all_vars

    def _get_variables(self, lit, free_vars):
        if isinstance(lit, Literal):
            return {v for v in lit.variables if v not in free_vars}
        if isinstance(lit, (LiteralConjunction, LiteralDisjunction)):
            return {v for nested_lit in lit.literals for v in self._get_variables(nested_lit, free_vars)}
        if isinstance(lit, (ForAll, Exists)):
            for var in lit.variables:
                assert var not in free_vars
                free_vars.add(var)
            result = self._get_variables(lit.body, free_vars)
            for var in lit.variables:
                assert var in free_vars
                free_vars.remove(var)
            return result
        raise Exception("Unsupported lit: {}".format(lit))

    def _prolog_goal_line(self, lit):
        """ """
        if isinstance(lit, LiteralConjunction):
            inner_str = ",".join(self._prolog_goal_line(l) for l in lit.literals)
            return "({})".format(inner_str)
        if isinstance(lit, LiteralDisjunction):
            inner_str = ";".join(self._prolog_goal_line(l) for l in lit.literals)
            return "({})".format(inner_str)
        if lit.is_negative:
            raise NotImplementedError("Prolog behaves unexpectedly with negative literals")
            # pos_pred_str = self._prolog_goal_line(lit.positive)
            # pred_str = "\\+({})".format(pos_pred_str)
            # return pred_str
        if isinstance(lit, Literal):
            pred_name = self._clean_predicate_name(lit.predicate.name)
            variables = ",".join([self._clean_variable_name(a.name) for a in lit.variables])
            pred_str = "{}({})".format(pred_name, variables)
            return pred_str
        if isinstance(lit, ForAll):
            variables = ",".join([self._clean_variable_name(a.name) for a in lit.variables])
            assert len(lit.variables) == 1, "TODO: support ForAlls over multiple variables"
            variable = variables
            var_type = lit.variables[0].var_type
            objects_of_type = self._type_to_atomnames[var_type]
            objects_str = "[" + ",".join(objects_of_type) + "]"
            pred_str_body = self._prolog_goal_line(lit.body)
            pred_str = "foreach(member({}, {}), {})".format(variable, objects_str, pred_str_body)
            return pred_str
        if isinstance(lit, Exists):
            variables = ",".join([self._clean_variable_name(a.name) for a in self._get_variables(lit, set())])
            rand_num = random.randint(0, 1e6)
            body = self._prolog_goal_line(lit.body)
            self._kb_str += "\nhelper{}({}) :- {}.".format(rand_num, variables, body)
            pred_str = "helper{}({})".format(rand_num, variables)
            return pred_str
        raise NotImplementedError(lit)

    @classmethod
    def _prolog_preamble(cls, conds):
        cond_lits = cls._get_lits_from_conds(conds)
        pred_definitions = ""
        preds = set()
        for lit in cond_lits:
            preds.update(cls._get_predicates_from_literal(lit))
        preds = sorted(preds)
        for pred in preds:
            pred_name = cls._clean_predicate_name(pred.name)
            pred_definitions += "\n:- multifile({}/{}).".format(pred_name, pred.arity)

        return """print_solutions([]).
print_solutions([H|T]) :- write(H), nl, print_solutions(T).
:- style_check(-singleton).
{}
""".format(
            pred_definitions
        )

    @classmethod
    def _get_predicates_from_literal(cls, lit):
        if isinstance(lit, Literal):
            return {lit.predicate.positive}
        if isinstance(lit, LiteralConjunction):
            return {p for l in lit.literals for p in cls._get_predicates_from_literal(l)}
        if isinstance(lit, ForAll) or isinstance(lit, Exists):
            return cls._get_predicates_from_literal(lit.body)
        raise NotImplementedError()

    @classmethod
    def _prolog_end(cls, variables, max_assignment_count):
        lowercase_vars = ",".join([cls._clean_variable_name(v).lower() for v in variables])
        uppercase_vars = ",".join([cls._clean_variable_name(v).capitalize() for v in variables])
        return """
:- use_module(library(bounds)).
:- initialization (
    write([{0}]),
    nl,
    findnsols({1}, [{2}], goal({3}), L),
    print_solutions(L),
    halt).
""".format(
            lowercase_vars, max_assignment_count, uppercase_vars, uppercase_vars
        )

    def _parse_output_line(self, output_line):
        """ """
        output_line = output_line[1:-1]
        if output_line == "":
            return []
        return output_line.split(",")

    def run(self):
        """ """
        file = tempfile.NamedTemporaryFile(suffix=".pl")
        tmp_name = file.name
        with open(tmp_name, "w") as f:
            f.write(self._prolog_str)
        timeout_str = "gtimeout" if sys.platform == "darwin" else "timeout"
        cmd_str = "{} {} swipl {}".format(timeout_str, self._timeout, tmp_name)
        output = subprocess.getoutput(cmd_str)
        if "ERROR" in output or "Warning" in output:
            import ipdb

            ipdb.set_trace()
            raise Exception("Prolog terminated with an error: \n{}".format(output))
        lines = output.split("\n")
        varnames = self._parse_output_line(lines.pop(0))
        vs = [self._varnames_to_var[v] for v in varnames]
        bindings = lines
        if len(bindings) == 0:
            return []
        assignments = []
        for binding in bindings:
            atomnames = self._parse_output_line(binding)
            # Recover original (typed) atoms
            atoms = [self._atomname_to_atom[v] for v in atomnames]
            assignment = dict(zip(vs, atoms))
            assignments.append(assignment)
        return assignments
