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

"""Implements PDDLEnv, a gym.Env parameterized by PDDL.

One PDDLEnv corresponds to one PDDL domain. Each episode corresponds to
one one PDDL problem; calling env.reset() sets up a new problem.

Observations are namedtuples with attributes `literals`, `objects`, `goal`.
Actions are single ground literals (not operators -- see README).

The debug_info returned by reset and step contains the domain PDDL file
and current problem PDDL file to facilitate interaction with a planner.

Usage example:
>>> env = PDDLEnv("pddl/sokoban.pddl", "pddl/sokoban")
>>> obs, debug_info = env.reset()
>>> action = env.action_space.sample()
>>> obs, reward, done, debug_info = env.step(action)
"""
import glob
import os
from itertools import product

import gym
import numpy as np
from pddlgym.inference import check_goal, find_satisfying_assignments
from pddlgym.parser import PDDLDomainParser, PDDLProblemParser
from pddlgym.spaces import LiteralActionSpace, LiteralSetSpace, LiteralSpace
from pddlgym.structs import Literal, LiteralConjunction, NoChange, ProbabilisticEffect, State, ground_literal


class InvalidAction(Exception):
    """See PDDLEnv docstring"""

    pass


def get_successor_state(
    state,
    action,
    domain,
    raise_error_on_invalid_action=False,
    inference_mode="infer",
    require_unique_assignment=True,
    get_all_transitions=False,
    return_probs=False,
):
    """
    Compute successor state(s) using operators in the domain

    Parameters
    ----------
    state : State
    action : Literal
    domain : PDDLDomain
    raise_error_on_invalid_action : bool
    inference_mode : "csp" or "prolog" or "infer"
    require_unique_assignment : bool
    get_all_transitions : bool
        If true, this function returns all possible successor states in the case that probabilistic effects exist in the domain.

    Returns
    -------
    next_state : State
    """
    selected_operator, assignment = _select_operator(
        state, action, domain, inference_mode=inference_mode, require_unique_assignment=require_unique_assignment
    )

    # A ground operator was found; execute the ground effects
    if assignment is not None:
        # Get operator effects
        if isinstance(selected_operator.effects, LiteralConjunction):
            effects = selected_operator.effects.literals
        else:
            assert isinstance(selected_operator.effects, Literal)
            effects = [selected_operator.effects]

        state = _apply_effects(
            state,
            effects,
            assignment,
            get_all_transitions,
            return_probs=return_probs,
        )

    # No operator was found
    elif raise_error_on_invalid_action:
        raise InvalidAction(f"called get_successor_state with invalid action '{action}' for given state")

    return state


def get_successor_states(
    state,
    action,
    domain,
    raise_error_on_invalid_action=False,
    inference_mode="infer",
    require_unique_assignment=True,
    return_probs=False,
):
    return get_successor_state(
        state,
        action,
        domain,
        raise_error_on_invalid_action,
        inference_mode,
        require_unique_assignment,
        get_all_transitions=True,
        return_probs=return_probs,
    )


def _select_operator(state, action, domain, inference_mode="infer", require_unique_assignment=True):
    """
    Helper for successor generation
    """
    if inference_mode == "infer":
        inference_mode = "csp" if _check_domain_for_strips(domain) else "prolog"

    if domain.operators_as_actions:
        # There should be only one possible operator if actions are operators
        possible_operators = set()
        for name, operator in domain.operators.items():
            if name.lower() == action.predicate.name.lower():
                assert len(possible_operators) == 0
                possible_operators.add(operator)
    else:
        # Possibly multiple operators per action
        possible_operators = set(domain.operators.values())

    # Knowledge base: literals in the state + action taken
    kb = set(state.literals) | {action}

    selected_operator = None
    assignment = None
    for operator in possible_operators:
        if isinstance(operator.preconds, Literal):
            conds = [operator.preconds]
        else:
            conds = operator.preconds.literals
        # Necessary for binding the operator arguments to the variables
        if domain.operators_as_actions:
            conds = [action.predicate(*operator.params)] + conds
        # Check whether action is in the preconditions
        action_literal = None
        for lit in conds:
            if lit.predicate == action.predicate:
                action_literal = lit
                break
        if action_literal is None:
            continue
        # For proving, consider action variable first
        action_variables = action_literal.variables

        def variable_sort_fn(variable):
            return (variable not in action_variables, variable)

        assignments = find_satisfying_assignments(
            kb,
            conds,
            variable_sort_fn=variable_sort_fn,
            type_to_parent_types=domain.type_to_parent_types,
            constants=domain.constants,
            mode=inference_mode,
        )
        num_assignments = len(assignments)
        if num_assignments > 0:
            if require_unique_assignment:
                assert num_assignments == 1, "Nondeterministic envs not supported"
            selected_operator = operator
            assignment = assignments[0]
            break

    return selected_operator, assignment


def _check_domain_for_strips(domain):
    """
    Check whether all operators in a domain are STRIPS
    """
    for operator in domain.operators.values():
        if not _check_struct_for_strips(operator.preconds):
            return False
    return True


def _check_struct_for_strips(struct):
    """
    Helper for _check_domain_for_strips
    """
    if isinstance(struct, Literal):
        return True
    if isinstance(struct, LiteralConjunction):
        return all(_check_struct_for_strips(l) for l in struct.literals)
    return False


def _compute_new_state_from_lifted_effects(lifted_effects, assignments, new_literals):
    for lifted_effect in lifted_effects:
        if lifted_effect == NoChange():
            continue
        effect = ground_literal(lifted_effect, assignments)
        # Negative effect
        if effect.is_anti:
            literal = effect.inverted_anti
            if literal in new_literals:
                new_literals.remove(literal)
    for lifted_effect in lifted_effects:
        if lifted_effect == NoChange():
            continue
        effect = ground_literal(lifted_effect, assignments)
        if not effect.is_anti:
            new_literals.add(effect)
    return new_literals


def _apply_effects(state, lifted_effects, assignments, get_all_transitions=False, return_probs=False):
    """
    Update a state given lifted operator effects and
    assignments of variables to objects.

    Parameters
    ----------
    state : State
        The state on which the effects are applied.
    lifted_effects : { Literal }
    assignments : { TypedEntity : TypedEntity }
        Maps variables to objects.
    get_all_transitions : bool
        If true, this function returns all possible successor states in the case that probabilistic effects exist in the domain.
    """
    new_literals = set(state.literals)
    determinized_lifted_effects = []
    # Handle probabilistic effects.

    # Each element of this list contain
    #   a pair of outcomes from a probabilistic effect
    probabilistic_lifted_effects = []
    for lifted_effect in lifted_effects:
        if isinstance(lifted_effect, ProbabilisticEffect):
            effect_outcomes = lifted_effect.literals
            probas = dict(zip(lifted_effect.literals, lifted_effect.probabilities))
            cur_probabilistic_lifted_effects = []

            if get_all_transitions:
                lifted_effects_list = cur_probabilistic_lifted_effects
            else:
                lifted_effects_list = determinized_lifted_effects
            sampled_effect = lifted_effect.sample()

            # If get_all_transitions == False, create list with sampled state only
            # Otherwise, populate it with possible outcomes
            effects_to_process = [sampled_effect] if not get_all_transitions else effect_outcomes

            for chosen_effect in effects_to_process:
                if isinstance(chosen_effect, LiteralConjunction):
                    for lit in chosen_effect.literals:
                        lifted_effects_list.append(lit)
                        lit.proba = probas[chosen_effect]
                else:
                    lifted_effects_list.append(chosen_effect)
                    chosen_effect.proba = probas[chosen_effect]

            if get_all_transitions:
                probabilistic_lifted_effects.append(cur_probabilistic_lifted_effects)
        else:
            determinized_lifted_effects.append(lifted_effect)

    states = []
    if not get_all_transitions:
        new_literals = _compute_new_state_from_lifted_effects(determinized_lifted_effects, assignments, new_literals)

        return state.with_literals(new_literals)

    # else - get all possible transitions

    # Construct combinations of probabilistic effects
    probabilistic_effects_combinations = list(product(*probabilistic_lifted_effects))

    states_to_probs = {}
    for prob_efs_combination in probabilistic_effects_combinations:
        total_proba = np.prod([lit.proba for lit in prob_efs_combination])
        if total_proba == 0:
            continue
        new_prob_literals = set(state.literals)
        new_determinized_lifted_effects = determinized_lifted_effects + list(prob_efs_combination)
        new_prob_literals = _compute_new_state_from_lifted_effects(
            new_determinized_lifted_effects, assignments, new_prob_literals
        )

        new_state = state.with_literals(new_prob_literals)
        if new_state in states_to_probs:
            # If there are multiple ways of reaching next state,
            #   then these probabilities have to be summed
            states_to_probs[new_state] += total_proba
        else:
            states_to_probs[new_state] = total_proba
        states.append(new_state)
    if return_probs:
        return states_to_probs
    # convert list of states to set
    return frozenset(states)


class PDDLEnv(gym.Env):
    """
    Parameters
    ----------
    domain_file : str
        Path to a PDDL domain file.
    problem_dir : str
        Path to a directory of PDDL problem files.
    render : fn or None
        An optional render function (obs -> img).
    seed : int
        Random seed used to sample new problems upon reset.
    raise_error_on_invalid_action : bool
        When an action is taken for which no operator's
        preconditions holds, raise InvalidAction() if True;
        otherwise silently make no changes to the state.
    operators_as_actions : bool
        If True, the PDDL operators are treated as the actions.
        Otherwise, actions must be specified separately in the PDDL file.
    dynamic_action_space : bool
        Let self.action_space dynamically change on each iteration to
        include only valid actions (must match operator preconditions).
    """

    def __init__(
        self,
        domain_file,
        problem_dir,
        render=None,
        seed=0,
        raise_error_on_invalid_action=False,
        operators_as_actions=False,
        dynamic_action_space=False,
    ):
        self._state = None
        self._domain_file = domain_file
        self._problem_dir = problem_dir
        self._render = render
        self.seed(seed)
        self._raise_error_on_invalid_action = raise_error_on_invalid_action
        self.operators_as_actions = operators_as_actions

        # Set by self.fix_problem_index
        self._problem_index_fixed = False

        self._problem_idx = None

        # Parse the PDDL files
        self.domain, self.problems = self.load_pddl(
            domain_file, problem_dir, operators_as_actions=self.operators_as_actions
        )

        # Determine if the domain is STRIPS
        self._domain_is_strips = _check_domain_for_strips(self.domain)
        self._inference_mode = "csp" if self._domain_is_strips else "prolog"

        # Initialize action space with problem-independent components
        actions = list(self.domain.actions)
        self.action_predicates = [self.domain.predicates[a] for a in actions]
        self._dynamic_action_space = dynamic_action_space
        if dynamic_action_space:
            if self.domain.operators_as_actions and self._domain_is_strips:
                self._action_space = LiteralActionSpace(
                    self.domain,
                    self.action_predicates,
                    type_hierarchy=self.domain.type_hierarchy,
                    type_to_parent_types=self.domain.type_to_parent_types,
                )
            else:
                self._action_space = LiteralSpace(
                    self.action_predicates,
                    lit_valid_test=self._action_valid_test,
                    type_hierarchy=self.domain.type_hierarchy,
                    type_to_parent_types=self.domain.type_to_parent_types,
                )

        else:
            self._action_space = LiteralSpace(
                self.action_predicates, type_to_parent_types=self.domain.type_to_parent_types
            )

        # Initialize observation space with problem-independent components
        self._observation_space = LiteralSetSpace(
            set(self.domain.predicates.values()) - set(self.action_predicates),
            type_hierarchy=self.domain.type_hierarchy,
            type_to_parent_types=self.domain.type_to_parent_types,
        )

    @staticmethod
    def load_pddl(domain_file, problem_dir, operators_as_actions=False):
        """
        Parse domain and problem PDDL files.

        Parameters
        ----------
        domain_file : str
            Path to a PDDL domain file.
        problem_dir : str
            Path to a directory of PDDL problem files.
        operators_as_actions : bool
            See class docstirng.

        Returns
        -------
        domain : PDDLDomainParser
        problems : [ PDDLProblemParser ]
        """
        domain = PDDLDomainParser(
            domain_file, expect_action_preds=(not operators_as_actions), operators_as_actions=operators_as_actions
        )
        problems = []
        problem_files = [f for f in glob.glob(os.path.join(problem_dir, "*.pddl"))]
        for problem_file in sorted(problem_files):
            problem = PDDLProblemParser(
                problem_file, domain.domain_name, domain.types, domain.predicates, domain.actions, domain.constants
            )
            problems.append(problem)
        return domain, problems

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def set_state(self, state):
        self._state = state

    def get_state(self):
        return self._state

    def seed(self, seed):
        self._seed = seed
        self.rng = np.random.RandomState(seed)

    def fix_problem_index(self, problem_idx):
        """
        Fix the PDDL problem used when reset is called.

        Useful for reproducible testing.

        The order of PDDL problems is determined by the names
        of their files. See PDDLEnv.load_pddl.

        Parameters
        ----------
        problem_idx : int
        """
        self._problem_idx = problem_idx
        self._problem_index_fixed = True

    def reset(self):
        """
        Set up a new PDDL problem and start a new episode.

        Note that the PDDL files are included in debug_info.

        Returns
        -------
        obs : { Literal }
            The set of active predicates.
        debug_info : dict
            See self._get_debug_info()
        """
        if not self._problem_index_fixed:
            self._problem_idx = self.rng.choice(len(self.problems))
        self._problem = self.problems[self._problem_idx]

        initial_state = State(
            frozenset(self._problem.initial_state), frozenset(self._problem.objects), self._problem.goal
        )
        initial_state = self._handle_derived_literals(initial_state)
        self.set_state(initial_state)

        self._goal = self._problem.goal
        debug_info = self._get_debug_info()

        self._action_space.reset_initial_state(initial_state)

        return self.get_state(), debug_info

    def _get_debug_info(self):
        """
        Contains the problem file and domain file
        for interaction with a planner.
        """
        info = {"problem_file": self._problem.problem_fname, "domain_file": self.domain.domain_fname}
        return info

    def step(self, action):
        """
        Execute an action and update the state.

        Tries to find a ground operator for which the
        preconditions hold when this action is taken. If none
        exist, optionally raises InvalidAction. If multiple
        exist, raises an AssertionError, since we assume
        deterministic environments only. Once the operator
        is found, the ground effects are executed to update
        the state.

        Parameters
        ----------
        action : Literal

        Returns
        -------
        state : State
            The set of active predicates.
        reward : float
            1 if the goal is reached and 0 otherwise.
        done : bool
            True if the goal is reached.
        debug_info : dict
            See self._get_debug_info.
        """
        state, reward, done, debug_info = self.sample_transition(action)
        self.set_state(state)
        return state, reward, done, debug_info

    def _get_new_state_info(self, state):
        state = self._handle_derived_literals(state)

        done = self._is_goal_reached(state)

        reward = self.extrinsic_reward(state, done)
        debug_info = self._get_debug_info()

        return state, reward, done, debug_info

    def sample_transition(self, action):
        state = self._get_successor_state(
            self._state,
            action,
            self.domain,
            inference_mode=self._inference_mode,
            raise_error_on_invalid_action=self._raise_error_on_invalid_action,
        )
        return self._get_new_state_info(state)

    def _get_successor_state(self, *args, **kwargs):
        """Separated out to allow for overrides in subclasses"""
        return get_successor_state(*args, **kwargs)

    def _get_successor_states(self, *args, **kwargs):
        """Separated out to allow for overrides in subclasses"""
        return get_successor_states(*args, **kwargs)

    def get_all_possible_transitions(self, action, return_probs=False):
        assert self.domain.is_probabilistic
        states = self._get_successor_states(
            self._state,
            action,
            self.domain,
            inference_mode=self._inference_mode,
            raise_error_on_invalid_action=self._raise_error_on_invalid_action,
            return_probs=return_probs,
        )
        if return_probs:
            return [(self._get_new_state_info(state), prob) for state, prob in states.items()]

        return [self._get_new_state_info(state) for state in states]

    def extrinsic_reward(self, state, done):
        if done:
            reward = 1.0
        else:
            reward = 0.0

        return reward

    def _is_goal_reached(self, state):
        """
        Check if the terminal condition is met, i.e., the goal is reached.
        """
        return check_goal(state, self._goal)

    def _action_valid_test(self, state, action):
        _, assignment = _select_operator(state, action, self.domain, inference_mode=self._inference_mode)
        return assignment is not None

    def render(self, *args, **kwargs):
        if self._render:
            return self._render(self._state.literals, *args, **kwargs)

    def _handle_derived_literals(self, state):
        # first remove any old derived literals since they're outdated
        to_remove = set()
        for lit in state.literals:
            if lit.predicate.is_derived:
                to_remove.add(lit)
        state = state.with_literals(state.literals - to_remove)

        # add negative basic literals for checking derived predicates
        state_literals = state.literals
        all_ground_literals = self._observation_space.all_ground_literals(state)
        for lit in all_ground_literals:
            if not lit.predicate.is_derived and lit not in state_literals:
                state_literals = {lit.negative} | state_literals

        while True:  # loop, because derived predicates can be recursive
            new_derived_literals = set()
            for pred in self.domain.predicates.values():
                if not pred.is_derived:
                    continue
                assignments = find_satisfying_assignments(
                    state_literals,
                    pred.body,
                    type_to_parent_types=self.domain.type_to_parent_types,
                    constants=self.domain.constants,
                    mode="prolog",
                    max_assignment_count=99999,
                )
                for assignment in assignments:
                    objects = [
                        assignment[param_type(param_name)]
                        for param_name, param_type in zip(pred.param_names, pred.var_types)
                    ]
                    derived_literal = pred(*objects)
                    if derived_literal not in state.literals:
                        new_derived_literals.add(derived_literal)
            if new_derived_literals:
                # update state_literals for recursive checking
                state_literals = state_literals | new_derived_literals
                # save derived literals in state
                state = state.with_literals(state.literals | new_derived_literals)
            else:  # terminate
                break
        return state
