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

import argparse
import sys


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("domain", help="path to domain pddl file")
    argparser.add_argument("task", help="path to task pddl file")
    argparser.add_argument(
        "--relaxed", dest="generate_relaxed_task", action="store_true", help="output relaxed task (no delete effects)"
    )
    argparser.add_argument(
        "--full-encoding",
        dest="use_partial_encoding",
        action="store_false",
        help="By default we represent facts that occur in multiple "
        "mutex groups only in one variable. Using this parameter adds "
        "these facts to multiple variables. This can make the meaning "
        "of the variables clearer, but increases the number of facts.",
    )
    argparser.add_argument(
        "--invariant-generation-max-candidates",
        default=100000,
        type=int,
        help="max number of candidates for invariant generation "
        "(default: %(default)d). Set to 0 to disable invariant "
        "generation and obtain only binary variables. The limit is "
        "needed for grounded input files that would otherwise produce "
        "too many candidates.",
    )
    argparser.add_argument(
        "--sas-file", default="output.sas", help="path to the SAS output file (default: %(default)s)"
    )
    argparser.add_argument(
        "--invariant-generation-max-time",
        default=300,
        type=int,
        help="max time for invariant generation (default: %(default)ds)",
    )
    argparser.add_argument(
        "--add-implied-preconditions",
        action="store_true",
        help="infer additional preconditions. This setting can cause a "
        "severe performance penalty due to weaker relevance analysis "
        "(see issue7).",
    )
    argparser.add_argument(
        "--keep-unreachable-facts",
        dest="filter_unreachable_facts",
        action="store_false",
        help="keep facts that can't be reached from the initial state",
    )
    argparser.add_argument(
        "--skip-variable-reordering",
        dest="reorder_variables",
        action="store_false",
        help="do not reorder variables based on the causal graph. Do not use "
        "this option with the causal graph heuristic!",
    )
    argparser.add_argument(
        "--keep-unimportant-variables",
        dest="filter_unimportant_vars",
        action="store_false",
        help="keep variables that do not influence the goal in the causal graph",
    )
    argparser.add_argument(
        "--dump-task", action="store_true", help="dump human-readable SAS+ representation of the task"
    )
    argparser.add_argument(
        "--layer-strategy",
        default="min",
        choices=["min", "max"],
        help="How to assign layers to derived variables. 'min' attempts to put as "
        "many variables into the same layer as possible, while 'max' puts each variable "
        "into its own layer unless it is part of a cycle.",
    )
    return argparser.parse_args()


def copy_args_to_module(args):
    module_dict = sys.modules[__name__].__dict__
    for key, value in vars(args).items():
        module_dict[key] = value


def setup():
    args = parse_args()
    copy_args_to_module(args)


# setup()
