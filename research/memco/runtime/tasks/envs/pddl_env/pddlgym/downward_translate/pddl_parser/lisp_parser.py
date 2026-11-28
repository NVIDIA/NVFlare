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

__all__ = ["ParseError", "parse_nested_list"]


class ParseError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value


# Basic functions for parsing PDDL (Lisp) files.
def parse_nested_list(input_file):
    tokens = tokenize(input_file)
    next_token = next(tokens)
    if next_token != "(":
        raise ParseError("Expected '(', got %s." % next_token)
    result = list(parse_list_aux(tokens))
    for tok in tokens:  # Check that generator is exhausted.
        raise ParseError("Unexpected token: %s." % tok)
    return result


def tokenize(input):
    for line in input:
        line = line.split(";", 1)[0]  # Strip comments.
        try:
            line.encode("ascii")
        except UnicodeEncodeError:
            raise ParseError("Non-ASCII character outside comment: %s" % line[0:-1])
        line = line.replace("(", " ( ").replace(")", " ) ").replace("?", " ?")
        for token in line.split():
            yield token.lower()


def parse_list_aux(tokenstream):
    # Leading "(" has already been swallowed.
    while True:
        try:
            token = next(tokenstream)
        except StopIteration:
            raise ParseError("Missing ')'")
        if token == ")":
            return
        elif token == "(":
            yield list(parse_list_aux(tokenstream))
        else:
            yield token
