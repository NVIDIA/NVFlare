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

# Renamed from types.py to avoid clash with stdlib module.
# In the future, use explicitly relative imports or absolute
# imports as a better solution.

import itertools


def _get_type_predicate_name(type_name):
    # PDDL allows mixing types and predicates, but some PDDL files
    # have name collisions between types and predicates. We want to
    # support both the case where such name collisions occur and the
    # case where types are used as predicates.
    #
    # We internally give types predicate names that cannot be confused
    # with non-type predicates. When the input uses a PDDL type as a
    # predicate, we automatically map it to this internal name.
    return "type@%s" % type_name


class Type:
    def __init__(self, name, basetype_name=None):
        self.name = name
        self.basetype_name = basetype_name

    def __str__(self):
        return self.name

    def __repr__(self):
        return "Type(%s, %s)" % (self.name, self.basetype_name)

    def get_predicate_name(self):
        return _get_type_predicate_name(self.name)


class TypedObject:
    def __init__(self, name, type_name):
        self.name = name
        self.type_name = type_name

    def __hash__(self):
        return hash((self.name, self.type_name))

    def __eq__(self, other):
        return self.name == other.name and self.type_name == other.type_name

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return "%s: %s" % (self.name, self.type_name)

    def __repr__(self):
        return "<TypedObject %s: %s>" % (self.name, self.type_name)

    def uniquify_name(self, type_map, renamings):
        if self.name not in type_map:
            type_map[self.name] = self.type_name
            return self
        for counter in itertools.count(1):
            new_name = self.name + str(counter)
            if new_name not in type_map:
                renamings[self.name] = new_name
                type_map[new_name] = self.type_name
                return TypedObject(new_name, self.type_name)

    def get_atom(self):
        # TODO: Resolve cyclic import differently.
        from . import conditions

        predicate_name = _get_type_predicate_name(self.type_name)
        return conditions.Atom(predicate_name, [self.name])
