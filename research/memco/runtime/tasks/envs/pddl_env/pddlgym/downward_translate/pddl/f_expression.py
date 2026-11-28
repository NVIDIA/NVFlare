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


class FunctionalExpression:
    def __init__(self, parts):
        self.parts = tuple(parts)

    def dump(self, indent="  "):
        print("%s%s" % (indent, self._dump()))
        for part in self.parts:
            part.dump(indent + "  ")

    def _dump(self):
        return self.__class__.__name__

    def instantiate(self, var_mapping, init_facts):
        raise ValueError("Cannot instantiate condition: not normalized")


class NumericConstant(FunctionalExpression):
    parts = ()

    def __init__(self, value):
        if value != int(value):
            raise ValueError("Fractional numbers are not supported")
        self.value = int(value)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.value == other.value

    def __str__(self):
        return "%s %s" % (self.__class__.__name__, self.value)

    def _dump(self):
        return str(self)

    def instantiate(self, var_mapping, init_facts):
        return self


class PrimitiveNumericExpression(FunctionalExpression):
    parts = ()

    def __init__(self, symbol, args):
        self.symbol = symbol
        self.args = tuple(args)
        self.hash = hash((self.__class__, self.symbol, self.args))

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.symbol == other.symbol and self.args == other.args

    def __str__(self):
        return "%s %s(%s)" % ("PNE", self.symbol, ", ".join(map(str, self.args)))

    def dump(self, indent="  "):
        print("%s%s" % (indent, self._dump()))

    def _dump(self):
        return str(self)

    def instantiate(self, var_mapping, init_assignments):
        args = [var_mapping.get(arg, arg) for arg in self.args]
        pne = PrimitiveNumericExpression(self.symbol, args)
        assert self.symbol != "total-cost"
        # We know this expression is constant. Substitute it by corresponding
        # initialization from task.
        result = init_assignments.get(pne)
        assert result is not None, "Could not find instantiation for PNE: %r" % (str(pne),)
        return result


class FunctionAssignment:
    def __init__(self, fluent, expression):
        self.fluent = fluent
        self.expression = expression

    def __str__(self):
        return "%s %s %s" % (self.__class__.__name__, self.fluent, self.expression)

    def dump(self, indent="  "):
        print("%s%s" % (indent, self._dump()))
        self.fluent.dump(indent + "  ")
        self.expression.dump(indent + "  ")

    def _dump(self):
        return self.__class__.__name__

    def instantiate(self, var_mapping, init_facts):
        if not (
            isinstance(self.expression, PrimitiveNumericExpression) or isinstance(self.expression, NumericConstant)
        ):
            raise ValueError("Cannot instantiate assignment: not normalized")
        # We know that this assignment is a cost effect of an action (for initial state
        # assignments, "instantiate" is not called). Hence, we know that the fluent is
        # the 0-ary "total-cost" which does not need to be instantiated
        assert self.fluent.symbol == "total-cost"
        fluent = self.fluent
        expression = self.expression.instantiate(var_mapping, init_facts)
        return self.__class__(fluent, expression)


class Assign(FunctionAssignment):
    def __str__(self):
        return "%s := %s" % (self.fluent, self.expression)


class Increase(FunctionAssignment):
    pass
