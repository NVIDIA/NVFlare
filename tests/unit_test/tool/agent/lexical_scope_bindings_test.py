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

from nvflare.tool.agent.frameworks.base import LexicalScopeBindings

MODULE = ()
OUTER = ("function:outer:1",)
INNER = ("function:outer:1", "function:inner:2")
CLASS_METHOD = ("class:Model:1", "function:train:2")


def test_candidate_scopes_walk_from_inner_function_to_module():
    bindings = LexicalScopeBindings()

    assert bindings.candidate_scopes("trainer", INNER) == [INNER, OUTER, MODULE]


def test_candidate_scopes_skip_class_scope_for_nested_function():
    bindings = LexicalScopeBindings()

    assert bindings.candidate_scopes("trainer", CLASS_METHOD) == [CLASS_METHOD, MODULE]


def test_global_declaration_resolves_and_binds_at_module_scope():
    bindings = LexicalScopeBindings()
    bindings.declare_scope(INNER, {"trainer"}, {"trainer"}, set())

    assert bindings.candidate_scopes("trainer", INNER) == [MODULE]
    assert bindings.binding_scope("trainer", INNER) == MODULE
    assert bindings.can_resolve_from_enclosing_scope("trainer", INNER)


def test_nonlocal_declaration_resolves_and_binds_at_existing_enclosing_scope():
    bindings = LexicalScopeBindings()
    bindings.declare_scope(OUTER, {"trainer"}, set(), set())
    bindings.declare_scope(INNER, {"trainer"}, set(), {"trainer"})
    bindings.bind("trainer", OUTER)

    assert bindings.candidate_scopes("trainer", INNER) == [INNER, OUTER]
    assert bindings.binding_scope("trainer", INNER) == OUTER
    assert bindings.can_resolve_from_enclosing_scope("trainer", INNER)


def test_local_binding_blocks_identity_from_enclosing_scope():
    bindings = LexicalScopeBindings()
    bindings.declare_scope(INNER, {"trainer"}, set(), set())
    identities = {(MODULE, "trainer")}

    assert not bindings.has_identity("trainer", INNER, identities)
    assert not bindings.can_resolve_from_enclosing_scope("trainer", INNER)


def test_class_scope_is_not_eligible_for_deferred_enclosing_resolution():
    bindings = LexicalScopeBindings()
    class_scope = ("class:Config:1",)

    assert not bindings.can_resolve_from_enclosing_scope("patch", class_scope)


def test_deferred_function_detection_covers_function_async_and_lambda_scopes():
    assert LexicalScopeBindings.has_deferred_function_scope(("function:run:1",))
    assert LexicalScopeBindings.has_deferred_function_scope(("async-function:run:1",))
    assert LexicalScopeBindings.has_deferred_function_scope(("lambda:<anonymous>:1",))
    assert not LexicalScopeBindings.has_deferred_function_scope(("class:Config:1",))


def test_binding_invalidates_set_and_mapping_identities_in_binding_scope():
    bindings = LexicalScopeBindings()
    set_identities = {(MODULE, "trainer")}
    mapping_identities = {(MODULE, "trainer"): "Trainer"}

    assert bindings.bind("trainer", MODULE, set_identities, mapping_identities) == MODULE
    assert set_identities == set()
    assert mapping_identities == {}


def test_nested_rebinding_shadows_but_does_not_delete_outer_identity():
    bindings = LexicalScopeBindings()
    bindings.declare_scope(INNER, {"trainer"}, set(), set())
    identities = {(MODULE, "trainer")}

    assert bindings.bind("trainer", INNER, identities) == INNER
    assert identities == {(MODULE, "trainer")}
    assert not bindings.has_identity("trainer", INNER, identities)
