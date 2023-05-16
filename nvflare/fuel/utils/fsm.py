# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.security.logging import secure_format_exception


class State(object):
    def __init__(self, name: str):
        assert isinstance(name, str), "name must be str but got {}".format(type(name))
        name = name.strip()
        assert len(name) > 0, "name must not be empty"
        self.name = name
        self.fsm = None

    def execute(self, **kwargs):
        pass

    def leave(self):
        pass

    def enter(self):
        pass


class FSM(object):

    STATE_NAME_EXIT = "__exit__"

    def __init__(self, name: str):
        self.name = name
        self.props = {}
        self.states = {}  # state name => State
        self.current_state = None
        self.error = None

    def set_prop(self, name, value):
        self.props[name] = value

    def get_prop(self, name, default=None):
        return self.props.get(name, default=default)

    def add_state(self, state: State):
        assert isinstance(state, State), "state must be State but got {}".format(type(state))
        s = self.states.get(state.name, None)
        assert s is None, 'duplicate state "{}"'.format(state.name)
        state.fsm = self
        self.states[state.name] = state

    def set_current_state(self, name: str):
        s = self.states.get(name)
        assert s, 'unknown state "{}"'.format(name)
        self.current_state = s

    def get_current_state(self):
        return self.current_state

    def execute(self, **kwargs) -> State:
        try:
            self.current_state = self._try_execute(**kwargs)
        except Exception as e:
            self.error = f"exception occurred in state execution: {secure_format_exception(e)}"
            self.current_state = None
        return self.current_state

    def _try_execute(self, **kwargs) -> State:
        assert self.current_state, "FSM has no current state"
        next_state_name = self.current_state.execute(**kwargs)
        if next_state_name:
            if next_state_name == FSM.STATE_NAME_EXIT:
                # go to the end
                return None

            # enter next state
            next_state = self.states.get(next_state_name, None)
            assert next_state, 'FSM has no such state "{}"'.format(next_state_name)

            # leave current state
            self.current_state.leave()

            # enter the next state
            next_state.enter()

            # change to the new state
            return next_state
        else:
            # stay in current state!
            return self.current_state
