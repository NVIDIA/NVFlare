import threading


class State(object):

    def __init__(self, name: str):
        assert isinstance(name, str), 'name must be str but got {}'.format(type(name))
        name = name.strip()
        assert len(name) > 0, 'name must not be empty'
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
        self.states = {}   # state name => State
        self.lock = threading.Lock()
        self.current_state = None

    def add_state(self, state: State):
        assert isinstance(state, State), 'state must be State but got {}'.format(type(state))
        s = self.states.get(state.name, None)
        assert s is None, 'duplicate state "{}"'.format(state.name)
        state.fsm = self
        self.states[state.name] = state

    def set_current_state(self, name: str):
        with self.lock:
            s = self.states.get(name)
            assert s, 'unknown state "{}"'.format(name)
            self.current_state = s

    def get_current_state(self):
        with self.lock:
            return self.current_state

    def execute(self, **kwargs) -> State:
        with self.lock:
            assert self.current_state, 'FSM has no current state'
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
                self.current_state = next_state
            else:
                # stay in current state!
                pass

            return self.current_state
