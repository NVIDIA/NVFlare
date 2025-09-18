from .backend import Backend
from .ctx import Context
from .constants import CollabMethodArgName


class Proxy:

    def __init__(self, target_name, backend: Backend, caller_name: str):
        self.target_name = target_name
        self.backend = backend
        self.caller_name = caller_name

    @property
    def name(self):
        return self.target_name

    def get_target(self, name: str):
        obj = getattr(self, name, None)
        if not obj:
            return None
        if isinstance(obj, Proxy):
            return obj
        else:
            return None

    def __getattr__(self, func_name):
        """
        This method is called when Python cannot find an invoked method func_name of this class.
        """

        def method(*args, **kwargs):
            ctx = Context(self.caller_name, self.name)
            kwargs[CollabMethodArgName.CONTEXT] = ctx
            return self.backend.call_target(self.target_name, func_name, *args, **kwargs)

        return method
