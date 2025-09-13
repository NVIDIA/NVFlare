class Context:

    def __init__(self, caller: str, callee: str):
        self.caller = caller
        self.callee = callee

    def set_prop(self, name: str, value):
        setattr(self, name, value)
