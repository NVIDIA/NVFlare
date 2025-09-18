import inspect

from .constants import CollabMethodArgName


def check_optional_args(func, kwargs):
    signature = inspect.signature(func)
    parameter_names = signature.parameters.keys()

    # make sure to expose the optional args if the collab method supports them
    for n in [CollabMethodArgName.ABORT_SIGNAL, CollabMethodArgName.CONTEXT]:
        if n not in parameter_names:
            kwargs.pop(n, None)
