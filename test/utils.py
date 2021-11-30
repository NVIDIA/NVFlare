import os
import unittest


quick_test_var = "QUICKTEST"


def test_is_quick():
    return os.environ.get(quick_test_var, "").lower() == "true"


def skip_if_quick(obj):
    """
    Skip the unit tests if environment variable `quick_test_var=true`.
    For example, the user can skip the relevant tests by setting ``export QUICKTEST=true``.
    """
    is_quick = test_is_quick()

    return unittest.skipIf(is_quick, "Skipping slow tests")(obj)
