import os
import pytest

# don't catch exceptions if _PYTEST_RAISE is set
# useful for IDE debugging tests
if os.getenv("_PYTEST_RAISE", "0") != "0":
    # https://stackoverflow.com/questions/62419998/how-can-i-get-pytest-to-not-catch-exceptions/62563106#62563106

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value
