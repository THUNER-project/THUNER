import pytest


def test_add(x=1):
    assert x + 1 == 2


def test_mult(x=1):
    assert x * 1 == 1


@pytest.mark.parametrize("test_input,expected", [("3+5", 8), ("2+4", 6), ("6*9", 54)])
def test_eval(test_input, expected):
    assert eval(test_input) == expected
