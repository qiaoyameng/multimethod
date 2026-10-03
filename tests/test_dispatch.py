from collections.abc import Iterable
from concurrent import futures

import pytest

from multimethod import DispatchError, DispatchKey, multidispatch, multimethod, signature


def test_signature():
    with pytest.raises(TypeError):
        signature([list]) <= signature([None])


# roshambo
rock, paper, scissors = (type("", (), {}) for _ in range(3))


@multimethod
def roshambo(left, right):
    return "tie"


@roshambo.register(scissors, rock)
@roshambo.register(rock, scissors)
def _(left, right):
    return "rock smashes scissors"


@roshambo.register(paper, scissors)
@roshambo.register(scissors, paper)
def _(left, right):
    return "scissors cut paper"


@roshambo.register(rock, paper)
@roshambo.register(paper, rock)
def _(left, right):
    return "paper covers rock"


def test_roshambo():
    assert roshambo.__name__ == "roshambo"
    r, p, s = rock(), paper(), scissors()
    assert len(roshambo) == 7
    assert roshambo(r, p) == "paper covers rock"
    assert roshambo(p, r) == "paper covers rock"
    assert roshambo(r, s) == "rock smashes scissors"
    assert roshambo(p, s) == "scissors cut paper"
    assert roshambo(r, r) == "tie"
    assert len(roshambo) == 8
    del roshambo[()]
    del roshambo[rock, paper]
    assert len(roshambo) == 5
    with pytest.raises(DispatchError, match="0 methods"):
        roshambo(r, r)
    r = roshambo.copy()
    assert isinstance(r, multimethod)
    assert r == roshambo


# methods
class cls:
    method = multidispatch(lambda self, other: None)

    @method.register(Iterable, object)
    def _(self, other):
        return "left"

    @method.register(object, Iterable)
    def _(self, other):
        return "right"


def test_cls():
    obj = cls()
    assert obj.method(None) is cls.method(None, None) is None
    assert obj.method("") == "right"
    assert cls.method("", None) == "left"
    with pytest.raises(DispatchError, match="2 methods"):
        cls.method("", "")
    cls.method[object, Iterable] = cls.method[Iterable, object]
    assert cls.method("", "") == "left"


def test_arguments():
    def func(a, b: int, /, c: int = 0, d=None, *, f: int): ...

    assert signature.from_hints(func) == (object, int, int)

    @multidispatch
    def func(arg: ...): ...


def test_defaults():
    def func(a: int, b: float = 0.0):
        return b

    assert signature.from_hints(func) == (int, float)
    method = multimethod(func)
    assert method(1) == 0.0
    assert method(0, 1.0) == method(0, b=1.0) == 1.0
    with pytest.raises(DispatchError, match="0 methods"):
        method(0, 0)
    assert multidispatch(func)(0, b=1)
    assert multimethod(bool)(1)


@pytest.mark.benchmark
def test_keywords():
    @multidispatch
    def func(arg): ...

    @func.register
    def _(arg: int):
        return int

    @func.register
    def _(arg: int, extra: float):
        return float

    assert func(0) is func(arg=0) is int
    assert func(0, 0.0) is func(arg=0, extra=0.0) is float


def test_bound_keyword_dispatch():
    @multimethod
    def func(value: int, mode: str):
        return "int-str"

    @multimethod
    def func(value: str, mode: int):
        return "str-int"

    assert func(1, "x") == func(value=1, mode="x") == "int-str"
    assert func("x", 1) == func(value="x", mode=1) == "str-int"


def test_bound_defaults_and_keyword_only():
    @multimethod
    def func(value: int, mode: str = "a"):
        return "default-mode"

    @multimethod
    def func(value: int, mode: str = "b", *, flag: bool):
        return "keyword-only"

    assert func(1) is func(value=1) == "default-mode"
    assert func(1, flag=True) is func(value=1, flag=True) == "keyword-only"
    assert func(1, "x", flag=False) is func(value=1, mode="x", flag=False) == "keyword-only"
    assert func(1, "x") is func(value=1, mode="x") == "default-mode"

    @multimethod
    def ambiguous(value: int, mode: str = "a"):
        return "a"

    @multimethod
    def ambiguous(value: int, mode: str = "b"):
        return "b"

    with pytest.raises(DispatchError, match="2 methods"):
        ambiguous(1)

    @multimethod
    def valid_default(value: int, mode: str = "x"):
        return mode

    @valid_default.register
    def _(value: int, mode: str = 1):
        return mode

    assert valid_default(1) == "x"


def test_bound_varargs():
    @multimethod
    def func(value: int, *extra: float):
        return "floats"

    @multimethod
    def func(value: int, *extra: str, **options: bool):
        return "strings"

    assert func(1, 2.0, 3.0) == "floats"
    assert func(1, "x", "y", option=True) == "strings"


def test_bound_dispatch_ambiguity_and_cache_shape():
    @multimethod
    def func(value: int, mode: int, /):
        return "positional-only"

    @multimethod
    def func(value: int, *, mode: int):
        return "keyword-only"

    assert func(1, 2) == "positional-only"
    assert func(value=1, mode=2) == "keyword-only"
    assert func(mode=2, value=1) == "keyword-only"
    assert func(1, 2) == "positional-only"
    cached = [key for key in func if isinstance(key, DispatchKey)]
    assert len(cached) == 2

    @multimethod
    def other(value: int, mode: int):
        return "ordered"

    @multimethod
    def other(mode: int, value: int):
        return "reordered"

    with pytest.raises(DispatchError, match="2 methods"):
        other(1, 2)
    with pytest.raises(DispatchError, match="2 methods"):
        other(value=1, mode=2)
    with pytest.raises(DispatchError):
        other(mode=2, value=1)


def test_function_typeerror_propagates():
    @multimethod
    def func(value: int):
        return [value][1]

    with pytest.raises(IndexError):
        func(1)

    @multimethod
    def broken(value: int):
        raise TypeError("business error")

    with pytest.raises(TypeError, match="business error"):
        broken(1)


def test_concurrency():
    @multimethod
    def func(arg: int): ...

    submit = futures.ThreadPoolExecutor().submit
    args = [type("", (int,), {})() for _ in range(500)]
    fs = [submit(func, arg) for arg in args]
    assert all(future.result() is None for future in fs)
