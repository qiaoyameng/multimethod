from collections.abc import Iterable
from concurrent import futures

import pytest

from multimethod import DispatchError, multidispatch, multimethod, signature


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


def test_concurrency():
    @multimethod
    def func(arg: int): ...

    submit = futures.ThreadPoolExecutor().submit
    args = [type("", (int,), {})() for _ in range(500)]
    fs = [submit(func, arg) for arg in args]
    assert all(future.result() is None for future in fs)


def test_keyword_binding_equivalence():
    """Equivalent positional and keyword calls resolve identically."""

    @multidispatch
    def func(value, mode):
        return "default"

    @func.register
    def _(value: int, mode: str):
        return "int,str"

    @func.register
    def _(value: int, mode: float):
        return "int,float"

    for positional, keyword in (
        ((1, "x"), {"value": 1, "mode": "x"}),
        ((1, 1.0), {"value": 1, "mode": 1.0}),
    ):
        assert func(*positional) == func(**keyword)
        assert func(positional[0], **{list(keyword)[-1]: positional[-1]}) == func(*positional)
    assert func(1.0, "x") == func(value=1.0, mode="x") == "default"
    # unbound shapes raise dispatch errors, not a bare binding TypeError
    with pytest.raises(DispatchError, match="0 methods"):
        func(value=1)


def test_keyword_positional_only():
    """A positional-only base still allows keyword calls via other candidates."""

    def base(value, mode, /):
        return "base"

    func = multidispatch(base)

    @func.register
    def _(value: int, mode: str):
        return "int,str"

    assert func(1, "x") == func(value=1, mode="x") == "int,str"
    # calling the base function directly still rejects keyword arguments
    with pytest.raises(TypeError):
        base(value=1, mode="x")


def test_keyword_only_specialization():
    @multidispatch
    def func(x, *, mode):
        return "base"

    @func.register
    def _(x: int, *, mode: str):
        return "int+str"

    @func.register
    def _(x: int, *, mode: float):
        return "int+float"

    @func.register
    def _(x: str, *, mode: str):
        return "str+str"

    assert func(1, mode="x") == func(x=1, mode="x") == "int+str"
    assert func(x=1, mode=1.0) == "int+float"
    assert func(x="a", mode="x") == "str+str"


def test_keyword_defaults_participate():
    """Declared defaults are bound and omitted args don't hit other shapes."""

    @multidispatch
    def func(a, b=2.0):
        return ("base", b)

    @func.register
    def _(a: int, b: float = 2.0):
        return ("int,float", b)

    @func.register
    def _(a: int):
        return ("int-only",)

    assert func(1) == func(a=1) == ("int-only",)
    assert func(1, 2.0) == func(1, b=2.0) == func(a=1, b=2.0) == ("int,float", 2.0)
    assert func(a="x") == ("base", 2.0)


def test_keyword_shape_cache():
    """Cache keys distinguish binding shapes and ignore kwarg order."""

    @multidispatch
    def func(a, b=0):
        return "2-params"

    @func.register
    def _(a: int):
        return "1-param"

    @func.register
    def _(a: int, b: int = 0):
        return "2-int"

    calls = [
        ((1,), {}),
        ((1, 2), {}),
        ((), {"a": 1}),
        ((), {"a": 1, "b": 2}),
        ((1,), {"b": 2}),
    ]
    expected = ["1-param", "2-int", "1-param", "2-int", "2-int"]
    for repeat in range(3):  # second+ iterations hit the cache
        for (args, kwargs), result in zip(calls, expected):
            assert func(*args, **kwargs) == result

    @multidispatch
    def ordered(x=1, y="a", z=3.0):
        return "base"

    @ordered.register
    def _(x: int, y: str, z: float):
        return "all"

    assert ordered(x=1, y="a", z=2.0) == ordered(z=2.0, x=1, y="a") == "all"


def test_keyword_ambiguity():
    """Equally specific applicable candidates raise a dispatch error."""

    @multidispatch
    def func(x, *, flag="a"):
        return "base"

    @func.register
    def _(x: int, *, flag: str = "a"):
        return "str"

    @func.register
    def _(x: int, *, flag: int = 1):
        return "int"

    with pytest.raises(DispatchError, match="2 methods"):
        func(1)
    assert func(1, flag="z") == "str"
    assert func(1, flag=5) == "int"

    # same positional types, different binding shapes coexist and stay distinct

    @multidispatch
    def shaped(a, /, b):
        return "positional-only"

    @shaped.register
    def _(a, b):
        return "positional-or-keyword"

    assert sum(isinstance(key, signature) for key in shaped) == 2
    with pytest.raises(DispatchError, match="2 methods"):
        shaped(1, 2)
    assert shaped(a=1, b=2) == "positional-or-keyword"


def test_keyword_same_types_different_signatures():
    """Signatures with equal types but differing defaults are separate keys."""

    @multidispatch
    def func(x, *, mode="a"):
        return "a"

    @func.register
    def _(x: int, *, mode: str = "b"):
        return "b"

    assert sum(isinstance(key, signature) for key in func) == 2


def test_variadic_keyword_dispatch():
    @multidispatch
    def func(*xs, **kw):
        return ("base", xs, kw)

    @func.register
    def _(*xs: int, tag: str = "t"):
        return ("ints", xs, tag)

    assert func(1, 2, tag="z") == ("ints", (1, 2), "z")
    assert func(1, 2) == func(1, 2, tag="t") == ("ints", (1, 2), "t")
    assert func(1, "a") == ("base", (1, "a"), {})


def test_business_typeerror_propagates():
    """A TypeError raised inside the body is not treated as a dispatch error."""

    class BusinessError(TypeError):
        pass

    @multimethod
    def mm(x: int):
        raise BusinessError("body")

    with pytest.raises(BusinessError):
        mm(1)
    with pytest.raises(DispatchError, match="0 methods"):
        mm()

    @multimethod
    def needs_two(x: int, y):
        return y

    with pytest.raises(DispatchError, match="test_dispatch.py"):
        needs_two(1)  # dispatch matches (y untyped), the call itself has wrong arity

    @multidispatch
    def md(x):
        raise BusinessError("body")

    with pytest.raises(BusinessError):
        md(1)
    with pytest.raises(DispatchError, match="0 methods"):
        md(x=1, unexpected=2)
