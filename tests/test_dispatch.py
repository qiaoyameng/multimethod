import sys
from concurrent import futures
from collections.abc import Iterable, Mapping, Sequence

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


def test_strict_containers():
    @multimethod(strict_containers=True)
    def func(arg):
        return "object"

    @func.register
    def _(arg: list[int]):
        return "int"

    @func.register
    def _(arg: list[str]):
        return "str"

    @func.register
    def _(arg: list[int | str]):
        return "mixed"

    @func.register
    def _(arg: tuple[int, str]):
        return "tuple"

    @func.register
    def _(arg: tuple[bool, ...]):
        return "bools"

    @func.register
    def _(arg: dict[str, int | str]):
        return "dict"

    @func.register
    def _(arg: set[int | str]):
        return "set"

    @func.register
    def _(arg: Sequence[float]):
        return "floats"

    assert func([0, 1]) == "int"
    assert func(["a", "b"]) == "str"
    assert func([0, "a"]) == "mixed"
    assert func([0, 1.0]) == "object"
    with pytest.raises(DispatchError, match="3 methods"):
        func([])
    assert func((0, "a")) == "tuple"
    assert func((0, "a", ())) == "object"
    assert func((True, False)) == "bools"
    assert func((True, 0)) == "object"
    assert func({"a": 0, "b": "c"}) == "dict"
    assert func({0: 0}) == "object"
    assert func({0, "a"}) == "set"
    assert func({0, "a", b"x"}) == "object"
    assert func(iter([0])) == "object"
    assert func(value for value in [0]) == "object"
    assert func((True, "a")) == "tuple"
    assert func((0.0, 1.0)) == "floats"

    reused = [0]
    assert func(reused) == "int"
    reused[0] = "a"
    assert func(reused) == "str"


def test_strict_nested_union_and_empty_containers():
    @multimethod(strict_containers=True)
    def func(
        arg: list[int | str]
        | Mapping[str, list[int | None]]
        | tuple[list[int], ...],
    ):
        return "matched"

    @func.register
    def _(arg: list[float] | dict[str, list[bool]] | tuple[list[float], ...]):
        return "other"

    assert func([0, "a"]) == "matched"
    assert func([0.0, 1.0]) == "other"
    assert func({"a": [0, None], "b": []}) == "matched"
    assert func({"a": [0, None], "b": [None]}) == "matched"
    with pytest.raises(DispatchError, match="0 methods"):
        func({"a": [1.0]})
    with pytest.raises(DispatchError, match="0 methods"):
        func({0: [0]})
    assert func(([0, 1], [2, 3])) == "matched"
    assert func(([0.0],)) == "other"
    with pytest.raises(DispatchError, match="2 methods"):
        func([])
    with pytest.raises(DispatchError, match="2 methods"):
        func({})
    with pytest.raises(DispatchError, match="2 methods"):
        func(())
    with pytest.raises(DispatchError, match="2 methods"):
        func({"a": []})


def test_strict_recursive_containers():
    @multimethod(strict_containers=True)
    def func(arg: list[int]):
        return "ints"

    @func.register
    def _(arg: dict[str, str]):
        return "strings"

    recursive_list = []
    recursive_list.append(recursive_list)
    recursive_dict = {}
    recursive_dict["a"] = recursive_dict
    with pytest.raises(DispatchError):
        func(recursive_list)
    with pytest.raises(DispatchError):
        func(recursive_dict)

    if (3, 12) <= sys.version_info:
        namespace = {}
        exec(
            """
type Json = int | list[Json]


def values(arg: Json):
    return arg
""",
            namespace,
        )
        json_func = multimethod(namespace["values"], strict_containers=True)
        assert json_func([0, [1, []]]) == [0, [1, []]]
        with pytest.raises(DispatchError):
            json_func([0, ["a"]])
        recursive_json = []
        recursive_json.append(recursive_json)
        assert json_func(recursive_json) is recursive_json


def test_strict_multidispatch_keywords():
    @multidispatch(strict_containers=True)
    def func(arg):
        return "object"

    @func.register
    def _(arg: list[int]):
        return "int"

    @func.register
    def _(arg: list[str]):
        return "str"

    assert func(arg=[0]) == "int"
    assert func(arg=["a"]) == "str"
    with pytest.raises(DispatchError, match="2 methods"):
        func(arg=[])
    reused = [0]
    assert func(arg=reused) == "int"
    reused[0] = "a"
    assert func(arg=reused) == "str"
