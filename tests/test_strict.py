from collections.abc import Iterable, Mapping
from typing import Union

import pytest

from multimethod import DispatchError, multimethod


def test_list():
    @multimethod
    def func(arg: list[int]):
        return int

    @func.register
    def _(arg: list[str]):
        return str

    func.strict = True
    assert func([1, 2]) is int
    assert func(["a", "b"]) is str
    with pytest.raises(DispatchError, match="0 methods"):  # mixed elements
        func([1, "a"])
    with pytest.raises(DispatchError, match="2 methods"):  # empty: no element evidence
        func([])


def test_tuple():
    @multimethod
    def func(arg: tuple[int, str]):
        return "fixed"

    @func.register
    def _(arg: tuple[int, ...]):
        return "variadic"

    func.strict = True
    assert func((0, "")) == "fixed"
    assert func((0, 1)) == "variadic"
    assert func((0, 1, 2)) == "variadic"
    assert func(()) == "variadic"
    with pytest.raises(DispatchError, match="0 methods"):
        func((0, 1.0))
    with pytest.raises(DispatchError, match="0 methods"):
        func((0, 1, ""))


def test_dict():
    @multimethod
    def func(arg: dict[str, list[int]]):
        return int

    @func.register
    def _(arg: dict[str, list[str]]):
        return str

    func.strict = True
    assert func({"a": [1, 2]}) is int
    assert func({"a": ["b"]}) is str
    with pytest.raises(DispatchError, match="0 methods"):  # mixed nested elements
        func({"a": [1, "b"]})
    with pytest.raises(DispatchError, match="0 methods"):  # wrong key type
        func({0: [1]})
    with pytest.raises(DispatchError, match="2 methods"):  # empty: no evidence
        func({})
    with pytest.raises(DispatchError, match="2 methods"):  # empty nested container
        func({"a": []})


def test_set():
    @multimethod
    def func(arg: set[int]):
        return int

    @func.register
    def _(arg: set[str]):
        return str

    func.strict = True
    assert func({1, 2}) is int
    assert func({"a"}) is str
    with pytest.raises(DispatchError, match="0 methods"):
        func({1, "a"})
    with pytest.raises(DispatchError, match="2 methods"):
        func(set())


def test_nested_union():
    @multimethod
    def func(arg: list[Union[int, str]]):
        return "union"

    @func.register
    def _(arg: list[int]):
        return int

    @func.register
    def _(arg: dict[str, Union[list[int], list[str]]]):
        return "nested"

    func.strict = True
    assert func([1, "a"]) == "union"
    assert func([1, 2]) is int  # most specific of multiple matches
    assert func({"a": [1]}) == "nested"
    assert func({"a": ["b"]}) == "nested"
    with pytest.raises(DispatchError, match="0 methods"):
        func([1.0])
    with pytest.raises(DispatchError, match="0 methods"):
        func({"a": [1.0]})


def test_recursive():
    @multimethod
    def func(arg: list[int]):
        return int

    @func.register
    def _(arg: list[list]):
        return list

    func.strict = True
    data = []
    data.append(data)
    assert func(data) is list  # terminates: self-reference is a list
    data.append(1)
    with pytest.raises(DispatchError, match="0 methods"):  # terminates: mixed content
        func(data)


def test_iterator():
    @multimethod
    def func(arg: Iterable[int]):
        return int

    @func.register
    def _(arg: Iterable[str]):
        return str

    func.strict = True
    assert func([1, 2]) is int
    gen = (x for x in [1, 2])
    with pytest.raises(DispatchError, match="2 methods"):  # falls back to runtime type
        func(gen)
    assert list(gen) == [1, 2]  # not consumed by dispatch

    @multimethod
    def single(arg: Iterable[int]):
        return int

    single.strict = True
    gen = (x for x in [1])
    assert single(gen) is int
    assert list(gen) == [1]


def test_mutation():
    @multimethod
    def func(arg: list[int]):
        return int

    @func.register
    def _(arg: list[str]):
        return str

    func.strict = True
    data = [1, 2]
    assert func(data) is int
    data.append("a")  # same object, new content: re-evaluated, not cached by type
    with pytest.raises(DispatchError, match="0 methods"):
        func(data)
    data[:] = ["a", "b"]
    assert func(data) is str
    data.clear()
    with pytest.raises(DispatchError, match="2 methods"):
        func(data)


def test_candidate_set():
    @multimethod
    def func(arg: list[int]):
        return int

    @func.register
    def _(arg: list[str]):
        return str

    @func.register
    def _(arg: list[float]):
        return float

    func.strict = True
    assert func([1.0, 2.0]) is float
    assert func([1]) is int
    with pytest.raises(DispatchError, match="3 methods"):
        func([])


def test_mapping():
    @multimethod
    def func(arg: Mapping[str, int]):
        return int

    func.strict = True
    assert func({"a": 1}) is int
    with pytest.raises(DispatchError, match="0 methods"):
        func({"a": 1, "b": "c"})


def test_default_sampling():
    @multimethod
    def func(arg: list[int]):
        return int

    assert func.strict is False
    assert func([1, "a"]) is int  # sample-based default is unchanged
