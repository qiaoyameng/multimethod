from collections.abc import Mapping
from typing import Literal, TypeVar

import pytest

from multimethod import DispatchError, generic_key, multimethod
from multimethod import multidispatch

T = TypeVar("T")
S = TypeVar("S")


class Parent:
    pass


class Child(Parent):
    pass


class Other(Parent):
    pass


def test_same_type_pair():
    @multimethod
    def merge(left: object, right: object):
        return "object"

    @merge.register
    def _(left: T, right: T):
        return "same"

    @merge.register
    def _(left: int, right: int):
        return "int"

    assert merge(1, 2) == "int"
    assert merge(True, False) == "same"
    assert merge("a", "b") == "same"
    assert merge(1, "a") == "object"
    assert merge(True, 1) == "int"
    assert merge(Child(), Child()) == "same"
    assert merge(Child(), Other()) == "object"


def test_bound_and_constrained_typevars():
    bound = TypeVar("bound", bound=Parent)
    constrained = TypeVar("constrained", int, str)

    @multimethod
    def func(left: object, right: object):
        return "object"

    @func.register
    def _(left: bound, right: bound):
        return "bound"

    @func.register
    def _(left: constrained, right: constrained):
        return "constrained"

    assert func(Child(), Child()) == "bound"
    assert func(Child(), Other()) == "object"
    assert func(1, 2) == "constrained"
    assert func("a", "b") == "constrained"
    assert func(True, False) == "constrained"
    assert func(1, "a") == "object"


def test_abstract_bound():
    bound = TypeVar("bound", bound=Mapping)

    @multimethod
    def func(arg: bound):
        return "mapping"

    @func.register
    def _(arg: object):
        return "object"

    assert func({"a": 1}) == "mapping"
    assert func([]) == "object"


def test_nested_conflict():
    @multimethod
    def func(arg: tuple[list[T], list[T]]):
        return "same"

    @func.register
    def _(arg: object):
        return "object"

    assert func(([1], [2])) == "same"
    assert func(([False], [True])) == "same"
    assert func(([1], [False])) == "object"
    assert func(([1], [1.0])) == "object"
    assert func(([], [1])) == "same"
    assert func(([], [])) == "same"

    @multimethod
    def empty(arg: list[T]):
        return "var"

    @empty.register
    def _(arg: list[bool]):
        return "bool"

    @empty.register
    def _(arg: list[int]):
        return "int"

    assert empty([False]) == "bool"
    assert empty([0]) == "int"
    assert empty([]) == "bool"
    assert empty([0.0]) == "var"


def test_shared_variables():
    @multimethod
    def func(left: dict[T, S], right: dict[T, S]):
        return "shared"

    @func.register
    def _(left: object, right: object):
        return "object"

    assert func({1: "a"}, {2: "b"}) == "shared"
    assert func({"a": 1}, {"b": 2}) == "shared"
    assert func({1: "a"}, {True: "b"}) == "object"
    assert func({1: "a"}, {2: 1}) == "object"
    assert func({1: "a", "b": 2}, {3: "c"}) == "object"
    assert func({}, {1: "a"}) == "shared"
    assert func({}, {}) == "shared"


def test_type_constructor_and_variadic_tuple():
    class Parent:
        pass

    class Child(Parent):
        pass

    @multimethod
    def constructor(cls: type[T], instance: T):
        return "same"

    @constructor.register
    def _(cls: object, instance: object):
        return "object"

    assert constructor(int, 1) == "same"
    assert constructor(bool, False) == "same"
    assert constructor(int, False) == "object"
    assert constructor(Parent, Child()) == "object"
    assert constructor(Child, Parent()) == "object"

    @multimethod
    def variadic(values: tuple[T, ...]):
        return "same"

    @variadic.register
    def _(values: object):
        return "object"

    assert variadic(()) == "same"
    assert variadic((1, 2, 3)) == "same"
    assert variadic((False, True)) == "same"
    assert variadic((1, False)) == "object"

    bounded = TypeVar("bounded", bound=int)

    @multimethod
    def bounded_variadic(values: tuple[bounded, ...]):
        return "bounded"

    @bounded_variadic.register
    def _(values: object):
        return "object"

    assert bounded_variadic((1, 2)) == "bounded"
    assert bounded_variadic(("x",)) == "object"


def test_specific_concrete_overload_ambiguity():
    bound = TypeVar("bound", bound=int)

    @multimethod
    def func(left: object, right: object):
        return "object"

    @func.register
    def _(left: bound, right: bound):
        return "bound"

    @func.register
    def _(left: int, right: int):
        return "int"

    with pytest.raises(DispatchError, match="2 methods"):
        func(1, 2)
    assert func(1.0, 2.0) == "object"


def test_candidates_use_independent_bindings():
    @multimethod
    def func(left: object, right: object):
        return "object"

    @func.register
    def _(left: S, right: S):
        return "same"

    assert func("a", "b") == "same"
    assert func(1, "a") == "object"

    constrained = TypeVar("constrained", int, float)

    @func.register
    def _(left: constrained, right: constrained):
        return "constrained"

    assert func(1, 2) == "constrained"
    assert func(1.0, 2.0) == "constrained"
    assert func(1, 1.0) == "object"


def test_equivalent_typevar_overloads_are_ambiguous():
    other = TypeVar("other")

    @multimethod
    def func(left: T, right: T):
        return "first"

    @func.register
    def _(left: other, right: other):
        return "second"

    with pytest.raises(DispatchError, match="2 methods"):
        func(1, 2)


def test_typevar_copy():
    @multimethod
    def func(arg: list[T]):
        return "var"

    copied = func.copy()
    assert copied([0]) == "var"
    assert copied.typevars
    assert copied.cacheable


def test_multidispatch_keywords():
    @multidispatch
    def func(left, right):
        return "object"

    @func.register
    def _(left: T, right: T):
        return "same"

    @func.register
    def _(left: int, right: int):
        return "int"

    assert func(1, right=2) == "int"
    assert func(True, right=False) == "same"
    assert func(1, right="a") == "object"


def test_generic_cache_contains_container_evidence():
    @multimethod
    def func(arg: list[T]):
        return "var"

    @func.register
    def _(arg: list[int]):
        return "int"

    @func.register
    def _(arg: list[bool]):
        return "bool"

    assert func([0]) == "int"
    assert func([False]) == "bool"
    assert func([]) == "bool"
    assert func([0]) == "int"

    cache_entries = [key for key in func if type(key) is generic_key]
    assert len(cache_entries) == 3
    assert len({key[-1] for key in cache_entries}) == 3


def test_predicate_bound_disables_caching():
    literal = TypeVar("literal", bound=Literal[0])

    @multimethod
    def func(arg: literal):
        return "literal"

    @func.register
    def _(arg: object):
        return "object"

    assert not func.cacheable
    assert func(0) == "literal"
    assert func(1) == "object"
    assert not [key for key in func if type(key) is generic_key]
