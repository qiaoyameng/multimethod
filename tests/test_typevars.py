import sys

from typing import TypeVar

import pytest

from multimethod import DispatchError, multidispatch, multimethod


def test_same_concrete_type():
    variable = TypeVar("T")

    @multimethod
    def merge(left: variable, right: variable):
        return "generic"

    @merge.register
    def _(left: object, right: object):
        return "object"

    assert merge(1, 2) == "generic"
    assert merge(True, False) == "generic"
    assert merge(1, False) == "object"
    assert merge(False, 1) == "object"
    assert merge(1, "x") == "object"


def test_concrete_overloads_compare_after_unification():
    variable = TypeVar("T")

    @multimethod
    def merge(left: variable, right: variable):
        return "generic"

    @merge.register
    def _(left: int, right: int):
        return "int"

    @merge.register
    def _(left: bool, right: bool):
        return "bool"

    @merge.register
    def _(left: object, right: object):
        return "object"

    assert merge(1, 2) == "int"
    assert merge(True, False) == "bool"
    assert merge(1, False) == "int"
    assert merge(1, "x") == "object"


def test_generic_concrete_overloads_remain_comparable():
    variable = TypeVar("T")

    @multimethod
    def func(left: variable, right: variable):
        return "generic"

    @func.register
    def _(left: int, right: object):
        return "integer"

    assert func(1, 2) == "generic"
    assert func(True, False) == "generic"
    assert func(1, "x") == "integer"


def test_concrete_overloads_are_still_ambiguous():
    @multimethod
    def func(left: int, right: object):
        return "integer"

    @func.register
    def _(left: object, right: int):
        return "integer-right"

    with pytest.raises(DispatchError, match="2 methods found"):
        func(1, 1)


def test_subclass_and_bound_unification():
    class Base: ...

    class Sub(Base): ...

    bounded = TypeVar("T", bound=Base)

    @multimethod
    def func(left: bounded, right: bounded):
        return "same bound"

    @func.register
    def _(left: object, right: object):
        return "object"

    assert func(Base(), Base()) == "same bound"
    assert func(Sub(), Sub()) == "same bound"
    assert func(Sub(), Base()) == "object"


def test_concrete_base_overload_keeps_normal_inheritance():
    variable = TypeVar("T")

    class Base: ...

    class Sub(Base): ...

    @multimethod
    def func(left: variable, right: variable):
        return "generic"

    @func.register
    def _(left: Base, right: Base):
        return "base"

    @func.register
    def _(left: object, right: object):
        return "object"

    assert func(Sub(), Sub()) == "generic"
    assert func(Sub(), Base()) == "base"
    assert func(Sub(), 1) == "object"


def test_candidate_bindings_are_isolated():
    variable = TypeVar("T")

    def generic(left: variable, right: variable):
        return "repeated"

    @multimethod
    def isolated(left: object, right: object):
        return "object"

    isolated[variable, variable] = generic
    isolated[variable, int] = generic
    isolated[object, int] = generic

    assert isolated(True, 1) == "repeated"
    assert isolated(1, True) == "repeated"
    assert isolated(True, False) == "repeated"


def test_nested_conflict_and_empty_container():
    variable = TypeVar("T")

    @multimethod
    def merge(left: list[variable], right: list[variable]):
        return "generic"

    @merge.register
    def _(left: object, right: object):
        return "object"

    assert merge([1], [2]) == "generic"
    assert merge([True], [False]) == "generic"
    assert merge([1], [False]) == "object"
    assert merge([1], ["x"]) == "object"
    assert merge([1], [None]) == "object"
    assert merge([], []) == "generic"
    assert merge([1], []) == "generic"
    assert merge([], [1]) == "generic"

    assert merge([1], [False]) == "object"
    assert merge([1], [2]) == "generic"


def test_tuple_and_dict_nested_variables():
    variable = TypeVar("T")
    key_type = TypeVar("K")
    value_type = TypeVar("V")

    @multimethod
    def tupled(left: tuple[variable, variable], right: variable):
        return "tuple"

    @tupled.register
    def _(left: object, right: object):
        return "object"

    assert tupled((1, 2), 3) == "tuple"
    assert tupled((1, True), 3) == "object"

    @multimethod
    def mapped(mapping: dict[key_type, value_type], pair: tuple[key_type, value_type]):
        return "shared"

    @mapped.register
    def _(mapping: object, pair: object):
        return "object"

    assert mapped({1: "a"}, (1, "a")) == "shared"
    assert mapped({1: "a"}, (1, "b")) == "shared"
    assert mapped({1: "a"}, ("1", "a")) == "object"


def test_empty_concrete_container_overloads_are_ambiguous():
    variable = TypeVar("T")

    @multimethod
    def func(left: list[variable], right: list[variable]):
        return "generic"

    @func.register
    def _(left: list[int], right: list[int]):
        return "int"

    @func.register
    def _(left: list[bool], right: list[bool]):
        return "bool"

    @func.register
    def _(left: object, right: object):
        return "object"

    assert func([1], [2]) == "int"
    assert func([True], [False]) == "bool"
    with pytest.raises(DispatchError, match="2 methods found"):
        func([], [])


def test_bound_and_constrained_type_variables():
    bounded = TypeVar("T", bound=int)
    constrained = TypeVar("U", int, str)

    @multimethod
    def func(left: bounded, right: bounded):
        return "bounded"

    @func.register
    def _(left: constrained, right: constrained):
        return "constrained"

    @func.register
    def _(left: object, right: object):
        return "object"

    assert func(1, 2) == "bounded"
    assert func(True, False) == "bounded"
    assert func(True, 1) == "object"
    assert func(1.0, 1) == "object"
    assert func("a", "b") == "constrained"
    assert func("a", 1) == "object"


def test_dispatch_cache_uses_container_evidence():
    module = sys.modules["multimethod"]
    variable = TypeVar("T")

    @multimethod
    def func(left: list[variable], right: list[variable]):
        return "generic"

    @func.register
    def _(left: object, right: object):
        return "object"

    assert func([1], [False]) == "object"
    assert func([1], [2]) == "generic"
    assert func([1], [False]) == "object"

    keys = {key for key in func if isinstance(key, tuple) and key[0] is module.dispatch_cache_key}
    assert (module.dispatch_cache_key, (list, (int,)), (list, (bool,))) not in func
    assert (module.dispatch_cache_key, (list, int), (list, int)) in func
    assert (module.dispatch_cache_key, (list, int), (list, bool)) not in keys


def test_multidispatch_keeps_positional_and_keyword_unification():
    variable = TypeVar("T")

    @multidispatch
    def func(left, right):
        return "object"

    @func.register
    def _(left: variable, right: variable):
        return "same"

    assert func(1, 2) == "same"
    assert func(left=1, right=2) == "same"
    assert func(1, False) == "object"
    assert func(left=1, right=False) == "object"
