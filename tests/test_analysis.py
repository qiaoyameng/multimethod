"""Registry-only ambiguity analysis: provable, covered and undecidable overlaps."""
from array import array
from collections.abc import Iterable
from typing import Literal

import pytest

from multimethod import DispatchError, LiteralValue, multimethod, parametric, signature, subtype


def build(*pairs):
    """Build a multimethod directly from type tuples without calling functions."""
    registry = None
    for index, types in enumerate(pairs):
        def func(*args, index=index):
            return index

        func.__name__ = f"func{index}"
        types = types if isinstance(types, tuple) else (types,)
        if registry is None:
            registry = multimethod(func)
        registry[tuple(map(subtype, types))] = func
    return registry


def sigs(registry):
    return {key for key in registry if isinstance(key, signature)}


# registries which are provably ambiguous -------------------------------------


def test_union_ambiguity():
    registry = build((int, object), (object, int))
    report = registry.check()
    assert len(report.conflicts) == 1
    assert not report.unknown
    (conflict,) = report.conflicts
    assert conflict.witness == (int, int)
    # the witness really dispatches ambiguously at runtime
    with pytest.raises(DispatchError, match="2 methods"):
        registry(0, 0)


def test_literal_ambiguity():
    left = subtype(Literal["a", "b"])
    right = subtype(Literal["b", "c"])
    registry = build(left, right)
    report = registry.check()
    assert len(report.conflicts) == 1
    assert report.conflicts[0].witness == (LiteralValue("b"),)
    with pytest.raises(DispatchError, match="2 methods"):
        registry("b")


def test_container_empty_witness():
    registry = build(subtype(list[int]), subtype(list[str]))
    # element types are incomparable, but the empty list belongs to both
    report = registry.check()
    assert len(report.conflicts) == 1
    with pytest.raises(DispatchError):
        registry([])


def test_multiple_conflicts():
    registry = build((int, object), (object, int), (bytes, object))
    assert len(registry.check().conflicts) == 2


# registries where a third overload covers the intersection ------------------


def test_diamond_exact_cover():
    registry = build((int, object), (object, int), (int, int))
    report = registry.check()
    assert not report.conflicts and not report.unknown
    assert registry(0, 0) == 2


def test_diamond_partial_cover_still_ambiguous():
    # ``(str, str)`` only covers a corner of the ``Iterable`` intersection,
    # e.g. ``(list, list)`` remains ambiguous, so it must not be silenced.
    registry = build((Iterable, object), (object, Iterable), (str, str))
    report = registry.check()
    assert len(report.conflicts) == 1
    with pytest.raises(DispatchError, match="2 methods"):
        registry([], [])
    assert registry("", "") == 2


def test_literal_cover():
    registry = build(
        subtype(Literal["a", "b"]), subtype(Literal["b", "c"]), subtype(Literal["b"])
    )
    assert not registry.check().conflicts
    assert registry("b") == 2


def test_partial_cover_keeps_residual_intersection():
    class Base: ...

    class Child(Base): ...

    registry = build((Base, object), (object, Base), (Child, Child))
    report = registry.check()
    assert len(report.conflicts) == 1
    assert report.conflicts[0].witness == (Base, Base)


def test_comparable_signatures_are_ordered():
    assert not build(object, int).check().conflicts
    assert not build((int,), (subtype(int | str),)).check().conflicts


def test_disjoint_signatures():
    registry = build(subtype(Literal["a"]), int)
    report = registry.check()
    assert not report.conflicts and not report.unknown
    assert registry("a") == 0
    assert registry(0) == 1


def test_tuple_lengths_disjoint():
    registry = build(subtype(tuple[int, int]), subtype(tuple[int]))
    assert not registry.check().conflicts and not registry.check().unknown


# registries which cannot be proved safe --------------------------------------


def test_unrelated_plain_classes_unknown():
    class Base: ...

    registry = build(int, Base)
    report = registry.check()
    assert not report.conflicts
    assert len(report.unknown) == 1
    assert "future common subclass" in report.unknown[0].reason


def test_abc_unknown():
    class Base: ...

    registry = build(Iterable, Base)
    report = registry.check()
    assert not report.conflicts
    assert len(report.unknown) == 1


def test_parametric_unknown():
    ints = parametric(array, typecode="i")
    longs = parametric(array, typecode="l")
    registry = build(ints, longs)
    report = registry.check()
    assert len(report.unknown) == 1
    assert "parametric" in report.unknown[0].reason


def test_report_is_rendered():
    report = build((int, object), (object, int)).check()
    text = str(report)
    assert "ambiguous" in text and "<class 'int'>" in text


# registry updates ------------------------------------------------------------


def test_add_and_remove_registration():
    class Base: ...

    registry = build(object, Base)
    assert not registry.check().conflicts  # comparable, resolved

    def string_overload(value):
        return "str"

    registry[(str,)] = string_overload
    assert len(registry.check().unknown) == 1  # Base vs str, undecidable overlap
    del registry[(str,)]
    assert not registry.check().conflicts and not registry.check().unknown


def test_duplicate_function_registration_is_safe():
    @multimethod
    def func(arg: int): ...

    @func.register
    def _(arg: int, extra: float = 0.0):
        return 0  # different arity, no intersection at equal specificity

    assert not func.check().conflicts and not func.check().unknown


def test_analysis_does_not_invoke_predicates():
    calls = []

    def predicate(value):
        calls.append(value)
        return True

    first = parametric(array, predicate)
    second = parametric(array, typecode="i")
    build(first, second).check()
    assert not calls
