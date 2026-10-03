"""Tests for registry-only static ambiguity analysis."""

import typing
from collections.abc import Hashable, Iterable, Mapping

import pytest

from multimethod import DispatchError, multimethod, parametric
from multimethod.analysis import AnalysisReport, analyze


def registry(*signatures):
    """Build a multimethod whose every key is intentional."""
    method = multimethod(lambda *args: None)
    del method[()]
    for index, types in enumerate(signatures):
        method[tuple(types)] = (lambda *args, _i=index: _i)
    return method


def confidences(method):
    return {finding.confidence for finding in analyze(method).findings}


# --------------------------------------------------------------------------- ambiguous
def test_true_ambiguity_two_arguments():
    method = registry(
        [object, int],
        [int, object],
        [object, object],
    )
    report = analyze(method)
    ambiguous = report.ambiguous
    assert len(ambiguous) == 1
    finding = ambiguous[0]
    assert finding.witness == (0, 0)
    # the static witness genuinely fails at runtime
    with pytest.raises(DispatchError):
        method(1, 1)


def test_literal_overlap_witness():
    method = registry(
        [typing.Literal["a", "b"]],
        [typing.Literal["b", "c"]],
    )
    finding = analyze(method).ambiguous[0]
    assert finding.witness == ("b",)
    with pytest.raises(DispatchError):
        method("b")


def test_existing_multiple_inheritance_is_described_not_constructed():
    created = []

    class Left:
        def __new__(cls):
            created.append("constructed")
            return object.__new__(cls)

    class Right:
        pass

    class Both(Left, Right):
        pass

    method = registry([Left], [Right])
    finding = analyze(method).ambiguous[0]
    assert isinstance(finding.witness[0], type)
    assert created == []  # described as a type, never instantiated


# --------------------------------------------------------------------------- covered
def test_diamond_covered_by_third_overload():
    method = registry(
        [object, object],
        [object, int],
        [int, object],
        [int, int],
    )
    report = analyze(method)
    assert not report.ambiguous
    covered = report.covered
    assert len(covered) == 1
    assert covered[0].witness is None
    assert method(1, 1) == 3


def test_single_inheritance_diamond_is_silent():
    # bool strictly refines int; no value reaches both object and int
    # without bool already being a separate, resolved sub-region.
    assert not analyze(registry([object], [int], [bool])).findings
    assert not analyze(registry([object], [bool])).findings


def test_union_arm_dominates():
    assert not analyze(registry([int], [int | str])).findings
    assert not analyze(registry([bool], [int])).findings


def test_empty_container_intersection_covered_by_bare_container():
    # [] matches both parameterizations but the bare list loses to both,
    # so runtime itself is ambiguous: the report must agree.
    method = registry([list[int]], [list[str]], [list])
    witnesses = [f.witness for f in analyze(method).ambiguous]
    assert witnesses and all(witness == ([],) for witness in witnesses)
    with pytest.raises(DispatchError):
        method([])
    method2 = registry([list[int]], [list[str]])
    assert len(analyze(method2).ambiguous) == 1
    assert analyze(method2).ambiguous[0].witness == ([],)


# --------------------------------------------------------------------------- unknown
def test_complex_abc_is_unknown_with_reason():
    # Iterable and Hashable overlap (e.g. str), but a full proof needs more
    # than the frozen ABC/class graph, so the report stays conservative.
    report = analyze(registry([Iterable], [Hashable]))
    assert not report.ambiguous
    finding = report.unknown[0]
    assert "ABC" in finding.reason


def test_provable_abc_relation_is_resolved():
    # Mapping is structurally an Iterable, so no ambiguity is reported.
    assert not analyze(registry([Iterable], [Mapping])).findings


def test_future_multiple_inheritance_is_unknown():
    class Left:
        pass

    class Right:
        pass

    finding = analyze(registry([Left], [Right])).unknown[0]
    assert "future subclass" in finding.reason
    assert finding.witness is None


def test_parametric_predicates_are_unknown_and_not_called():
    calls = []

    def positive(value):
        calls.append(value)
        return value > 0

    def negative(value):
        calls.append(value)
        return value < 0

    method = registry([parametric(int, positive)], [parametric(int, negative)])
    finding = analyze(method).unknown[0]
    assert "parametric" in finding.reason
    assert calls == []


def test_no_user_constructor_or_predicate_invoked():
    constructed = []

    class Forbidden:
        def __new__(cls, *args):
            constructed.append("new")
            raise RuntimeError("analyzer must not construct")

    method = registry([Forbidden], [parametric(int, lambda value: value == 0)])
    analyze(method)
    assert constructed == []


# --------------------------------------------------------------------------- provable disjoint
def test_concrete_container_layouts_are_disjoint():
    assert not analyze(registry([list[int]], [dict[str, int]])).findings
    assert not analyze(registry([int], [str])).findings


def test_fixed_tuple_arity_is_disjoint():
    assert not analyze(registry([tuple[int, str]], [tuple[int, int, int]])).findings


# --------------------------------------------------------------------------- registry mutation
def test_report_tracks_registration_and_deletion():
    method = registry([object, object], [object, int], [int, object])
    assert analyze(method).ambiguous
    method[(int, int)] = lambda *args: None
    assert not analyze(method).ambiguous
    del method[(int, int)]
    assert analyze(method).ambiguous


def test_runtime_remains_authoritative():
    # A covered static diamond resolves at runtime exactly as predicted.
    method = registry(
        [object, object], [object, int], [int, object], [int, int],
    )
    assert method(1, 1) == 3
    assert method(1, "x") == 2
    assert method("x", 1) == 1


def test_report_shapes():
    report = analyze(registry([Iterable], [Hashable]))
    assert isinstance(report, AnalysisReport)
    assert report.ambiguous == []
    assert report.unknown and report.covered == []
    assert "unknown" in str(report)
    empty = analyze(registry([int], [str]))
    assert str(empty) == "no overlapping signature pairs"
