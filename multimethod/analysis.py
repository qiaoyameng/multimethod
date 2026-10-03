"""Static ambiguity analysis for a multimethod registry.

The analyzer inspects registered signatures *only*.  It never dispatches a
request, never calls a registered function, never instantiates user classes and
never evaluates user predicates.  For every pair of signatures that could both
apply to some call, it reports whether a third, more specific signature
resolves the shared region.

Results are deliberately conservative.

*   ``ambiguous`` - an inhabited witness (a safe value, or an existing concrete
    class that is merely described) demonstrates two equally specific
    candidates with no covering overload.
*   ``covered``   - the whole provable intersection is owned by one more
    specific registered signature, including resolved diamonds.
*   ``unknown``   - sound proof is impossible, with explicit reasons: complex
    ABCs, ``parametric`` predicates, or a future multiple-inheritance subclass.

Runtime dispatch stays authoritative: this module only reports.
"""

from __future__ import annotations

import abc
import collections.abc
import dataclasses
import enum
import functools
import types
import typing
from typing import Any

from . import multimethod as _multimethod
from . import parametric as _parametric_cls
from . import signature as _signature
from . import subtype as _subtype_cls

Literal = typing.Literal
UNION_ORIGINS = (typing.Union, types.UnionType)

# Concrete builtin origins for which the analyser reasons structurally.
CONCRETE_ORIGINS = frozenset(
    {
        bool, bytearray, bytes, complex, dict, float, frozenset, int, list,
        memoryview, range, set, str, tuple,
    }
)

# Safe, side-effect-free witness values for concrete origins.  Empty
# collections vacuously satisfy every element parameterization.
SAFE_VALUES: dict[Any, Any] = {
    bool: False,
    bytearray: bytearray(),
    bytes: b"",
    complex: 0j,
    dict: {},
    float: 0.0,
    frozenset: frozenset(),
    int: 0,
    list: [],
    set: set(),
    str: "",
    tuple: (),
    range: range(0),
    type(None): None,
}


def _is_stdlib_abc(cls: Any) -> bool:
    return (
        isinstance(cls, abc.ABCMeta)
        and getattr(cls, "__module__", "") in ("collections.abc", "numbers", "contextlib")
    )


def _kind(tp) -> str:
    origin = getattr(tp, "__origin__", None)
    if origin is Literal:
        return "literal"
    if origin in UNION_ORIGINS:
        return "union"
    if isinstance(tp, _parametric_cls):
        return "parametric"
    return "generic" if origin is not None else "plain"


def _origin(tp):
    return getattr(tp, "__origin__", None)


def _args(tp):
    return getattr(tp, "__args__", ())


# --------------------------------------------------------------------------- structure
def _plain_subset(left: type, right: type) -> bool:
    """Structural ``left <: right`` over the MRO; never runs user hooks."""
    if right in left.__mro__:
        return True
    # Concrete-builtin <-> stdlib-ABC relations are frozen in the interpreter.
    if left in CONCRETE_ORIGINS and _is_stdlib_abc(right):
        return issubclass(left, right)
    return False


def _tuple_subset(la: tuple, ra: tuple) -> bool:
    left_var, right_var = ... in la, ... in ra
    if right_var:
        if left_var:
            return _provable_subset(la[0], ra[0])
        return all(_provable_subset(arg, ra[0]) for arg in la if arg is not ...)
    return (
        not left_var
        and len(la) == len(ra)
        and all(_provable_subset(a, b) for a, b in zip(la, ra))
    )


def _provable_subset(left, right) -> bool:
    """Side-effect-free proof that denotation ``left`` is contained in ``right``."""
    if left is right or left == right:
        return True
    if _kind(right) == "union":
        return any(_provable_subset(left, arg) for arg in _args(right))
    if _kind(left) == "union":
        return all(_provable_subset(arg, right) for arg in _args(left))
    if _kind(left) == "literal":
        if _kind(right) == "literal":
            return {*_args(left)} <= {*_args(right)}
        return all(_literal_matches(value, right) is True for value in _args(left))
    if _kind(right) == "literal" or _kind(left) == "parametric" or _kind(right) == "parametric":
        return False
    lo, ro, la, ra = _origin(left), _origin(right), _args(left), _args(right)
    if lo is tuple or ro is tuple:
        return lo is ro and _tuple_subset(la, ra)
    if lo is not None or ro is not None:
        lo, ro = lo or left, ro or right
        if not (isinstance(lo, type) and isinstance(ro, type)) or not _plain_subset(lo, ro):
            return False
        return not ra or (len(la) == len(ra) and all(map(_provable_subset, la, ra)))
    return isinstance(left, type) and isinstance(right, type) and _plain_subset(left, right)


def _provably_disjoint(left, right) -> bool:
    return _meet(left, right).empty


# --------------------------------------------------------------------------- literals
def _literal_matches(value, tp) -> bool | None:
    """Prove whether a safe literal ``value`` is an instance of structural ``tp``."""
    kind = _kind(tp)
    if kind == "literal":
        return any(type(value) is type(other) and value == other for other in _args(tp))
    if kind == "union":
        verdicts = {_literal_matches(value, arg) for arg in _args(tp)}
        return True if True in verdicts else (False if verdicts == {False} else None)
    if kind == "parametric":
        return None
    origin = _origin(tp)
    if kind == "plain":
        return isinstance(value, tp)
    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        args = _args(tp)
        if ... in args:
            return all(_literal_matches(item, args[0]) is True for item in value)
        if len(value) != len(args):
            return False
        return all(_literal_matches(item, arg) is True for item, arg in zip(value, args))
    if origin in CONCRETE_ORIGINS:
        if not isinstance(value, origin):
            return False
        args = _args(tp)
        if not value:
            return True
        if origin in (list, set, frozenset):
            return all(_literal_matches(item, args[0]) is True for item in value)
        if origin is dict:
            return all(
                _literal_matches(k, args[0]) is True
                and _literal_matches(v, args[1]) is True
                for k, v in value.items()
            )
    return None


# --------------------------------------------------------------------------- meets
@dataclasses.dataclass(frozen=True)
class Node:
    """A provably inhabited subset of an intersection, with a safe witness."""

    types: tuple
    witness: Any  # a safe value, or an existing concrete class (never called)

    def admits_type(self, tp) -> bool | None:
        """Whether this node's witness is provably an instance of ``tp``."""
        witness = self.witness
        if isinstance(witness, type):
            if _provable_subset(witness, tp):
                return True
            if _provably_disjoint(witness, tp):
                return False
            return None
        return _literal_matches(witness, tp)


@dataclasses.dataclass
class TypeMeet:
    node: Node | None = None
    unknown: tuple = ()
    reasons: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        return self.node is None and not self.unknown

    @property
    def exact(self) -> bool:
        return self.node is not None and not self.unknown and self.node.witness is not None


def _safe_value(tp):
    match _kind(tp):
        case "literal":
            return _args(tp)[0]
        case "union":
            return next(
                (value for arg in _args(tp) if (value := _safe_value(arg)) is not None), None
            )
        case "parametric":
            return None
        case "plain":
            value = SAFE_VALUES.get(tp)
            if value is not None:
                return value
            if _is_stdlib_abc(tp):
                return _abc_value(tp)
            return None
    origin, args = _origin(tp), _args(tp)
    if origin is tuple:
        if ... in args:
            return ()
        values = []
        for arg in args:
            value = _safe_value(arg)
            if value is None:
                return None
            values.append(value)
        return tuple(values)
    if origin in CONCRETE_ORIGINS:
        return SAFE_VALUES[origin]
    if _is_stdlib_abc(origin):
        return _abc_value(origin)
    return None


def _abc_value(origin):
    groups = (
        (collections.abc.Callable,),
        (collections.abc.Mapping, collections.abc.MutableMapping,
         collections.abc.ItemsView, collections.abc.MappingView),
        (collections.abc.Set, collections.abc.MutableSet,
         collections.abc.KeysView, collections.abc.ValuesView),
        (collections.abc.Iterable, collections.abc.Collection,
         collections.abc.Sequence, collections.abc.MutableSequence,
         collections.abc.Iterator, collections.abc.Reversible),
    )
    values = (len, {}, frozenset(), [])
    for group, value in zip(groups, values):
        if any(issubclass(origin, base) for base in group):
            return value
    return None


def _inhabited_node(tp) -> Node | None:
    kind = _kind(tp)
    if kind in ("literal", "plain", "generic"):
        value = _safe_value(tp)
        if value is not None:
            return Node((tp,), value)
        if kind == "plain" and isinstance(tp, type) and not isinstance(tp, abc.ABCMeta):
            return Node((tp,), tp)  # describe; never instantiate
    return None


@functools.lru_cache(maxsize=None)
def _meet(left, right) -> TypeMeet:
    left, right = _subtype_cls(left), _subtype_cls(right)
    if left == right:
        node = _inhabited_node(left)
        return (
            TypeMeet(node)
            if node is not None
            else TypeMeet(unknown=(left,), reasons=(f"inhabitation of {left!s} cannot be proved",))
        )
    lkind = _kind(left)
    if lkind == "literal" or _kind(right) == "literal":
        return _literal_meet(left, right) if lkind == "literal" else _literal_meet(right, left)
    if lkind == "union" or _kind(right) == "union":
        return _union_meet(left, right)
    return _structural_meet(left, right)


def _literal_meet(left, right) -> TypeMeet:
    matches, uncertain = [], False
    literal, other = (left, right) if _kind(left) == "literal" else (right, left)
    for value in _args(literal):
        verdict = _literal_matches(value, other)
        if verdict is True:
            matches.append(value)
        elif verdict is None:
            uncertain = True
    if matches:
        args = tuple(matches)
        literal_type = Literal[args] if len(args) > 1 else Literal[args[0]]
        return TypeMeet(Node((_subtype_cls(literal_type),), matches[0]))
    if uncertain:
        return TypeMeet(unknown=(other,), reasons=(f"literal membership in {other!s} cannot be proved",))
    return TypeMeet()


def _union_meet(left, right) -> TypeMeet:
    l_args = _args(left) if _kind(left) == "union" else (left,)
    r_args = _args(right) if _kind(right) == "union" else (right,)
    nodes, unknown, reasons = [], [], []
    for la in l_args:
        for ra in r_args:
            meet = _meet(la, ra)
            if meet.node is not None:
                nodes.append(meet.node)
            unknown.extend(meet.unknown)
            reasons.extend(meet.reasons)
    if nodes:
        return TypeMeet(nodes[0], tuple(dict.fromkeys(unknown)), tuple(dict.fromkeys(reasons)))
    if unknown:
        return TypeMeet(None, tuple(dict.fromkeys(unknown)), tuple(dict.fromkeys(reasons)))
    return TypeMeet()


@functools.lru_cache(maxsize=None)
def _layout_compatible(left: type, right: type) -> bool:
    """Whether two concrete builtins may share instances.

    Probes interpreter layout restrictions; only concrete builtins are probed,
    so no user code runs.
    """
    try:
        types.new_class("_LayoutProbe", (left, right), {})
    except TypeError:
        return False
    return True


def _structural_meet(left, right) -> TypeMeet:
    if _provable_subset(left, right):
        return _inhabited_result(left)
    if _provable_subset(right, left):
        return _inhabited_result(right)
    generic = _generic_meet(left, right)
    if generic is not None:
        return generic
    if _kind(left) == "parametric" or _kind(right) == "parametric":
        return TypeMeet(
            unknown=(left, right),
            reasons=("parametric predicates cannot be decided without calling them",),
        )
    lo, ro = _origin(left) or left, _origin(right) or right
    if lo in CONCRETE_ORIGINS and ro in CONCRETE_ORIGINS:
        existing = _find_concrete(left, right)
        if existing is not None:
            return TypeMeet(Node((left, right), existing))
        if not _layout_compatible(lo, ro):
            return TypeMeet()
        return _common_subclass(left, right, "a concrete multiple-inheritance subclass may exist")
    if isinstance(lo, type) and isinstance(ro, type):
        witness = _generic_abc_meet(left, right, lo, ro)
        if witness is not None:
            return witness
        if _is_stdlib_abc(lo) or _is_stdlib_abc(ro):
            return _common_subclass(
                left, right, f"complex ABC intersection of {left!s} and {right!s} cannot be proved"
            )
        return _common_subclass(
            left, right, f"a future subclass of both {left!s} and {right!s} may exist"
        )
    return TypeMeet(
        unknown=(left, right),
        reasons=(f"intersection of {left!s} and {right!s} cannot be proved",),
    )


def _inhabited_result(tp) -> TypeMeet:
    node = _inhabited_node(tp)
    if node is not None:
        return TypeMeet(node)
    return TypeMeet(unknown=(tp,), reasons=(f"inhabitation of {tp!s} cannot be proved",))


def _generic_abc_meet(left, right, lo, ro) -> TypeMeet | None:
    """Intersection of a concrete generic container with a stdlib ABC generic.

    When the container origin is provably an implementation of the ABC, the
    empty container witnesses the overlap regardless of element parameters
    (it vacuously satisfies them).
    """
    if _origin(left) in CONCRETE_ORIGINS and _is_stdlib_abc(ro):
        concrete, abstract, abstract_args = left, right, _args(right)
    elif _origin(right) in CONCRETE_ORIGINS and _is_stdlib_abc(lo):
        concrete, abstract, abstract_args = right, left, _args(left)
    else:
        return None
    if not _plain_subset(_origin(concrete), _origin(abstract) or abstract):
        return None
    return TypeMeet(Node((left, right), SAFE_VALUES[_origin(concrete)]))


def _common_subclass(left, right, reason: str) -> TypeMeet:
    concrete = _find_concrete(left, right)
    return (
        TypeMeet(Node((left, right), concrete))
        if concrete is not None
        else TypeMeet(unknown=(left, right), reasons=(reason,))
    )


def _find_concrete(left, right) -> type | None:
    """Find an existing concrete class provably below both types.

    A concrete subclass reached from a bare ABC is itself a safe witness class
    (described, never constructed); parameterization is checked structurally.
    """
    origins = [
        origin for tp in (left, right)
        if isinstance((origin := _origin(tp) or tp), type)
    ]
    origins = list(dict.fromkeys(origins))
    seen, queue = set(), list(origins)
    while queue:
        cls = queue.pop()
        if cls in seen or not isinstance(cls, type):
            continue
        seen.add(cls)
        getter = getattr(cls, "__subclasses__", None)
        try:
            subclasses = getter() if callable(getter) else ()
        except TypeError:
            subclasses = ()
        if isinstance(cls, abc.ABCMeta) and getattr(cls, "__abstractmethods__", frozenset()):
            queue.extend(subclasses)
            continue
        if cls in CONCRETE_ORIGINS or any(origin in cls.__mro__ for origin in origins):
            if _provable_subset(cls, left) and _provable_subset(cls, right):
                return cls
        queue.extend(subclasses)
    return None


def _generic_meet(left, right) -> TypeMeet | None:
    """Intersection of same-origin concrete containers.

    Empty containers vacuously satisfy every element parameterization, so
    parameterizations of the same mutable container overlap at the empty value.
    Fixed-length tuples intersect element-wise (and only with matching arity);
    variable-length tuples always share the empty tuple.
    """
    lo, ro = _origin(left), _origin(right)
    if lo is not ro or lo not in (tuple, list, set, frozenset, dict):
        return None
    la, ra = _args(left), _args(right)
    if lo is not tuple:
        # Empty mutable/set containers vacuously satisfy any parameterization,
        # so the intersection is always inhabited by the empty value.
        return TypeMeet(Node((left, right), SAFE_VALUES[lo]))
    if ... in la or ... in ra:
        return TypeMeet(Node((_subtype_cls(tuple),), ()))
    if len(la) != len(ra):
        return TypeMeet()
    element_meets = tuple(_meet(a, b) for a, b in zip(la, ra))
    if any(meet.empty for meet in element_meets):
        return TypeMeet()
    values = []
    for meet in element_meets:
        if meet.node is None or meet.node.witness is None:
            unknown = tuple(dict.fromkeys(u for meet in element_meets for u in meet.unknown))
            reasons = tuple(dict.fromkeys(r for meet in element_meets for r in meet.reasons))
            return TypeMeet(None, unknown, reasons)
        values.append(meet.node.witness)
    return TypeMeet(Node((left, right), tuple(values)))


# --------------------------------------------------------------------------- signatures
@dataclasses.dataclass
class Region:
    meets: tuple[TypeMeet, ...]

    @property
    def empty(self) -> bool:
        return any(meet.empty for meet in self.meets)

    @property
    def exact(self) -> bool:
        return all(meet.exact for meet in self.meets)

    @property
    def witness(self) -> tuple | None:
        values = []
        for meet in self.meets:
            if meet.node is None or meet.node.witness is None:
                return None
            values.append(meet.node.witness)
        return tuple(values)

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(
            f"arg {index}: {reason}"
            for index, meet in enumerate(self.meets)
            for reason in meet.reasons
        )


def _pair_region(a: _signature, b: _signature) -> Region:
    length = max(len(a), len(b))
    meets = tuple(
        _meet(a[index] if index < len(a) else object, b[index] if index < len(b) else object)
        for index in range(length)
    )
    return Region(meets)


def _more_specific(more: _signature, less: _signature) -> bool:
    """Global structural precedence, mirroring the runtime signature DAG.

    A key refines only keys it is at least as long as; absent positions in
    ``less`` are implicitly ``object``.
    """
    if len(more) < len(less):
        return False
    for index in range(len(more)):
        candidate = less[index] if index < len(less) else object
        if not _provable_subset(more[index], candidate):
            return False
    return True


def _position_overlap(key_tp, meet: TypeMeet) -> str:
    """Overlap of a registered type with one region position.

    * ``full``  - provably covers every provable part of the region;
    * ``part``  - overlaps an inhabited strict sub-region only;
    * ``none``  - provably disjoint;
    * ``maybe`` - overlap depends on an unprovable (unknown) branch.
    """
    if meet.node is None and not meet.unknown:
        return "none"
    known_verdict = meet.node.admits_type(key_tp) if meet.node is not None else None
    if not meet.unknown:
        return {True: "full", False: "none"}.get(known_verdict, "part")
    if known_verdict is True and all(
        _provable_subset(candidate, key_tp) for candidate in meet.unknown
    ):
        return "full"
    if any(
        not _meet(candidate, key_tp).empty and not _provable_subset(candidate, key_tp)
        for candidate in meet.unknown
    ):
        return "maybe"
    return "part" if known_verdict is True else "none"


def _overlap(key: _signature, region: Region) -> list[str]:
    return [
        _position_overlap(key[index] if index < len(key) else object, meet)
        for index, meet in enumerate(region.meets)
    ]


def _maximal(keys: list[_signature]) -> list[_signature]:
    return [
        key
        for key in keys
        if not any(other is not key and _more_specific(other, key) for other in keys)
    ]


# --------------------------------------------------------------------------- report
class Confidence(enum.Enum):
    AMBIGUOUS = "ambiguous"
    COVERED = "covered"
    UNKNOWN = "unknown"


@dataclasses.dataclass
class Finding:
    signatures: tuple[_signature, _signature]
    confidence: Confidence
    witness: tuple | None
    reason: str

    def __str__(self) -> str:
        pair = ", ".join(map(_describe_signature, self.signatures))
        lines = [f"{self.confidence.value}: ({pair})"]
        if self.witness is not None:
            lines.append(f"  witness: {_describe_witness(self.witness)}")
        if self.reason:
            lines.append(f"  reason: {self.reason}")
        return "\n".join(lines)


def _describe_signature(sig: _signature) -> str:
    return "(" + ", ".join(getattr(tp, "__name__", str(tp)) for tp in sig) + ")"


def _describe_witness(values: tuple) -> str:
    parts = []
    for value in values:
        if isinstance(value, type):
            parts.append(f"value of type {value.__name__} (described, not constructed)")
        else:
            parts.append(repr(value))
    return "(" + ", ".join(parts) + ")"


@dataclasses.dataclass
class AnalysisReport:
    findings: list[Finding] = dataclasses.field(default_factory=list)

    @property
    def ambiguous(self) -> list[Finding]:
        return [f for f in self.findings if f.confidence is Confidence.AMBIGUOUS]

    @property
    def covered(self) -> list[Finding]:
        return [f for f in self.findings if f.confidence is Confidence.COVERED]

    @property
    def unknown(self) -> list[Finding]:
        return [f for f in self.findings if f.confidence is Confidence.UNKNOWN]

    def __bool__(self) -> bool:
        return bool(self.ambiguous)

    def __str__(self) -> str:
        if not self.findings:
            return "no overlapping signature pairs"
        order = (Confidence.AMBIGUOUS, Confidence.UNKNOWN, Confidence.COVERED)
        lines = []
        for confidence in order:
            group = [f for f in self.findings if f.confidence is confidence]
            if not group:
                continue
            lines.append(f"[{confidence.value}] {len(group)} pair(s)")
            for finding in group:
                lines.extend("  " + line for line in str(finding).splitlines())
        return "\n".join(lines)


def analyze(method: _multimethod.multimethod) -> AnalysisReport:
    """Analyze a multimethod registry without dispatching.

    The registry is read on every call, so registering or deleting overloads
    updates the result.  Runtime dispatch remains the final authority.
    """
    method.evaluate()
    keys = [key for key in list(method) if isinstance(key, _signature)]
    report = AnalysisReport()
    for index, a in enumerate(keys):
        for b in keys[index + 1 :]:
            finding = _analyze_pair(method, a, b)
            if finding is not None:
                report.findings.append(finding)
    return report


def _analyze_pair(method, a: _signature, b: _signature) -> Finding | None:
    region = _pair_region(a, b)
    if region.empty:
        return None
    keys = [key for key in method if isinstance(key, _signature)]
    overlap = {key: _overlap(key, region) for key in keys}

    def covers(key) -> bool:
        return all(value == "full" for value in overlap[key])

    def reaches(key) -> bool:
        return all(value in ("full", "part") for value in overlap[key])

    full_keys = [key for key in keys if covers(key)]
    covering = [
        key for key in full_keys
        if not any(other is not key and _more_specific(other, key) for other in full_keys)
    ]
    funcs = {method[key] for key in covering}
    resolver = covering[0] if covering else None
    if covering:
        resolver = next(
            (key for key in covering if all(
                other is key or not _more_specific(other, key) for other in covering
            )),
            covering[0],
        )

    # A third-party key touching only a strict inhabited sub-region cannot win
    # values elsewhere; it is absorbed when the rest of the region resolves
    # uniquely.  Unprovable overlap (``maybe``) always makes a key a spoiler.
    def absorbed(key) -> bool:
        values = overlap[key]
        if "maybe" in values:
            return False
        if covers(key) or not reaches(key):
            return True
        return len(funcs) == 1

    spoilers = [key for key in keys if key not in (a, b) and not absorbed(key)]

    if not spoilers and len(funcs) == 1:
        # If a covering pair arm strictly dominates the other, the pair is
        # simply nested (no diamond) and produces no finding.
        if resolver in (a, b):
            other = b if resolver is a else a
            if _more_specific(resolver, other):
                return None
        return Finding(
            (a, b), Confidence.COVERED, None,
            f"intersection resolved by {_describe_signature(resolver)}",
        )
    if not spoilers and len(funcs) > 1 and region.exact:
        return Finding(
            (a, b), Confidence.AMBIGUOUS, region.witness,
            "no unique more specific candidate",
        )
    reasons = list(region.reasons)
    if spoilers:
        reasons.append(
            "overload may intersect the region without provably covering it: "
            + ", ".join(_describe_signature(key) for key in spoilers)
        )
    if len(funcs) > 1 and not region.exact:
        reasons.append("multiple candidates cover a region whose full extent is unknown")
    return Finding(
        (a, b), Confidence.UNKNOWN, region.witness if region.exact else None,
        "; ".join(reasons) or "not provable",
    )
