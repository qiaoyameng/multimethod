import abc
import builtins
import collections
import contextlib
import dataclasses
import functools
import inspect
import itertools
import operator
import types
import typing
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, Self, TypeVar, Union, get_type_hints

TypeAliasType = getattr(typing, "TypeAliasType", types.new_class(""))  # python <3.12


class DispatchError(TypeError):
    """A `TypeError` raised when no single applicable method is found."""


def get_origin(tp):
    return tp.__origin__ if isinstance(tp, subtype) else typing.get_origin(tp)


def get_args(tp) -> tuple:
    if isinstance(tp, subtype) or typing.get_origin(tp) is Callable:
        return getattr(tp, "__args__", ())
    return typing.get_args(tp)


def common_bases(*bases):
    counts = collections.Counter(cls for base in bases for cls in base.__mro__)
    return tuple(cls for cls in counts if counts[cls] == len(bases))


class subtype(abc.ABCMeta):
    """A normalized generic type which checks subscripts.

    Transforms a generic alias into a concrete type which supports `issubclass` and `isinstance`.
    If the type ends up being equivalent to a builtin, the builtin is returned.
    """

    __origin__: type
    __args__: tuple

    def __new__(cls, tp):
        match tp:
            case typing.Any:
                return object
            case typing.NewType():
                return cls(tp.__supertype__)
            case TypeVar():
                return cls(Union[tp.__constraints__]) if tp.__constraints__ else object
            case typing._AnnotatedAlias():
                return cls(tp.__origin__)
            case TypeAliasType():
                return cls(tp.__value__)
            case type():
                return tp
        origin = get_origin(tp) or tp
        args = tuple(map(cls, get_args(tp)))
        if not args and origin is not tuple:
            return origin
        bases = (origin,)
        match origin:
            case typing.Literal:
                args = typing.get_args(tp)
                bases = (cls(Union[tuple(map(type, args))]),)
            case typing.Union | types.UnionType:
                origin = types.UnionType
                bases = common_bases(*args)[:1]
                if bases[0] in args:
                    return bases[0]
            case builtins.type:  # special case type of `NewType`
                (arg,) = typing.get_args(tp)
                if isinstance(arg, typing.NewType):
                    origin, args = typing.NewType, (arg,)
        namespace = {"__origin__": origin, "__args__": args}
        return type.__new__(cls, str(tp), bases, namespace)

    def key(self) -> tuple:
        return self.__origin__, *self.__args__

    def __eq__(self, other) -> bool:
        return isinstance(other, subtype) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    def __subclasscheck__(self, subclass):
        args = get_args(subclass)
        match origin := get_origin(subclass):
            case None:
                origin = subclass
            case typing.Literal:
                return all(isinstance(arg, self) for arg in args)
            case typing.Union | types.UnionType:
                return all(issubclass(cls, self) for cls in args)
        match self.__origin__:
            case typing.Literal:
                return False
            case types.UnionType:
                return issubclass(subclass, self.__args__)
            case builtins.tuple:
                if issubclass(origin, tuple) and ... in self.__args__:
                    param = self.__args__[0]
                    return all(arg is ... or issubclass(arg, param) for arg in args)
            case collections.abc.Callable:
                params = self.__args__[:-1]
                return (
                    origin is Callable
                    and issubclass(args[-1], self.__args__[-1])  # covariant return
                    and (
                        ... in params
                        or (... not in args and signature(args[:-1]).subtypes(*params))
                    )  # contravariant args
                )
        return (  # check args first to avoid recursion error: python/cpython#73407
            len(args) == len(self.__args__)
            and issubclass(origin, self.__origin__)
            and all(pair[0] is pair[1] or issubclass(*pair) for pair in zip(args, self.__args__))
        )

    def __instancecheck__(self, instance):
        match self.__origin__:
            case typing.Literal:
                return any(type(arg) is type(instance) and arg == instance for arg in self.__args__)
            case types.UnionType:
                return isinstance(instance, self.__args__)
            case builtins.type:
                try:
                    return issubclass(subtype(instance), self.__args__)
                except TypeError:
                    return False
            case typing.NewType:
                return isinstance(instance, typing.NewType) and (
                    instance in self.__args__ or isinstance(instance.__supertype__, self)
                )
        if isinstance(instance, typing.Generic):  # user-defined generic type
            return issubclass(getattr(instance, "__orig_class__", type(instance)), self)
        if not isinstance(instance, self.__origin__) or isinstance(instance, Iterator):
            return False
        if self.__origin__ is Callable:
            hints = get_type_hints(instance)
            args = [hints.get(name, object) for name in inspect.signature(instance).parameters]
            return issubclass(Callable[args, hints.get("return", object)], self)
        if self.__origin__ is tuple and ... not in self.__args__:
            if len(instance) != len(self.__args__):
                return False
        elif issubclass(self, Mapping):
            instance = next(iter(instance.items()), ())
        else:
            instance = itertools.islice(instance, 1)
        return all(map(isinstance, instance, self.__args__))

    @functools.singledispatch
    def origins(self) -> Iterable[type]:
        """Return origin types which would require instance checks.

        Provisional custom usage: `subtype.origins.register(<metaclass>, lambda cls: ...)
        """
        match origin := get_origin(self):
            case typing.Literal:
                yield from set(map(type, self.__args__))
            case types.UnionType:
                for arg in self.__args__:
                    yield from subtype.origins(arg)
            case builtins.type:
                yield from (type, types.GenericAlias, typing._GenericAlias, types.UnionType)
            case _ if origin is not None:
                yield origin


class parametric(abc.ABCMeta):
    """A type which further customizes `issubclass` and `isinstance` beyond the base type.

    Args:
        base: base type
        funcs: all predicate functions are checked against the instance
        attrs: all attributes are checked for equality
    """

    def __new__(cls, base: type, *funcs: Callable, **attrs):
        return super().__new__(cls, base.__name__, (base,), {"funcs": funcs, "attrs": attrs})

    def __init__(self, *_, **__): ...

    def __subclasscheck__(self, subclass):
        missing = object()
        attrs = getattr(subclass, "attrs", {})
        return (
            set(subclass.__bases__).issuperset(self.__bases__)  # python/cpython#73407
            and set(getattr(subclass, "funcs", ())).issuperset(self.funcs)
            and all(attrs.get(name, missing) == self.attrs[name] for name in self.attrs)
        )

    def __instancecheck__(self, instance):
        missing = object()
        return (
            isinstance(instance, self.__bases__)
            and all(func(instance) for func in self.funcs)
            and all(getattr(instance, name, missing) == self.attrs[name] for name in self.attrs)
        )

    def __and__(self, other):
        (base,) = set(self.__bases__ + other.__bases__)
        return type(self)(base, *set(self.funcs + other.funcs), **(self.attrs | other.attrs))


subtype.origins.register(parametric, lambda cls: cls.__bases__)


class signature(tuple):
    """A tuple of types that supports partial ordering."""

    required: int
    parents: set
    sig: inspect.Signature

    def __new__(cls, types: Iterable, required: int | None = None):
        return tuple.__new__(cls, map(subtype, types))

    def __init__(self, types: Iterable, required: int | None = None):
        self.required = len(self) if required is None else required

    @classmethod
    def from_hints(cls, func: Callable) -> Self:
        """Return evaluated type hints for positional parameters in order."""
        if not hasattr(func, "__annotations__"):
            return cls(())
        type_hints = get_type_hints(func)
        positionals = {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        params: Iterable = inspect.signature(func).parameters.values()
        params = [param for param in params if param.kind in positionals]
        # missing annotations are padded with `object`, but trailing objects are unnecessary
        indices = [index for index, param in enumerate(params) if param.name in type_hints]
        params = params[: max(indices, default=-1) + 1]
        hints = [type_hints.get(param.name, object) for param in params]
        required = sum(param.default is param.empty for param in params)
        return cls(hints, required)

    def subtypes(self, *other) -> bool:
        """Return whether all types are subclasses."""
        return self.required <= len(other) and all(map(issubclass, other, self))

    def callable(self, *types) -> bool:
        """Check positional arity of associated function signature."""
        try:
            return not hasattr(self, "sig") or bool(self.sig.bind_partial(*types))
        except TypeError:
            return False

    def instances(self, *args) -> bool:
        """Return whether all arguments are instances."""
        return self.required <= len(args) and all(map(isinstance, args, self))


REGISTERED = TypeVar("REGISTERED", bound=Callable[..., Any])


@dataclasses.dataclass(frozen=True)
class Conflict:
    """A pair of signatures which may dispatch ambiguously.

    Attributes:
        signatures: the two overlapping registered signatures
        witness: a type tuple applicable to both signatures, or ``None``
        reason: why the analysis could or could not produce a definitive witness
        covered: a signature proved more specific than both over the witness, if any
    """

    signatures: tuple[signature, signature]
    witness: tuple | None
    reason: str
    covered: signature | None = None

    def __str__(self) -> str:
        first, second = self.signatures
        witness = tuple(str(item) for item in self.witness) if self.witness is not None else None
        text = f"{first} vs {second}: {self.reason}"
        if witness is not None:
            text += f"; witness {witness}"
        if self.covered is not None:
            text += f"; covered by {self.covered}"
        return text


@dataclasses.dataclass
class Report:
    """Result of a registry-only ambiguity analysis.

    ``conflicts`` holds provable ambiguities with a witness; ``unknown`` holds pairs
    whose applicability overlap could not be proven empty or covered.
    Runtime dispatch remains the final authority.
    """

    name: str
    conflicts: list[Conflict] = dataclasses.field(default_factory=list)
    unknown: list[Conflict] = dataclasses.field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"{self.name}: {len(self.conflicts)} ambiguous, {len(self.unknown)} unknown"]
        lines += [f"  ambiguous: {conflict}" for conflict in self.conflicts]
        lines += [f"  unknown: {conflict}" for conflict in self.unknown]
        return "\n".join(lines)


@dataclasses.dataclass(frozen=True)
class ValueWitness:
    """A safe, constructor-free value description (e.g. an empty container)."""

    origin: type

    def __str__(self) -> str:
        literals = {
            builtins.list: "[]",
            builtins.dict: "{}",
            builtins.set: "set()",
            builtins.frozenset: "frozenset()",
            builtins.tuple: "()",
        }
        return literals.get(self.origin, f"{self.origin.__name__}()")

    @property
    def value(self):
        return self.origin()


@dataclasses.dataclass(frozen=True)
class LiteralValue:
    """A safe literal value shared by two intersecting literal signatures."""

    value: object

    def __str__(self) -> str:
        return repr(self.value)


_EMPTY = "empty"
_NONEMPTY = "nonempty"
_UNKNOWN = "unknown"

_CONTAINER_ORIGINS = {
    builtins.list,
    builtins.dict,
    builtins.set,
    builtins.frozenset,
    builtins.tuple,
    collections.deque,
    collections.defaultdict,
    collections.OrderedDict,
}

_LEAF_TYPES = frozenset(
    {
        builtins.bool,
        builtins.int,
        builtins.float,
        builtins.complex,
        builtins.str,
        builtins.bytes,
        builtins.bytearray,
        builtins.range,
        builtins.frozenset,
        builtins.tuple,
    }
)


def _is_parametric(tp) -> bool:
    return isinstance(tp, type) and type(tp) is parametric


def _is_plain(tp) -> bool:
    """Whether ``isinstance``/``issubclass`` of ``tp`` runs no user predicates."""
    metacls = type(tp)
    if _is_parametric(tp) or not isinstance(metacls, type):
        return False
    return metacls in (type, abc.ABCMeta, subtype)


def _container_witness(origin, args) -> type:
    """Construct a concrete parameterized subtype, or ``None`` if impossible."""
    if not args:
        return None
    params = [arg for arg in args if arg is not Ellipsis]
    try:
        subscript = tuple(params) if Ellipsis not in args else (params[0], Ellipsis)
        return subtype(origin[subscript])
    except (TypeError, ValueError):
        return None


def _literal(values) -> type:
    return subtype(typing.Literal[tuple(values)] if len(values) > 1 else values[0])


def _sealed(value_type: type) -> bool:
    """Whether a literal value's type cannot acquire virtual subtypes."""
    return value_type.__module__ == "builtins" and not isinstance(value_type, abc.ABCMeta)


def _meet_literal_value(value, other) -> tuple[str, type | None, str]:
    if isinstance(value, other):
        return _NONEMPTY, LiteralValue(value), "literal"
    value_type = type(value)
    other_origin = get_origin(other)
    if other_origin in _CONTAINER_ORIGINS:
        return _EMPTY, None, "literal value is not a container"
    if (
        other_origin is not None
        and isinstance(other_origin, type)
        and issubclass(value_type, other_origin)
    ):
        args = get_args(other)
        if not args:
            reason = f"literal is an instance of {other_origin.__name__}"
            return _NONEMPTY, _literal([value]), reason
        reason = f"literal {value_type.__name__} cannot match generic {other_origin.__name__}"
        return _EMPTY, None, reason
    if _sealed(value_type) and (
        other_origin is None
        and not issubclass(value_type, other)
    ):
        return _EMPTY, None, f"sealed {value_type.__name__} cannot be a {other.__name__}"
    return _UNKNOWN, None, "literal type may satisfy a virtual or future subclass"


def _union(members) -> type:
    members = tuple(dict.fromkeys(members))
    return members[0] if len(members) == 1 else subtype(functools.reduce(operator.or_, members))


def _meet(left, right) -> tuple[str, type | None, str]:
    """Intersection of two normalized types.

    Returns a triple of status, a witness type and a reason.
    Status is one of ``'nonempty'`` (a common subtype is proven),
    ``'empty'`` (the intersection is proven empty) or ``'unknown'``.
    Only ordinary inheritance, unions, literals and concrete containers are decided;
    complex ABCs, parametric predicates and possible future multiple inheritance
    return ``'unknown'``.
    """
    left_literal = get_origin(left) is typing.Literal
    right_literal = get_origin(right) is typing.Literal
    if left_literal or right_literal:
        literal_tp, other = (left, right) if left_literal else (right, left)
        if _is_parametric(other):
            return _UNKNOWN, None, "parametric predicates cannot be evaluated statically"
        if get_origin(other) is types.UnionType:
            results = [_meet(literal_tp, arg) for arg in get_args(other)]
        elif get_origin(other) is typing.Literal:
            values = [
                value
                for value in get_args(literal_tp)
                if any(
                    type(value) is type(other_value) and value == other_value
                    for other_value in get_args(other)
                )
            ]
            results = (
                [(_NONEMPTY, LiteralValue(values[0]), "literal")]
                if values
                else [(_EMPTY, None, "literals share no values")]
            )
        else:
            values = get_args(literal_tp)
            if not _is_plain(other):
                return _UNKNOWN, None, "literal checked against a non-plain type"
            results = [_meet_literal_value(value, other) for value in values]
        return _fold(results, "literal")
    if get_origin(left) is types.UnionType or get_origin(right) is types.UnionType:
        left_args = get_args(left) if get_origin(left) is types.UnionType else (left,)
        right_args = get_args(right) if get_origin(right) is types.UnionType else (right,)
        results = [_meet(a, b) for a in left_args for b in right_args]
        return _fold(results, "union")
    if _is_parametric(left) or _is_parametric(right):
        return _UNKNOWN, None, "parametric predicates cannot be evaluated statically"
    try:
        if issubclass(left, right):
            return _NONEMPTY, left, "inheritance"
        if issubclass(right, left):
            return _NONEMPTY, right, "inheritance"
    except TypeError as error:
        return _UNKNOWN, None, f"uncomparable types: {error}"
    left_origin = get_origin(left)
    right_origin = get_origin(right)
    if left_origin is None or right_origin is None:
        if (
            left_origin is None
            and right_origin is None
            and left in _LEAF_TYPES
            and right in _LEAF_TYPES
            and not issubclass(left, right)
            and not issubclass(right, left)
        ):
            reason = f"sealed builtins {left.__name__} and {right.__name__} cannot share a subclass"
            return _EMPTY, None, reason
        names = f"{left.__name__}, {right.__name__}"
        return _UNKNOWN, None, f"unrelated classes {names}; a future common subclass is possible"
    if left_origin not in _CONTAINER_ORIGINS or right_origin not in _CONTAINER_ORIGINS:
        return _UNKNOWN, None, f"{left_origin.__name__} has structural or virtual subtypes"
    if left_origin is not right_origin:
        reason = "different concrete containers; a multiple-inheritance subclass is possible"
        return _UNKNOWN, None, reason
    left_args = get_args(left)
    right_args = get_args(right)
    if not left_args or not right_args:
        if left_origin is builtins.tuple:
            return _NONEMPTY, left if not left_args else right, "unparameterized tuple"
        reason = "empty instance matches any parameterization"
        return _NONEMPTY, ValueWitness(left_origin), reason
    if left_origin is builtins.tuple:
        if Ellipsis not in left_args and Ellipsis not in right_args:
            if len(left_args) != len(right_args):
                return _EMPTY, None, "fixed-length tuples have different lengths"
            meets = [_meet(a, b) for a, b in zip(left_args, right_args)]
        elif Ellipsis in left_args and Ellipsis in right_args:
            meets = [_meet(left_args[0], right_args[0])]
        else:
            fixed = left_args if Ellipsis not in left_args else right_args
            variadic = left_args[0] if Ellipsis in left_args else right_args[0]
            meets = [_meet(arg, variadic) for arg in fixed]
        if any(status == _EMPTY for status, _, _ in meets):
            return _EMPTY, None, "tuple element types are disjoint"
        if all(status == _NONEMPTY for status, _, _ in meets):
            args = tuple(witness for _, witness, _ in meets)
            variadic_tuple = Ellipsis in left_args or Ellipsis in right_args
            subscript = (args[0], Ellipsis) if variadic_tuple else args
            return _NONEMPTY, left_origin[subscript], "tuple elements intersect"
        return _UNKNOWN, None, "tuple element overlap not provable"
    meets = [_meet(a, b) for a, b in zip(left_args, right_args)]
    if all(status == _NONEMPTY for status, _, _ in meets):
        args = tuple(witness for _, witness, _ in meets)
        if all(isinstance(arg, type) for arg in args) and (
            witness := _container_witness(left_origin, args)
        ) is not None:
            return _NONEMPTY, witness, "container parameters intersect"
        return _UNKNOWN, ValueWitness(left_origin), "empty instance matches both parameterizations"
    if not any(status == _EMPTY for status, _, _ in meets) and any(
        status == _UNKNOWN for status, _, _ in meets
    ):
        return _UNKNOWN, ValueWitness(left_origin), "empty instance matches both parameterizations"
    return _NONEMPTY, ValueWitness(left_origin), "empty instance matches both parameterizations"


def _fold(results, kind) -> tuple[str, type | None, str]:
    """Combine branch intersections of a literal or union type."""
    statuses = {status for status, _, _ in results}
    witnesses = [witness for status, witness, _ in results if status == _NONEMPTY]
    if _UNKNOWN in statuses:
        reasons = "; ".join(
            dict.fromkeys(reason for status, _, reason in results if status == _UNKNOWN)
        )
        detail = f"{kind} overlap not provable ({reasons})"
        if witnesses:
            return _UNKNOWN, _combine_witnesses(witnesses), f"proven {kind} overlap, but {reasons}"
        return _UNKNOWN, None, detail
    if witnesses:
        return _NONEMPTY, _combine_witnesses(witnesses), f"{kind} branches intersect"
    return _EMPTY, None, f"all {kind} branches are disjoint"


def _combine_witnesses(witnesses):
    if len(witnesses) == 1:
        return witnesses[0]
    if all(isinstance(item, type) for item in witnesses):
        return _union(witnesses)
    return tuple(witnesses)


def _more_specific(sig: signature, other: signature) -> bool:
    """Whether ``sig`` is strictly more specific than ``other`` at every position."""
    return bool(sig) and other.subtypes(*sig) and not sig.subtypes(*other)


def _witness_applies(item, tp) -> bool:
    if isinstance(item, ValueWitness):
        return isinstance(item.value, tp)
    if isinstance(item, LiteralValue):
        return isinstance(item.value, tp)
    return issubclass(item, tp)


def _covered(first: signature, second: signature, witness: tuple, signatures) -> signature | None:
    """Find a unique registered signature dominating the ambiguity over a witness."""
    winners = []
    for candidate in signatures:
        if candidate is first or candidate is second or candidate.required > len(witness):
            continue
        checked = witness[: len(candidate)]
        if len(checked) != len(candidate) or not all(
            _witness_applies(item, tp) for item, tp in zip(checked, candidate)
        ):
            continue
        if not _more_specific(candidate, first) or not _more_specific(candidate, second):
            continue
        checked_types = tuple(
            type(item.value)
            if isinstance(item, (ValueWitness, LiteralValue))
            else item
            for item in checked
        )
        if not candidate.callable(*checked_types):
            continue
        winners.append(candidate)
    if len({tuple(winner) for winner in winners}) != 1:
        return None
    return winners[0]


def _analyze(multimethod) -> Report:
    signatures = [key for key in list(multimethod) if isinstance(key, signature)]
    report = Report(multimethod.__name__)
    for index, first in enumerate(signatures):
        for second in signatures[index + 1 :]:
            if first.subtypes(*second) or second.subtypes(*first):
                continue
            arity = max(first.required, second.required)
            if arity > len(first) or arity > len(second):
                continue
            meets = [
                _meet(first[position], second[position]) for position in range(arity)
            ]
            pair = (first, second)
            if any(status == _EMPTY for status, _, _ in meets):
                continue
            if multimethod[first] is multimethod[second]:
                continue  # duplicate registrations of one function resolve to it
            witness = tuple(tp for _, tp, _ in meets)
            proven = all(status == _NONEMPTY for status, _, _ in meets)
            if None not in witness:
                winner = _covered(first, second, witness, signatures)
                if winner is not None:
                    continue
                reason = "overlapping signatures with no unique more specific overload"
                if not proven:
                    unknown = "; ".join(
                        dict.fromkeys(
                            reason
                            for status, _, reason in meets
                            if status == _UNKNOWN
                        )
                    )
                    reason += f"; partial witness, {unknown}"
                report.conflicts.append(Conflict(pair, witness, reason, winner))
            else:
                reasons = "; ".join(
                    dict.fromkeys(reason for status, _, reason in meets if status == _UNKNOWN)
                )
                reason = f"applicability overlap undecidable ({reasons})"
                report.unknown.append(Conflict(pair, None, reason))
    return report


class multimethod(dict):
    """A callable directed acyclic graph of methods."""

    __name__: str
    pending: set
    generics: list[tuple]  # positional bases which require instance checks

    def __new__(cls, func):
        homonym = inspect.currentframe().f_back.f_locals.get(func.__name__)
        if isinstance(homonym, multimethod):
            return homonym

        self = functools.update_wrapper(dict.__new__(cls), func)
        self.pending = set()
        self.generics = []
        return self

    def __init__(self, func: Callable):
        try:
            self[signature.from_hints(func)] = func
        except (NameError, AttributeError):
            self.pending.add(func)

    @typing.overload
    def register(self, __func: REGISTERED) -> REGISTERED: ...

    @typing.overload
    def register(self, *args: type) -> Callable[[REGISTERED], REGISTERED]: ...

    def register(self, *args) -> Callable:
        """Decorator for registering a function.

        Optionally call with types to return a decorator for unannotated functions.
        """
        if len(args) == 1 and hasattr(args[0], "__annotations__"):
            multimethod.__init__(self, *args)
            return self if self.__name__ == args[0].__name__ else args[0]
        return lambda func: self.__setitem__(args, func) or func

    def __get__(self, instance, owner):
        return self if instance is None else types.MethodType(self, instance)

    def parents(self, types: tuple) -> set:
        """Find immediate parents of potential key."""
        parents = {key for key in list(self) if isinstance(key, signature) and key.subtypes(*types)}
        parents.discard(types)
        return parents - {ancestor for parent in parents for ancestor in parent.parents}

    def clean(self):
        """Empty the cache."""
        for key in list(self):
            if not isinstance(key, signature):
                super().__delitem__(key)

    def copy(self):
        """Return a new multimethod with the same methods."""
        return dict.__new__(type(self)).__ior__(self)

    def __setitem__(self, types: tuple, func: Callable):
        self.clean()
        if not isinstance(types, signature):
            types = signature(types)
        parents = types.parents = self.parents(types)
        with contextlib.suppress(ValueError):
            types.sig = inspect.signature(func)
        self.pop(types, None)  # ensure key is overwritten
        for key in self:
            if types.subtypes(*key) and (not parents or parents & key.parents):
                key.parents -= parents
                key.parents.add(types)
        for index, cls in enumerate(types):
            if origins := set(subtype.origins(cls)):
                self.generics += [()] * (index + 1 - len(self.generics))
                self.generics[index] = tuple(origins.union(self.generics[index]))
        super().__setitem__(types, func)
        self.__doc__ = self.docstring

    def __delitem__(self, types: tuple):
        self.clean()
        super().__delitem__(types)
        for key in self:
            if types in key.parents:
                key.parents = self.parents(key)
        self.__doc__ = self.docstring

    def select(self, types: tuple, keys: set[signature]) -> Callable:
        """Return the next applicable method of given types, or raise `DispatchError`.

        Provisional: override to customize ambiguity resolution.

        Args:
            types: the types to dispatch on
            keys: the most specific matching signatures, with no strict precedence
        """
        funcs = {key: self[key] for key in keys if key.callable(*types)}
        if len(set(funcs.values())) == 1:
            return funcs.popitem()[1]
        raise DispatchError(f"{self.__name__}: {len(funcs)} methods found", types, set(funcs))

    def __missing__(self, types: tuple) -> Callable:
        """Find and cache the next applicable method of given types."""
        self.evaluate()
        types = tuple(map(subtype, types))
        if types in self:
            return self[types]
        return self.setdefault(types, self.select(types, self.parents(types)))

    def dispatch(self, *args) -> Callable:
        self.evaluate()
        types = tuple(map(type, args))
        if not any(map(issubclass, types, self.generics)):
            return self[types]
        matches = {key for key in list(self) if isinstance(key, signature) and key.instances(*args)}
        matches -= {ancestor for match in matches for ancestor in match.parents}
        return self.select(types, matches)

    def __call__(self, *args, **kwargs):
        """Resolve and dispatch to best method."""
        func = self.dispatch(*args)
        try:
            return func(*args, **kwargs)
        except TypeError as ex:
            raise DispatchError(f"Function {func.__code__}") from ex

    def evaluate(self):
        """Evaluate any pending forward references."""
        while self.pending:
            func = self.pending.pop()
            self[signature.from_hints(func)] = func

    def check(self) -> Report:
        """Analyze the registry only, without dispatching or constructing values.

        Return a `Report` of signature pairs whose applicability overlap cannot be
        resolved by a unique more specific overload. Provable ambiguities include a
        type or safe-value witness; undecidable pairs (complex ABCs, parametric
        predicates, possible future subclasses) are reported as unknown with a reason.
        No user constructors or predicates are invoked.
        """
        self.evaluate()
        return _analyze(self)

    @property
    def docstring(self):
        """a descriptive docstring of all registered functions"""
        docs = []
        for key, func in self.items():
            sig = getattr(key, "sig", "")
            if func.__doc__:
                docs.append(f"{func.__name__}{sig}\n    {func.__doc__}")
        return "\n\n".join(docs)


RETURN = TypeVar("RETURN")


class multidispatch(multimethod, dict[tuple[type, ...], Callable[..., RETURN]]):
    """Wrapper for compatibility with `functools.singledispatch`.

    Only uses the `register` method instead of namespace lookup.
    Allows dispatching on keyword arguments based on the first function signature.
    """

    signatures: dict[tuple, inspect.Signature]

    def __new__(cls, func: Callable[..., RETURN]) -> "multidispatch[RETURN]":
        return functools.update_wrapper(dict.__new__(cls), func)  # type: ignore

    def __init__(self, func: Callable[..., RETURN]) -> None:
        self.pending = set()
        self.generics = []
        self.signatures = {}
        self[()] = func

    def __get__(self, instance, owner) -> Callable[..., RETURN]:
        return self if instance is None else types.MethodType(self, instance)

    def __setitem__(self, types: tuple, func: Callable):
        super().__setitem__(types, func)
        with contextlib.suppress(ValueError):
            signature = inspect.signature(func)
            self.signatures.setdefault(tuple(signature.parameters), signature)

    def __call__(self, *args: Any, **kwargs: Any) -> RETURN:
        """Resolve and dispatch to best method."""
        params = args
        if kwargs:
            for signature in self.signatures.values():  # pragma: no branch
                with contextlib.suppress(TypeError):
                    params = signature.bind(*args, **kwargs).args
                    break
        func = self.dispatch(*params)
        return func(*args, **kwargs)


class multimeta(type):
    """Convert all callables in namespace to multimethods."""

    class __prepare__(dict):
        def __init__(*args): ...

        def __setitem__(self, key, value):
            if callable(value):
                value = getattr(self.get(key), "register", multimethod)(value)
            super().__setitem__(key, value)
