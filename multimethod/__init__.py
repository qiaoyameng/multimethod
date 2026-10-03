import abc
import builtins
import collections
import contextlib
import functools
import inspect
import itertools
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
                return typevariable(tp)
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


class typevariable(subtype):
    """A normalized `TypeVar` which retains its identity within a signature."""

    __typevar__: TypeVar

    def __new__(cls, tp: TypeVar):
        self = typevariables.get(tp)
        if self is None:
            namespace = {"__origin__": object, "__args__": (), "__typevar__": tp}
            self = type.__new__(cls, str(tp), (object,), namespace)
            typevariables[tp] = self
        return self

    def key(self) -> tuple:
        return typevariable, self.__typevar__

    def __subclasscheck__(self, subclass):
        tp = self.__typevar__
        if isinstance(subclass, typevariable):
            return variable_satisfies(subclass.__typevar__, tp)
        if tp.__constraints__:
            return any(
                issubclass(subclass, subtype(constraint)) for constraint in tp.__constraints__
            )
        return tp.__bound__ is not None and issubclass(subclass, subtype(tp.__bound__))

    def __instancecheck__(self, instance):
        tp = self.__typevar__
        if tp.__constraints__:
            return isinstance(instance, tuple(map(subtype, tp.__constraints__)))
        return isinstance(instance, subtype(tp.__bound__ or object))


typevariables: dict[TypeVar, typevariable] = {}


def variable_satisfies(sub: TypeVar, sup: TypeVar) -> bool:
    """Return whether every type accepted by `sub` is accepted by `sup`."""
    if not sup.__bound__ and not sup.__constraints__:
        return not sub.__bound__ and not sub.__constraints__
    sub_types = sub.__constraints__ or (sub.__bound__ or object,)
    if sup.__constraints__:
        bound = subtype(Union[sup.__constraints__])
    else:
        bound = subtype(sup.__bound__ or object)
    return all(issubclass(subtype(tp), bound) for tp in sub_types)


def has_typevariable(tp) -> bool:
    if isinstance(tp, typevariable):
        return True
    return any(has_typevariable(arg) for arg in get_args(tp))


def typevariables_in(tp, variables: set[TypeVar]) -> None:
    if isinstance(tp, typevariable):
        variables.add(tp.__typevar__)
    for arg in get_args(tp):
        typevariables_in(arg, variables)


def pattern_issubclass(sub, sup, bindings: dict[TypeVar, object]) -> bool:
    """Check normalized signature types, unifying repeated TypeVars in `sup`."""
    if not has_typevariable(sub) and not has_typevariable(sup):
        return issubclass(sub, sup)
    if isinstance(sup, typevariable):
        variable = sup.__typevar__
        existing = bindings.get(variable)
        if existing is not None:
            if isinstance(existing, typevariable):
                return sub is existing
            return not isinstance(sub, typevariable) and issubclass(sub, existing)
        if isinstance(sub, typevariable):
            if not variable_satisfies(sub.__typevar__, variable):
                return False
        elif variable.__bound__ is not None or variable.__constraints__:
            if not issubclass(sub, sup):
                return False
        bindings[variable] = sub
        return True
    if isinstance(sub, typevariable):
        return issubclass(sub, sup)

    sub_origin = get_origin(sub)
    if sub_origin is None:
        sub_origin = sub
    sup_origin = get_origin(sup)
    if sup_origin is None:
        sup_origin = sup
    if sup_origin in (typing.Union, types.UnionType):
        return any(
            pattern_issubclass(sub, arg, bindings.copy())
            for arg in get_args(sup)
        )
    if sub_origin in (typing.Union, types.UnionType):
        return all(pattern_issubclass(arg, sup, bindings) for arg in get_args(sub))
    if not isinstance(sup_origin, type) or not issubclass(sub_origin, sup_origin):
        return False

    sub_args = get_args(sub)
    sup_args = get_args(sup)
    if not sup_args:
        return True
    if sup_origin is tuple and ... in sup_args:
        return all(
            arg is ... or pattern_issubclass(arg, sup_args[0], bindings) for arg in sub_args
        )
    if len(sub_args) != len(sup_args):
        return False
    return all(
        pattern_issubclass(sub_arg, sup_arg, bindings)
        for sub_arg, sup_arg in zip(sub_args, sup_args)
    )


def bind_typevariable(bindings: dict[TypeVar, type], variable: TypeVar, tp: type) -> bool:
    existing = bindings.get(variable)
    if existing is None:
        bindings[variable] = tp
        return True
    return tp is existing


def typevariable_accepts(tp: TypeVar, instance) -> bool:
    if tp.__constraints__:
        return any(
            issubclass(type(instance), subtype(constraint)) for constraint in tp.__constraints__
        )
    return issubclass(type(instance), subtype(tp.__bound__ or object))


def instance_match(tp, instance, bindings: dict[TypeVar, type]) -> bool:
    """Match an instance against a normalized parameterized signature type."""
    if isinstance(tp, typevariable):
        variable = tp.__typevar__
        return typevariable_accepts(variable, instance) and bind_typevariable(
            bindings, variable, type(instance)
        )
    if not has_typevariable(tp):
        return isinstance(instance, tp)

    origin = get_origin(tp)
    args = get_args(tp)
    if origin in (typing.Union, types.UnionType):
        for arg in args:
            candidate = bindings.copy()
            if instance_match(arg, instance, candidate):
                bindings.update(candidate)
                return True
        return False
    if not isinstance(origin, type) or not isinstance(instance, origin):
        return False
    if not args:
        return True

    if origin is tuple:
        if ... in args:
            return len(args) == 2 and all(
                instance_match(args[0], item, bindings) for item in instance
            )
        if len(instance) != len(args):
            return False
        return all(
            instance_match(arg, item, bindings) for arg, item in zip(args, instance)
        )
    if issubclass(origin, Mapping):
        if not instance:
            return True
        key, value = next(iter(instance.items()))
        return (
            len(args) == 2
            and instance_match(args[0], key, bindings)
            and instance_match(args[1], value, bindings)
        )
    if issubclass(origin, Iterable):
        if isinstance(instance, Iterator):
            return False
        if not instance:
            return True
        item = next(iter(instance))
        return len(args) == 1 and instance_match(args[0], item, bindings)
    return isinstance(instance, tp)


def type_shape(value):
    """Return the runtime type shape used by subscripted dispatch cache keys."""
    tp = type(value)
    if isinstance(value, type):
        return tp
    if issubclass(tp, tuple):
        return (tp, *map(type_shape, value))
    if isinstance(value, Mapping):
        if not value:
            return (tp, ())
        key, item = next(iter(value.items()))
        return tp, type_shape(key), type_shape(item)
    if isinstance(value, (str, bytes)):
        return tp
    if isinstance(value, Iterable) and not isinstance(value, Iterator):
        if not value:
            return tp, ()
        return tp, type_shape(next(iter(value)))
    return tp


def resolve_type(tp, bindings: dict[TypeVar, type]):
    if isinstance(tp, typevariable):
        return bindings.get(tp.__typevar__, tp)
    args = tuple(resolve_type(arg, bindings) for arg in get_args(tp))
    origin = get_origin(tp)
    if origin is None or not args:
        return tp
    return (origin, args)


def resolved_issubclass(sub, sup) -> bool:
    if sub is ... or sup is ...:
        return sub is sup
    if isinstance(sub, typevariable):
        sub = subtype(sub.__typevar__.__bound__ or object)
    if isinstance(sup, typevariable):
        sup = subtype(sup.__typevar__.__bound__ or object)
    if isinstance(sub, type) and isinstance(sup, type):
        return issubclass(sub, sup)
    if not isinstance(sub, tuple) or not isinstance(sup, tuple):
        if isinstance(sup, type):
            sub = sub[0] if isinstance(sub, tuple) else sub
            return isinstance(sub, type) and issubclass(sub, sup)
        return False

    sub_origin, sub_args = sub
    sup_origin, sup_args = sup
    if not issubclass(sub_origin, sup_origin):
        return False
    if not sup_args:
        return True
    if sup_origin is tuple and ... in sup_args:
        return all(arg is ... or resolved_issubclass(arg, sup_args[0]) for arg in sub_args)
    if len(sub_args) != len(sup_args):
        return False
    return all(map(resolved_issubclass, sub_args, sup_args))


subtype.origins.register(typevariable, lambda cls: (object,))


class signature(tuple):
    """A tuple of types that supports partial ordering."""

    required: int
    parents: set
    sig: inspect.Signature

    def __new__(cls, types: Iterable, required: int | None = None):
        return tuple.__new__(cls, map(subtype, types))

    def __init__(self, types: Iterable, required: int | None = None):
        self.required = len(self) if required is None else required

    @property
    def generic(self) -> bool:
        return any(map(has_typevariable, self))

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
        if self.required > len(other):
            return False
        bindings = {}
        return all(
            pattern_issubclass(subtype(sub), candidate, bindings)
            for sub, candidate in zip(other, self)
        )

    def callable(self, *types) -> bool:
        """Check positional arity of associated function signature."""
        try:
            return not hasattr(self, "sig") or bool(self.sig.bind_partial(*types))
        except TypeError:
            return False

    def instances(self, *args) -> bool:
        """Return whether all arguments are instances."""
        return self.bind(*args) is not None

    def bind(self, *args) -> dict[TypeVar, type] | None:
        """Return a local TypeVar binding for the arguments when this signature matches."""
        if self.required > len(args):
            return None
        if not any(map(has_typevariable, self)):
            return {} if all(map(isinstance, args, self)) else None
        variables = set()
        for candidate in self:
            typevariables_in(candidate, variables)
        bindings = {variable: None for variable in variables}
        if all(
            instance_match(candidate, arg, bindings)
            for candidate, arg in zip(self, args)
        ):
            return bindings
        return None

    def dominates(self, other: "signature", bindings: dict, other_bindings: dict) -> bool:
        if self is other:
            return False
        if len(self) < len(other):
            return False
        if self and not other:
            return True
        this_side = True
        other_side = True
        for this_type, other_type in zip(self, other):
            this_type = resolve_type(this_type, bindings)
            other_type = resolve_type(other_type, other_bindings)
            this_side &= resolved_issubclass(this_type, other_type)
            other_side &= resolved_issubclass(other_type, this_type)
        if not this_side:
            return False
        if not other_side:
            return True
        if not self.generic and other.generic:
            return True
        if self.generic and other.generic:
            bindings = {}
            this_specific = all(
                pattern_issubclass(this_type, other_type, bindings)
                for this_type, other_type in zip(self, other)
            )
            bindings = {}
            other_specific = all(
                pattern_issubclass(other_type, this_type, bindings)
                for other_type, this_type in zip(other, self)
            )
            return this_specific and not other_specific
        return False


REGISTERED = TypeVar("REGISTERED", bound=Callable[..., Any])
dispatch_cache_key = object()


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
        if any(
            isinstance(key, signature) and key.generic
            for key in list(self)
        ):
            cache_key = (dispatch_cache_key, *map(type_shape, args))
            cached = self.get(cache_key)
            if cached is not None:
                return cached
        else:
            cache_key = None
        matches = {
            key: key.bind(*args)
            for key in list(self)
            if isinstance(key, signature) and key.callable(*types)
        }
        matches = {key: bindings for key, bindings in matches.items() if bindings is not None}
        candidates = set(matches)
        concrete = {match for match in matches if not match.generic and match}
        for ancestor in {
            ancestor
            for match in concrete
            for ancestor in match.parents
            if ancestor in concrete
        }:
            candidates.discard(ancestor)
        if any(key.generic for key in matches):
            candidates -= {
                other
                for key in matches
                for other in matches
                if key.generic or other.generic
                if key.dominates(other, matches[key], matches[other])
            }
        func = self.select(types, candidates)
        if any(key.generic for key in candidates):
            cache_key = (dispatch_cache_key, *map(type_shape, args))
            super().__setitem__(cache_key, func)
        return func

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
