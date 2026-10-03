import abc
import builtins
import collections
import contextvars
import contextlib
import functools
import inspect
import itertools
import types
import typing
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, Self, TypeVar, Union, get_type_hints

TypeAliasType = getattr(typing, "TypeAliasType", types.new_class(""))  # python <3.12

_missing_func = object()
_subclass_checks = contextvars.ContextVar("_subclass_checks", default=())
_instance_checks = contextvars.ContextVar("_instance_checks", default=())
_alias_context = contextvars.ContextVar("_alias_context", default=None)


def _alias_contains(alias, tp=None, seen=None) -> bool:
    tp = alias.__value__ if tp is None else tp
    seen = set() if seen is None else seen
    if id(tp) in seen:
        return False
    seen.add(id(tp))
    if tp is alias:
        return True
    if isinstance(tp, TypeAliasType):
        if _alias_contains(alias, tp.__value__, seen):
            return True
    return any(_alias_contains(alias, arg, seen) for arg in get_args(tp))


def _strict_isinstance(instance, cls) -> bool:
    """Check every element of a materialized container without consuming iterators."""
    recursive: set[tuple[int, object]] = set()
    return _strict_isinstance_check(instance, cls, recursive)


def _strict_isinstance_value(instance, cls, recursive: set[tuple[int, object]]):
    return _strict_isinstance_check(instance, cls, recursive)


def _strict_isinstance_check(instance, cls, recursive: set[tuple[int, object]]):
    if not issubclass(type(cls), subtype):
        return builtins.isinstance(instance, cls)
    origin = get_origin(cls)
    args = get_args(cls)
    recursive_alias = getattr(cls, "__recursive_alias__", None)
    if not args:
        return builtins.isinstance(instance, cls)
    if origin in (typing.Literal, builtins.type, typing.NewType):
        return builtins.isinstance(instance, cls)

    pair = id(instance), recursive_alias
    repeated = recursive_alias is not None and pair in recursive
    if repeated and origin in (typing.Union, types.UnionType):
        return True
    if origin in (typing.Union, types.UnionType):
        tracked = recursive_alias is not None and not repeated
        if tracked:
            recursive.add(pair)
        result = any(_strict_isinstance_value(instance, arg, recursive) for arg in args)
        if tracked and not result:
            recursive.discard(pair)
        return result

    if isinstance(instance, typing.Generic):
        return builtins.isinstance(instance, cls)
    if not builtins.isinstance(instance, origin) or isinstance(instance, Iterator):
        return False
    if origin is Callable:
        return builtins.isinstance(instance, cls)
    if repeated:
        return True
    tracked = recursive_alias is not None
    if tracked:
        recursive.add(pair)
    try:
        if isinstance(instance, tuple):
            if origin is builtins.tuple:
                if ... not in args and len(instance) != len(args):
                    return False
                item_args = (args[0],) * len(instance) if ... in args else args
            else:
                item_args = (args[0],) * len(instance)
            return all([
                _strict_isinstance_value(value, item_args[index], recursive)
                for index, value in enumerate(instance)
            ])
        if isinstance(instance, dict):
            if issubclass(origin, Mapping):
                return all([
                    _strict_isinstance_check(key, args[0], set()) for key in instance
                ]) and all([
                    _strict_isinstance_value(value, args[1], recursive)
                    for value in instance.values()
                ])
            return all([
                _strict_isinstance_check(key, args[0], set()) for key in instance
            ])
        if isinstance(instance, (list, set, frozenset)):
            return all([
                _strict_isinstance_value(value, args[0], recursive)
                for value in instance
            ])
        return True
    finally:
        if tracked:
            recursive.discard(pair)


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
                aliases = _alias_context.get()
                if aliases is not None and tp in aliases:
                    return aliases[tp]
                if not _alias_contains(tp):
                    return cls(tp.__value__)
                origin = get_origin(tp.__value__)
                if origin is typing.Literal:
                    origin = typing.Literal
                elif origin in (typing.Union, types.UnionType):
                    origin = types.UnionType
                else:
                    origin = origin or tp.__value__
                namespace = {
                    "__origin__": origin,
                    "__args__": (),
                    "__recursive_alias__": tp,
                }
                converted = type.__new__(cls, str(tp), (object,), namespace)
                if aliases is None:
                    aliases = {}
                    token = _alias_context.set(aliases)
                else:
                    token = None
                aliases[tp] = converted
                try:
                    converted.__args__ = tuple(map(cls, get_args(tp.__value__)))
                    return converted
                finally:
                    if token is not None:
                        _alias_context.reset(token)
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
        if isinstance(other, subtype):
            recursive = getattr(self, "__recursive_alias__", None)
            other_recursive = getattr(other, "__recursive_alias__", None)
            if recursive is not None or other_recursive is not None:
                return recursive is other_recursive
            return self.key() == other.key()
        return False

    def __hash__(self):
        recursive = getattr(self, "__recursive_alias__", None)
        if recursive is not None:
            return hash(recursive)
        return hash(self.key())

    def __subclasscheck__(self, subclass):
        check = id(subclass), id(self)
        active = set(_subclass_checks.get())
        if check in active:
            return True
        token = _subclass_checks.set(tuple(active | {check}))
        try:
            return self.__subclasscheck_impl__(subclass)
        finally:
            _subclass_checks.reset(token)

    def __subclasscheck_impl__(self, subclass):
        args = get_args(subclass)
        match origin := get_origin(subclass):
            case None:
                origin = subclass
            case typing.Literal:
                return all(isinstance(arg, self) for arg in args)
            case typing.Union | types.UnionType:
                return all(issubclass(cls, self) for cls in args)
        if origin is builtins.tuple and ... in args and ... not in self.__args__:
            return False
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
        check = id(instance), id(self)
        active = set(_instance_checks.get())
        if check in active:
            return True
        token = _instance_checks.set(tuple(active | {check}))
        try:
            return self.__instancecheck_impl__(instance)
        finally:
            _instance_checks.reset(token)

    def __instancecheck_impl__(self, instance):
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

    def strict_instances(self, *args) -> bool:
        """Return whether all arguments are instances, checking materialized containers fully."""
        return self.required <= len(args) and all(map(_strict_isinstance, args, self))


REGISTERED = TypeVar("REGISTERED", bound=Callable[..., Any])


class multimethod(dict):
    """A callable directed acyclic graph of methods."""

    __name__: str
    pending: set
    generics: list[tuple]  # positional bases which require instance checks
    strict_containers: bool

    def __new__(
        cls,
        func: Callable = _missing_func,
        strict_containers: bool | None = None,
    ):
        if func is _missing_func:
            return lambda wrapped: cls(wrapped, strict_containers=strict_containers)
        homonym = inspect.currentframe().f_back.f_locals.get(func.__name__)
        if isinstance(homonym, multimethod):
            return homonym

        self = functools.update_wrapper(dict.__new__(cls), func)
        self.pending = set()
        self.generics = []
        if strict_containers is not None:
            self.strict_containers = strict_containers
        return self

    def __init__(
        self,
        func: Callable = _missing_func,
        strict_containers: bool | None = None,
    ):
        if func is _missing_func:
            return
        if strict_containers is not None:
            self.strict_containers = strict_containers
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
        instance_check = signature.instances
        if getattr(self, "strict_containers", False):
            instance_check = signature.strict_instances
        matches = {
            key
            for key in list(self)
            if isinstance(key, signature) and instance_check(key, *args)
        }
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

    def __new__(
        cls,
        func: Callable[..., RETURN] = _missing_func,
        strict_containers: bool | None = None,
    ) -> "multidispatch[RETURN]":
        if func is _missing_func:
            return lambda wrapped: cls(wrapped, strict_containers=strict_containers)
        self = functools.update_wrapper(dict.__new__(cls), func)  # type: ignore
        if strict_containers is not None:
            self.strict_containers = strict_containers
        return self

    def __init__(
        self,
        func: Callable[..., RETURN] = _missing_func,
        strict_containers: bool | None = None,
    ) -> None:
        if func is _missing_func:
            return
        if strict_containers is not None:
            self.strict_containers = strict_containers
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
