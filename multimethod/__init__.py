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
                limit = (
                    cls(Union[tp.__constraints__])
                    if tp.__constraints__
                    else cls(tp.__bound__) if tp.__bound__ is not None else object
                )
                namespace = {"__typevar__": tp, "__origin__": limit, "__args__": ()}
                return type.__new__(cls, str(tp), (limit,), namespace)
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
        if getattr(self, "__typevar__", None) is not None:
            return ("typevar", self.__typevar__)
        return self.__origin__, *self.__args__

    def __eq__(self, other) -> bool:
        return isinstance(other, subtype) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    def limit(self):
        limit = self.__origin__
        while getattr(limit, "__typevar__", None) is not None:
            limit = limit.__origin__
        return limit

    def __subclasscheck__(self, subclass):
        typevar = getattr(self, "__typevar__", None)
        if typevar is not None:
            other = getattr(subclass, "__typevar__", None)
            if other is not None:
                return other is typevar
            limit = self.limit()
            subclass_origin = get_origin(subclass)
            if subclass_origin in (typing.Union, types.UnionType):
                return all(issubclass(arg, limit) for arg in get_args(subclass))
            key = (id(self), id(subclass))
            if key in TYPEVAR_SUBCLASS_CHECKS:
                return isinstance(limit, type) and limit in getattr(subclass, "__mro__", ())
            TYPEVAR_SUBCLASS_CHECKS.add(key)
            try:
                return issubclass(subclass, limit)
            finally:
                TYPEVAR_SUBCLASS_CHECKS.discard(key)
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
        if getattr(self, "__typevar__", None) is not None:
            return isinstance(instance, self.limit())
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
        if getattr(self, "__typevar__", None) is not None:
            origins = list(subtype.origins(self.limit()))
            yield from origins or (object,)
            return
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


def typevar(tp):
    return getattr(tp, "__typevar__", None)


TYPEVAR_SUBCLASS_CHECKS: set[tuple[int, int]] = set()


def generic_origin(tp):
    origin = get_origin(tp)
    origins = (builtins.tuple, builtins.list, builtins.dict, builtins.type)
    return origin if origin in origins else None


def contains_typevar(tp) -> bool:
    if typevar(tp) is not None:
        return True
    origin = generic_origin(tp)
    return origin is not None and any(
        arg is not Ellipsis and contains_typevar(arg) for arg in get_args(tp)
    )


def reconstruct(origin, args):
    if origin is builtins.tuple:
        return subtype(tuple[args])
    if origin is builtins.list:
        return subtype(list[args[0]])
    if origin is builtins.dict:
        return subtype(dict[args[0], args[1]])
    return subtype(type[args[0]])


def upper_type(tp):
    tv = typevar(tp)
    if tv is not None:
        if tv.__constraints__:
            return subtype(Union[tv.__constraints__])
        return subtype(tv.__bound__) if tv.__bound__ is not None else object
    origin = generic_origin(tp)
    if origin is None:
        return tp
    return reconstruct(
        origin,
        tuple(Ellipsis if arg is Ellipsis else upper_type(arg) for arg in get_args(tp)),
    )


def bind_type(annotation, arg, bindings: dict, as_type=False) -> bool:
    tv = typevar(annotation)
    if tv is not None:
        allowed = issubclass(arg, annotation.limit()) if as_type else isinstance(arg, annotation)
        if not allowed:
            return False
        actual = arg if as_type else type(arg)
        previous = bindings.get(tv)
        if previous is None:
            bindings[tv] = actual
            return True
        return previous is actual
    origin = generic_origin(annotation)
    if origin is builtins.type:
        return isinstance(arg, type) and bind_type(
            get_args(annotation)[0], arg, bindings, as_type=True
        )
    if origin is builtins.tuple:
        if not isinstance(arg, tuple):
            return False
        if not contains_typevar(annotation):
            return isinstance(arg, annotation)
        args = get_args(annotation)
        if Ellipsis in args:
            return all(bind_type(args[0], item, bindings) for item in arg)
        return len(args) == len(arg) and all(
            map(bind_type, args, arg, itertools.repeat(bindings))
        )
    if origin is builtins.list:
        if not isinstance(arg, list):
            return False
        if not contains_typevar(annotation):
            return isinstance(arg, annotation)
        return all(bind_type(get_args(annotation)[0], item, bindings) for item in arg)
    if origin is builtins.dict:
        if not isinstance(arg, dict) or not contains_typevar(annotation):
            return isinstance(arg, annotation)
        return all(
            bind_type(get_args(annotation)[0], key, bindings)
            and bind_type(get_args(annotation)[1], value, bindings)
            for key, value in arg.items()
        )
    return isinstance(arg, annotation)


def substitute_type(tp, bindings: dict):
    tv = typevar(tp)
    if tv is not None:
        return bindings.get(tv, upper_type(tp))
    origin = generic_origin(tp)
    if origin is None:
        return tp
    return reconstruct(
        origin,
        tuple(
            Ellipsis if arg is Ellipsis else substitute_type(arg, bindings)
            for arg in get_args(tp)
        ),
    )


def cacheable_type(tp) -> bool:
    tv = typevar(tp)
    if tv is not None:
        return cacheable_type(tp.limit()) if isinstance(tp.limit(), subtype) else True
    origin = get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        return all(cacheable_type(arg) for arg in get_args(tp))
    if origin is typing.Literal or isinstance(tp, parametric):
        return False
    if origin is generic_origin(tp):
        return all(arg is Ellipsis or cacheable_type(arg) for arg in get_args(tp))
    return origin is None


def evidence(arg, seen: dict):
    arg_id = id(arg)
    if arg_id in seen:
        return ("cycle", arg_id)
    if isinstance(arg, (tuple, list, dict)):
        seen[arg_id] = None
        if isinstance(arg, dict):
            items = frozenset(
                (evidence(key, seen), evidence(value, seen)) for key, value in arg.items()
            )
        else:
            items = tuple(evidence(item, seen) for item in arg)
        return (type(arg), items)
    if isinstance(arg, type):
        return (type, arg)
    return type(arg)


class generic_key(tuple):
    """A cache key which includes runtime evidence for generic arguments."""


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

    def bind_instances(self, *args) -> dict | None:
        """Bind TypeVars shared by this signature to one set of arguments."""
        if self.required > len(args):
            return None
        bindings: dict = {}
        for annotation, arg in zip(self, args):
            if not bind_type(annotation, arg, bindings):
                return None
        return bindings


REGISTERED = TypeVar("REGISTERED", bound=Callable[..., Any])


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
        self.typevars = set()
        self.cacheable = True
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
        other = dict.__new__(type(self))
        dict.__ior__(other, self)
        other.pending = set(self.pending)
        other.generics = self.generics.copy()
        other.typevars = set(self.typevars)
        other.cacheable = self.cacheable
        functools.update_wrapper(other, self)
        return other

    def __setitem__(self, types: tuple, func: Callable):
        self.clean()
        if not isinstance(types, signature):
            types = signature(types)
        dict.pop(self, types, None)
        if any(contains_typevar(tp) for tp in types):
            self.typevars.add(types)
        self.cacheable = all(
            all(cacheable_type(tp) for tp in key)
            for key in [*self, types]
            if isinstance(key, signature)
        )
        parents = types.parents = self.parents(types)
        with contextlib.suppress(ValueError):
            types.sig = inspect.signature(func)
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
        self.typevars.discard(types)
        self.cacheable = all(
            all(cacheable_type(tp) for tp in key)
            for key in self
            if isinstance(key, signature)
        )
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

    @staticmethod
    def issubtype(subtype_type, supertype_type) -> bool:
        try:
            return issubclass(subtype_type, supertype_type)
        except TypeError:
            return False

    def candidates(self, args, types: tuple) -> set[tuple[signature, tuple, tuple, tuple]]:
        candidates = set()
        for key in list(self):
            if not isinstance(key, signature) or not key.callable(*types):
                continue
            bindings = key.bind_instances(*args)
            if bindings is None:
                continue
            resolved = tuple(substitute_type(tp, bindings) for tp in key[: len(types)])
            upper = tuple(upper_type(tp) for tp in key[: len(types)])
            candidates.add((key, resolved, upper, tuple(bindings.items())))
        return candidates

    @staticmethod
    def dominates(left, right) -> bool:
        _, left_resolved, left_upper, _ = left
        _, right_resolved, right_upper, _ = right
        if len(left_resolved) != len(right_resolved):
            return len(left_resolved) > len(right_resolved)
        if not all(map(multimethod.issubtype, left_resolved, right_upper)):
            return False
        left_narrower = map(multimethod.issubtype, left_resolved, right_resolved)
        right_narrower = map(multimethod.issubtype, right_resolved, left_resolved)
        if all(left_narrower) and not all(right_narrower):
            return True
        if not all(left_narrower) or not all(right_narrower):
            return False
        left_static = map(multimethod.issubtype, left_upper, right_upper)
        right_static = map(multimethod.issubtype, right_upper, left_upper)
        return all(left_static) and not all(right_static)

    def dispatch_typevars(self, args, types: tuple) -> Callable:
        candidates = self.candidates(args, types)
        while True:
            for candidate in tuple(candidates):
                dominated = any(
                    other is not candidate and self.dominates(other, candidate)
                    for other in candidates
                )
                if dominated:
                    candidates.remove(candidate)
                    break
            else:
                break
        return self.select(types, {candidate[0] for candidate in candidates})

    def dispatch(self, *args) -> Callable:
        self.evaluate()
        types = tuple(map(type, args))
        if self.typevars:
            if self.cacheable:
                key = generic_key((*types, tuple(evidence(arg, {}) for arg in args)))
                if dict.__contains__(self, key):
                    return dict.__getitem__(self, key)
            func = self.dispatch_typevars(args, types)
            if self.cacheable:
                dict.__setitem__(
                    self,
                    generic_key((*types, tuple(evidence(arg, {}) for arg in args))),
                    func,
                )
            return func
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
        self.typevars = set()
        self.cacheable = True
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
