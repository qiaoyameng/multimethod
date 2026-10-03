import abc
import builtins
import collections
import contextlib
import contextvars
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


_next_method: contextvars.ContextVar = contextvars.ContextVar("next_method", default=None)


def next_method():
    """Call the next method in the current `combined` chain.

    The next method is bound to the candidate snapshot of the current call,
    and is invoked with the same arguments; it takes no arguments here.
    It may be called multiple times, re-invoking the same next method.
    Raises `DispatchError` if there is no next method.
    """
    method = _next_method.get()
    if method is None:
        raise DispatchError("next_method: no next method")
    return method()


class qualified(multimethod):
    """A multimethod which cleans its owner's chain cache when updated."""

    owner: "combined"

    def __setitem__(self, types: tuple, func: Callable):
        super().__setitem__(types, func)
        self.owner.clean()

    def __delitem__(self, types: tuple):
        super().__delitem__(types)
        self.owner.clean()


class combined(multimethod):
    """A multimethod with explicit method combination.

    Methods are registered with one of four qualifiers:

    * `primary` (or `register`): the main methods. The most specific one runs,
      and may delegate to the next less specific one via `next_method`.
    * `before`: run before the primary method, from most to least specific.
    * `after`: run after the primary method returns normally, from least to
      most specific. Skipped if an exception unwinds the chain.
    * `around`: wrap the whole chain, from most to least specific. Each may
      short-circuit or delegate to the rest of the chain via `next_method`.

    All applicable methods of a qualifier must be totally ordered by
    specificity; incomparable candidates raise `DispatchError`.
    `next_method` is bound to the candidate snapshot of the current call,
    and exceptions raised inside methods propagate unchanged.
    """

    cache: dict
    befores: qualified
    afters: qualified
    arounds: qualified

    def __init__(self, func: Callable):
        if not hasattr(self, "cache"):
            self.cache = {}
            self.befores = self.qualifier("befores")
            self.afters = self.qualifier("afters")
            self.arounds = self.qualifier("arounds")
        super().__init__(func)

    def qualifier(self, name: str) -> qualified:
        """Return a new empty qualifier multimethod owned by this one."""
        methods = dict.__new__(qualified)
        methods.owner = self
        methods.pending = set()
        methods.generics = []
        methods.__name__ = f"{self.__name__}.{name}"
        return methods

    def primary(self, *args) -> Callable:
        """Decorator for registering a primary method, same as `register`."""
        return self.register(*args)

    def before(self, *args) -> Callable:
        """Decorator for registering a before method."""
        return self.befores.register(*args)

    def after(self, *args) -> Callable:
        """Decorator for registering an after method."""
        return self.afters.register(*args)

    def around(self, *args) -> Callable:
        """Decorator for registering an around method."""
        return self.arounds.register(*args)

    def clean(self):
        """Empty the chain cache."""
        super().clean()
        self.cache.clear()

    def evaluate(self):
        """Evaluate any pending forward references."""
        super().evaluate()
        self.befores.evaluate()
        self.afters.evaluate()
        self.arounds.evaluate()

    def copy(self):
        """Return a new `combined` with the same methods."""
        other = dict.__new__(type(self)).__ior__(self)
        other.__name__ = self.__name__
        other.pending = set(self.pending)
        other.generics = list(self.generics)
        other.cache = {}
        for name in ("befores", "afters", "arounds"):
            methods = getattr(self, name)
            replica = other.qualifier(name)
            dict.__ior__(replica, methods)
            replica.pending = set(methods.pending)
            replica.generics = list(methods.generics)
            setattr(other, name, replica)
        return other

    def __missing__(self, types: tuple):
        """Primary methods are stored exactly; chains are computed by `chain`."""
        raise KeyError(types)

    def applicable(self, methods: multimethod, types: tuple, args: tuple | None = None) -> tuple:
        """Return applicable methods of given types, from most to least specific."""
        if args is None:
            keys = {key for key in methods if isinstance(key, signature) and key.subtypes(*types)}
        else:
            keys = {key for key in methods if isinstance(key, signature) and key.instances(*args)}
        keys = {key for key in keys if key.callable(*types)}
        ordered = []
        while keys:
            most = {
                key
                for key in keys
                if not any(key.subtypes(*other) for other in keys if other is not key)
            }
            if len(most) != 1:
                funcs = {key: methods[key] for key in most}
                msg = f"{self.__name__}: {len(funcs)} methods found"
                raise DispatchError(msg, types, set(funcs))
            (key,) = most
            ordered.append(methods[key])
            keys.discard(key)
        return tuple(ordered)

    def compute(self, types: tuple, args: tuple | None = None) -> tuple:
        """Return the candidate snapshot for given types: (arounds, befores, primaries, afters)."""
        return (
            self.applicable(self.arounds, types, args),
            self.applicable(self.befores, types, args),
            self.applicable(self, types, args),
            self.applicable(self.afters, types, args),
        )

    def chain(self, *args) -> tuple:
        """Return the candidate snapshot for the given arguments."""
        self.evaluate()
        types = tuple(map(type, args))
        methods = (self, self.befores, self.afters, self.arounds)
        if any(any(map(issubclass, types, method.generics)) for method in methods):
            return self.compute(tuple(map(subtype, types)), args)
        key = tuple(map(subtype, types))
        try:
            return self.cache[key]
        except KeyError:
            return self.cache.setdefault(key, self.compute(key))

    def __call__(self, *args, **kwargs):
        """Resolve and dispatch to the combined method chain."""
        arounds, befores, primaries, afters = self.chain(*args)

        def run(func, continuation):
            token = _next_method.set(continuation)
            try:
                return func(*args, **kwargs)
            finally:
                _next_method.reset(token)

        def run_primary(index):
            continuation = (lambda: run_primary(index + 1)) if index + 1 < len(primaries) else None
            return run(primaries[index], continuation)

        def run_around(index):
            continuation = (lambda: run_around(index + 1)) if index + 1 < len(arounds) else body
            return run(arounds[index], continuation)

        def body():
            if not primaries:
                raise DispatchError(
                    f"{self.__name__}: 0 methods found", tuple(map(type, args)), set()
                )
            for func in befores:
                run(func, None)
            result = run_primary(0)
            for func in reversed(afters):
                run(func, None)
            return result

        return run_around(0) if arounds else body()


class multimeta(type):
    """Convert all callables in namespace to multimethods."""

    class __prepare__(dict):
        def __init__(*args): ...

        def __setitem__(self, key, value):
            if callable(value):
                value = getattr(self.get(key), "register", multimethod)(value)
            super().__setitem__(key, value)
