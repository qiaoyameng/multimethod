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
    identity: tuple | None
    fallback: bool

    def __new__(cls, types: Iterable, required: int | None = None):
        return tuple.__new__(cls, map(subtype, types))

    def __init__(self, types: Iterable, required: int | None = None):
        self.required = len(self) if required is None else required
        self.identity = None
        self.fallback = False

    @classmethod
    def of(cls, func: Callable) -> Self:
        """Return a signature describing all parameters of ``func``.

        Positional parameter types remain the tuple contents, preserving the
        partial ordering used for pure positional dispatch.  The ``identity``
        additionally records the binding shape (parameter names, kinds, default
        presence and types, including keyword-only and variadic parameters), so
        that signatures which share positional types but differ in how
        arguments can bind remain distinct candidates.
        """
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
        types = cls(hints, required)
        types.describe(func, type_hints)
        return types

    @classmethod
    def from_hints(cls, func: Callable) -> Self:
        """Return evaluated type hints for positional parameters in order."""
        return cls.of(func)

    def describe(self, func: Callable, type_hints: dict | None = None):
        """Attach the concrete signature and the full binding-shape identity."""
        with contextlib.suppress(ValueError):
            self.sig = inspect.signature(func)
        sig = getattr(self, "sig", None)
        if sig is None:
            return self
        if type_hints is None:
            try:
                type_hints = get_type_hints(func)
            except Exception:
                type_hints = {}
        positionals = (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        )
        info = []
        positional_index = 0
        for param in sig.parameters.values():
            if param.kind not in positionals[:2] or positional_index >= len(self):
                cls = subtype(type_hints.get(param.name, object))
            else:
                cls = tuple.__getitem__(self, positional_index)
            if param.kind in positionals[:2]:
                positional_index += 1
            default = None if param.default is param.empty else param.default
            info.append((param.name, param.kind, default, cls))
        self.identity = tuple(info)
        return self

    def param_types(self, bound: inspect.BoundArguments):
        """Return declared ``(positional, keyword)`` type vectors for a binding.

        Defaults are applied first by the caller, so declared defaults for
        omitted parameters produce positional entries just like explicit ones.
        Positional-or-keyword parameters always occupy positional slots
        (``BoundArguments`` normalizes keyword calls for them), so equivalent
        positional and keyword calls produce identical vectors.  Keyword-only
        and variadic-keyword entries land in the keyword vector.
        """
        info = self.identity or ()
        by_name = {item[0]: item for item in info}
        pos_types: list = []
        for name, value in bound.arguments.items():
            item = by_name.get(name)
            if item is None:
                continue
            _, kind, _, cls = item
            if kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                pos_types.append(cls)
            elif kind == inspect.Parameter.VAR_POSITIONAL:
                pos_types.extend([cls] * len(value))
        keyword = {}
        var_keyword = next(
            (cls for name, kind, _, cls in self.identity if kind == inspect.Parameter.VAR_KEYWORD),
            object,
        )
        for name, value in bound.arguments.items():
            item = by_name.get(name)
            if item is not None and item[1] == inspect.Parameter.VAR_KEYWORD:
                keyword.update({sub_name: item[3] for sub_name in value})
                continue
            if item is None:
                keyword[name] = var_keyword
            elif item[1] == inspect.Parameter.KEYWORD_ONLY:
                keyword[name] = item[3]
        return pos_types, keyword

    def explicit(self, raw_bound: inspect.BoundArguments) -> tuple[int, set]:
        """Return ``(positional slots, keyword names)`` supplied by the call.

        Computed before defaults are applied, so omitted parameters and their
        declared defaults are distinguished from explicit arguments.
        """
        by_name = {item[0]: item for item in (self.identity or ())}
        positional = 0
        keyword = set()
        for name, value in raw_bound.arguments.items():
            item = by_name.get(name)
            if item is None:
                continue
            kind = item[1]
            if kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                positional += 1
            elif kind == inspect.Parameter.VAR_POSITIONAL:
                positional += len(value)
            elif kind == inspect.Parameter.KEYWORD_ONLY:
                keyword.add(name)
            elif kind == inspect.Parameter.VAR_KEYWORD:
                keyword.update(value)
        keyword.update(raw_bound.kwargs)
        return positional, keyword

    def check(self, bound: inspect.BoundArguments) -> bool:
        """Check every bound value (including defaults) against declared types."""
        pos_types, keyword = self.param_types(bound)
        if len(pos_types) != len(bound.args):
            return False
        values = itertools.chain(bound.args, (bound.arguments[name] for name in sorted(keyword)))
        types = itertools.chain(pos_types, (keyword[name] for name in sorted(keyword)))
        pairs = zip(values, types)
        return all(isinstance(value, cls) for value, cls in pairs)

    def __eq__(self, other) -> bool:
        if not isinstance(other, tuple) or len(self) != len(other) or not tuple.__eq__(self, other):
            return False
        identity = getattr(other, "identity", None)
        if self.identity is None or identity is None:
            return True
        return self.identity == identity

    def __hash__(self):
        return hash((tuple(self), self.identity))

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

    def lookup(self, types: tuple):
        """Find a registered signature equal to ``types`` by positional types."""
        return next(
            (
                key
                for key in list(self)
                if isinstance(key, signature) and tuple(key) == tuple(types)
            ),
            None,
        )

    def __getitem__(self, types: tuple):
        if isinstance(types, signature):
            try:
                return dict.__getitem__(self, types)
            except KeyError:
                return self.__missing__(types)
        hit = dict.get(self, types)  # cached plain-tuple lookup
        if hit is not None:
            return hit
        key = self.lookup(types)
        if key is not None:
            return dict.__getitem__(self, key)
        return self.__missing__(types)

    def get(self, key, default=None):
        if isinstance(key, signature):
            return dict.get(self, key, default)
        hit = dict.get(self, key)
        if hit is not None:
            return hit
        found = self.lookup(key)
        return dict.__getitem__(self, found) if found is not None else default

    def pop(self, key, *default):
        if not isinstance(key, signature):
            found = dict.get(self, key)
            if found is None:
                registered = self.lookup(key)
            else:
                registered = key
            if registered is None:
                if default:
                    return default[0]
                raise KeyError(key)
            key = registered
        return dict.pop(self, key, *default)

    def __contains__(self, key) -> bool:
        if isinstance(key, signature):
            return dict.__contains__(self, key)
        return self.lookup(key) is not None

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
        other.__ior__(self)
        other.pending = set(self.pending)
        other.generics = list(self.generics)
        if hasattr(self, "call_cache"):
            other.call_cache = dict(self.call_cache)
        return functools.update_wrapper(other, self)

    def __setitem__(self, types: tuple, func: Callable):
        self.clean()
        if not isinstance(types, signature):
            types = signature(types)
        parents = types.parents = self.parents(types)
        if getattr(types, "identity", None) is None:
            with contextlib.suppress(Exception):
                types.describe(func)
        dict.pop(self, types, None)  # ensure identical key is overwritten
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
        if not isinstance(types, signature):
            key = self.lookup(types)
            if key is None:
                raise KeyError(types)
            types = key
        dict.__delitem__(self, types)
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
        key = types if isinstance(types, signature) else self.lookup(types)
        if key is not None:
            return dict.__getitem__(self, key)
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
            # a TypeError raised inside the function body propagates unchanged;
            # only an error in the call itself is reported as a dispatch error
            if ex.__traceback__ is not None and ex.__traceback__.tb_next is not None:
                raise
            raise DispatchError(f"Function {func.__code__}") from ex

    def evaluate(self):
        """Evaluate any pending forward references."""
        while self.pending:
            func = self.pending.pop()
            self[signature.of(func)] = func

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
    Every candidate is bound with its own `inspect.Signature`, so positional-only,
    keyword-only, default and variadic parameters are all part of candidate
    selection; equivalent positional and keyword calls resolve identically.
    """

    class callkey(tuple):
        """Cache key capturing the bound argument shape.

        Holds the types of positional arguments and name-sorted keyword
        ``(name, type)`` pairs, which distinguishes bindings with the same
        types but a different shape, independent of keyword iteration order.
        """

        def __new__(cls, args: tuple, kwargs: dict):
            return tuple.__new__(
                cls,
                (
                    tuple(map(type, args)),
                    tuple(sorted((name, type(value)) for name, value in kwargs.items())),
                ),
            )

        def __eq__(self, other):
            return type(other) is type(self) and tuple.__eq__(self, other)

        def __hash__(self):
            return tuple.__hash__(self)

    def __new__(cls, func: Callable[..., RETURN]) -> "multidispatch[RETURN]":
        return functools.update_wrapper(dict.__new__(cls), func)  # type: ignore

    def __init__(self, func: Callable[..., RETURN]) -> None:
        self.pending = set()
        self.generics = []
        self.call_cache = {}
        base = signature(())
        base.describe(func)
        base.fallback = True
        self[base] = func

    def __get__(self, instance, owner) -> Callable[..., RETURN]:
        return self if instance is None else types.MethodType(self, instance)

    def clean(self):
        self.call_cache.clear()
        super().clean()

    def select_call(self, args: tuple, kwargs: dict) -> Callable:
        """Bind every candidate and pick the most specific applicable one."""
        self.evaluate()
        candidates = []
        for key in list(self):
            if not isinstance(key, signature):
                continue
            raw_bound = self.bind(key, args, kwargs)
            if raw_bound is None:
                continue
            bound = raw_bound
            bound.apply_defaults()
            explicit_positions, explicit_kw = key.explicit(raw_bound)
            base = getattr(key, "fallback", False)
            if not base and not key.check(bound):
                continue
            pos_types, kw_types = key.param_types(bound)
            candidates.append(
                (
                    key,
                    self[key],
                    pos_types,
                    kw_types,
                    explicit_positions,
                    explicit_kw,
                    base,
                )
            )
        funcs = self.most_specific(candidates)
        if len(funcs) == 1:
            return funcs.pop()
        types = (tuple(map(type, args)), tuple(sorted(kwargs)))
        raise DispatchError(f"{self.__name__}: {len(funcs)} methods found", types, funcs)

    @staticmethod
    def bind(key: signature, args: tuple, kwargs: dict):
        """Bind a candidate signature; return ``None`` when it cannot apply."""
        sig = getattr(key, "sig", None)
        if sig is None:
            return None
        try:
            return sig.bind(*args, **kwargs)
        except TypeError:
            return None

    @staticmethod
    def most_specific(candidates) -> set:
        """Return functions of candidates not strictly dominated by another.

        Specificity mirrors pure positional multimethod resolution:
        * only slots of arguments explicitly supplied by the call establish
          strict specificity: declared defaults fill in for applicability but
          do not let a longer signature win when an argument was omitted,
        * a shorter positional signature is more specific (only common
          positions are compared),
        * keyword slots are compared by name for explicitly supplied keywords.
        """
        vectors = []
        for _, func, pos, kw, n_args, explicit_kw, base in candidates:
            vectors.append((func, pos, kw, n_args, explicit_kw, base))
        # collapse equivalent shapes pointing at the same function object
        unique = {}
        for vector in vectors:
            shape = (vector[0], tuple(vector[1]), tuple(sorted(vector[2])))
            unique.setdefault(shape, vector)
        vectors = list(unique.values())

        def dominates(left, right) -> bool:
            _, left_pos, left_kw, left_n_pos, left_explicit, left_base = left
            _, right_pos, right_kw, right_n_pos, right_explicit, right_base = right
            if not left_base and not right_base and len(left_pos) > len(right_pos):
                return False
            n_pos = min(left_n_pos, right_n_pos, len(left_pos), len(right_pos))
            pos_pairs = tuple(zip(left_pos[:n_pos], right_pos[:n_pos]))
            shared_kw = set(left_kw) & set(right_kw)
            explicit_kw = left_explicit & right_explicit
            pairs = pos_pairs + tuple(
                (left_kw[name], right_kw[name]) for name in sorted(shared_kw)
            )
            if any(not (a is b or issubclass(a, b)) for a, b in pairs):
                return False
            stricter_slots = tuple(
                (left_kw[name], right_kw[name]) for name in sorted(shared_kw & explicit_kw)
            )
            stricter = (
                not left_base
                and not right_base
                and len(left_pos) < len(right_pos)
            ) or any(a is not b for a, b in pos_pairs + stricter_slots)
            return stricter

        winners = set()
        for index, candidate in enumerate(vectors):
            dominated = False
            for other, other_vector in enumerate(vectors):
                if other == index or other_vector[0] is candidate[0]:
                    continue
                if dominates(other_vector, candidate):
                    dominated = True
                    break
            if not dominated:
                winners.add(candidate[0])
        return winners

    def dispatch_call(self, args: tuple, kwargs: dict) -> Callable:
        """Resolve and cache the function for the given call shape."""
        self.evaluate()
        key = self.callkey(args, kwargs)
        with contextlib.suppress(KeyError):
            return self.call_cache[key]
        func = self.select_call(args, kwargs)
        self.call_cache[key] = func
        return func

    def __call__(self, *args: Any, **kwargs: Any) -> RETURN:
        """Resolve and dispatch to best method."""
        func = self.dispatch_call(args, kwargs)
        try:
            return func(*args, **kwargs)
        except TypeError as ex:
            if ex.__traceback__ is not None and ex.__traceback__.tb_next is not None:
                raise
            raise DispatchError(f"Function {func.__code__}") from ex


class multimeta(type):
    """Convert all callables in namespace to multimethods."""

    class __prepare__(dict):
        def __init__(*args): ...

        def __setitem__(self, key, value):
            if callable(value):
                value = getattr(self.get(key), "register", multimethod)(value)
            super().__setitem__(key, value)
