import abc
import builtins
import collections
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
    named_types: dict[str, type]
    dispatch_masks: dict[str, bool]
    match_types: bool

    def __new__(cls, types: Iterable, required: int | None = None):
        return tuple.__new__(cls, map(subtype, types))

    def __init__(self, types: Iterable, required: int | None = None):
        self.required = len(self) if required is None else required
        self.named_types = {}
        self.dispatch_masks = {}
        self.match_types = True

    def __hash__(self):
        return tuple.__hash__(self)

    def __eq__(self, other) -> bool:
        if not tuple.__eq__(self, other):
            return False
        return not isinstance(other, signature) or self.dispatch_shape == other.dispatch_shape

    @property
    def dispatch_shape(self) -> tuple:
        sig = getattr(self, "sig", None)
        if sig is None:
            return ()
        return tuple(
            (
                param.name,
                param.kind,
                None if param.default is inspect.Parameter.empty else repr(param.default),
                self.named_types.get(param.name, object),
            )
            for param in sig.parameters.values()
        )

    @staticmethod
    def _annotation_type(annotation):
        return object if annotation is ... else subtype(annotation)

    def set_dispatch_types(self, sig: inspect.Signature, hints: dict[str, Any]):
        self.sig = sig
        positional_kinds = {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
        self.named_types = {}
        self.dispatch_masks = {}
        index = 0
        for param in sig.parameters.values():
            if param.kind in positional_kinds:
                if index < len(self):
                    annotation = self[index]
                    dispatch = True
                else:
                    annotation = hints.get(param.name, object)
                    dispatch = param.name in hints
                index += 1
            else:
                annotation = hints.get(param.name, object)
                dispatch = param.name in hints
            self.named_types[param.name] = self._annotation_type(annotation)
            self.dispatch_masks[param.name] = dispatch

    @classmethod
    def from_hints(cls, func: Callable) -> Self:
        """Return evaluated type hints for positional parameters in order."""
        if not hasattr(func, "__annotations__"):
            return cls(())
        type_hints = get_type_hints(func)
        sig = inspect.signature(func)
        positionals = {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        params: Iterable = sig.parameters.values()
        params = [param for param in params if param.kind in positionals]
        # missing annotations are padded with `object`, but trailing objects are unnecessary
        indices = [index for index, param in enumerate(params) if param.name in type_hints]
        params = params[: max(indices, default=-1) + 1]
        hints = [type_hints.get(param.name, object) for param in params]
        required = sum(param.default is param.empty for param in params)
        result = cls(hints, required)
        dispatch_masks = {
            param.name: param.name in type_hints
            for param in sig.parameters.values()
        }
        result.dispatch_masks = dispatch_masks
        result.named_types = {
            param.name: cls._annotation_type(type_hints.get(param.name, object))
            for param in sig.parameters.values()
        }
        return result

    def subtypes(self, *other) -> bool:
        """Return whether all types are subclasses."""
        return self.required <= len(other) and all(map(issubclass, other, self))

    def callable(self, *types) -> bool:
        """Check positional arity of associated function signature."""
        if any(isinstance(tp, DispatchKey) for tp in types):
            return True
        try:
            return not hasattr(self, "sig") or bool(self.sig.bind_partial(*types))
        except TypeError:
            return False

    def instances(self, *args) -> bool:
        """Return whether all arguments are instances."""
        return self.required <= len(args) and all(map(isinstance, args, self))

    def bind(self, *args, **kwargs) -> tuple[inspect.BoundArguments, set[str]] | None:
        """Bind arguments and check all values, including defaults and variadics."""
        sig = getattr(self, "sig", None)
        if sig is None:
            return None
        try:
            bound = sig.bind(*args, **kwargs)
        except TypeError:
            return None
        supplied = set(bound.arguments)
        bound.apply_defaults()
        if not self.bound_instances(bound, supplied):
            return None
        return bound, supplied

    def bound_instances(self, bound: inspect.BoundArguments, supplied: set[str]) -> bool:
        if not self.match_types:
            return True
        for name, param in self.sig.parameters.items():
            if not self.dispatch_masks.get(name, False):
                continue
            tp = self.named_types.get(name, object)
            value = bound.arguments[name]
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                values = value
            elif param.kind is inspect.Parameter.VAR_KEYWORD:
                values = value.values()
            else:
                if not isinstance(value, tp):
                    return False
                continue
            if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
                if not all(isinstance(item, tp) for item in values):
                    return False
        return True

    def dispatch_slots(
        self, bound: inspect.BoundArguments
    ) -> tuple[dict, dict]:
        positional = {}
        keywords = {}
        parameters = list(self.sig.parameters.values())
        fixed = [
            param
            for param in parameters
            if param.kind
            in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        ]
        var_positional = next(
            (param for param in parameters if param.kind is inspect.Parameter.VAR_POSITIONAL),
            None,
        )
        var_keyword = next(
            (param for param in parameters if param.kind is inspect.Parameter.VAR_KEYWORD),
            None,
        )
        for index, param in enumerate(fixed):
            if self.dispatch_masks.get(param.name, False):
                positional[index] = (2, self.named_types[param.name])
        if var_positional is not None:
            var_args = bound.arguments[var_positional.name]
            if var_args and self.dispatch_masks.get(var_positional.name, True):
                for offset in range(len(var_args)):
                    positional[len(fixed) + offset] = (1, self.named_types[var_positional.name])
        for param in parameters:
            if param.kind is inspect.Parameter.KEYWORD_ONLY and self.dispatch_masks.get(
                param.name, False
            ):
                keywords[param.name] = (2, self.named_types[param.name])
        if var_keyword is not None and self.dispatch_masks.get(var_keyword.name, True):
            for name in bound.arguments[var_keyword.name]:
                keywords[name] = (1, self.named_types[var_keyword.name])
        return positional, keywords

    def more_specific(
        self,
        other: "signature",
        binding,
        other_binding,
    ) -> bool:
        bound, _ = binding
        other_bound, _ = other_binding
        groups = zip(
            self.dispatch_slots(bound),
            other.dispatch_slots(other_bound),
            strict=True,
        )
        comparable = False
        for own_slots, other_slots in groups:
            for name in own_slots.keys() | other_slots.keys():
                own_rank, own_type = own_slots.get(name, (0, object))
                other_rank, other_type = other_slots.get(name, (0, object))
                if own_rank < other_rank or (
                    own_rank == other_rank and not issubclass(own_type, other_type)
                ):
                    return False
                comparable |= own_rank > other_rank or (
                    own_rank == other_rank and own_type is not other_type
                )
        return comparable


REGISTERED = TypeVar("REGISTERED", bound=Callable[..., Any])


class DispatchKey(tuple):
    """A cache key which distinguishes call shapes from registered type tuples."""

    __slots__ = ()

    __hash__ = tuple.__hash__

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and tuple.__eq__(self, other)


class multimethod(dict):
    """A callable directed acyclic graph of methods."""

    __name__: str
    pending: set
    generics: list[tuple]  # positional bases which require instance checks
    bound_dispatch: bool
    dispatch_origins: set[type]
    validate_positional: bool

    def __new__(cls, func):
        homonym = inspect.currentframe().f_back.f_locals.get(func.__name__)
        if isinstance(homonym, multimethod):
            return homonym

        self = functools.update_wrapper(dict.__new__(cls), func)
        self.pending = set()
        self.generics = []
        self.bound_dispatch = False
        self.dispatch_origins = set()
        self.validate_positional = False
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
        try:
            sig = inspect.signature(func)
            hints = get_type_hints(func)
        except (ValueError, NameError, AttributeError):
            pass
        else:
            types.set_dispatch_types(sig, hints)
        if isinstance(self, multidispatch) and len(types) == 0:
            types.match_types = False
        parents = types.parents = self.parents(types)
        if any(
            tuple.__eq__(existing, types)
            for existing in self
            if isinstance(existing, signature)
        ):
            self.bound_dispatch = True
        self.pop(types, None)  # ensure key is overwritten
        for key in self:
            if types.subtypes(*key) and (not parents or parents & key.parents):
                key.parents -= parents
                key.parents.add(types)
        for index, cls in enumerate(types):
            if origins := set(subtype.origins(cls)):
                self.generics += [()] * (index + 1 - len(self.generics))
                self.generics[index] = tuple(origins.union(self.generics[index]))
        for cls in types.named_types.values():
            self.dispatch_origins.update(subtype.origins(cls))
        sig = getattr(types, "sig", None)
        if sig is not None and any(
            param.kind
            in {
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }
            or (
                param.kind
                in {
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                }
                and param.default is not inspect.Parameter.empty
            )
            for param in sig.parameters.values()
        ):
            self.bound_dispatch = True
        positional_kinds = {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
        required_positional = (
            sum(
                param.default is inspect.Parameter.empty
                and param.kind in positional_kinds
                for param in sig.parameters.values()
            )
            if sig is not None
            else 0
        )
        if required_positional > types.required:
            self.validate_positional = True
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

    @staticmethod
    def dispatch_key(args: tuple, kwargs: dict[str, Any]) -> tuple:
        return DispatchKey((
            tuple(map(type, args)),
            DispatchKey(sorted((name, type(value)) for name, value in kwargs.items())),
        ))

    def uses_instance_checks(self, args: tuple, kwargs: dict[str, Any]) -> bool:
        values = itertools.chain(args, kwargs.values())
        return any(
            isinstance(value, origin)
            for value in values
            for origin in self.dispatch_origins
        )

    def filter_bound(self, candidates: dict[signature, tuple]) -> set[signature]:
        matches = set(candidates)
        for candidate in candidates:
            if any(
                other is not candidate
                and other.more_specific(candidate, candidates[other], candidates[candidate])
                for other in candidates
            ):
                matches.discard(candidate)
        return matches

    def bound_uses_instance_checks(self, candidates: dict[signature, tuple]) -> bool:
        for key, (bound, supplied) in candidates.items():
            for name, param in key.sig.parameters.items():
                if name in supplied or not key.dispatch_masks.get(name, False):
                    continue
                value = bound.arguments[name]
                values = (
                    value
                    if param.kind is inspect.Parameter.VAR_POSITIONAL
                    else value.values()
                    if param.kind is inspect.Parameter.VAR_KEYWORD
                    else (value,)
                )
                origins = set(subtype.origins(key.named_types[name]))
                if any(isinstance(item, origin) for item in values for origin in origins):
                    return True
        return False

    def dispatch(self, *args, **kwargs) -> Callable:
        self.evaluate()
        if kwargs or self.bound_dispatch:
            key = self.dispatch_key(args, kwargs)
            func = dict.get(self, key)
            if func is None:
                candidates = {}
                for sig in list(self):
                    if not isinstance(sig, signature):
                        continue
                    binding = sig.bind(*args, **kwargs)
                    if binding is not None:
                        candidates[sig] = binding
                if not candidates:
                    raise DispatchError(f"{self.__name__}: 0 methods found", key, set())
                if isinstance(self, multidispatch):
                    fallback = {
                        sig: binding
                        for sig, binding in candidates.items()
                        if not getattr(sig, "match_types", True)
                    }
                    specific = dict(candidates)
                    for sig in fallback:
                        del specific[sig]
                    if specific:
                        candidates = specific
                matches = self.filter_bound(candidates)
                if self.uses_instance_checks(args, kwargs) or (
                    self.bound_uses_instance_checks(candidates)
                ):
                    return self.select(key, matches)
                dict.setdefault(self, key, self.select(key, matches))
                func = dict.get(self, key)
            return func
        types = tuple(map(type, args))
        if not any(map(issubclass, types, self.generics)):
            return self[types]
        matches = {key for key in list(self) if isinstance(key, signature) and key.instances(*args)}
        matches -= {ancestor for match in matches for ancestor in match.parents}
        return self.select(types, matches)

    def __call__(self, *args, **kwargs):
        """Resolve and dispatch to best method."""
        func = self.dispatch(*args, **kwargs)
        if not kwargs and not self.bound_dispatch and self.validate_positional:
            try:
                inspect.signature(func).bind(*args, **kwargs)
            except TypeError as ex:
                raise DispatchError(f"Function {getattr(func, '__code__', func)}") from ex
        return func(*args, **kwargs)

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
    Allows dispatching on bound positional and keyword arguments.
    """

    def __new__(cls, func: Callable[..., RETURN]) -> "multidispatch[RETURN]":
        return functools.update_wrapper(dict.__new__(cls), func)  # type: ignore

    def __init__(self, func: Callable[..., RETURN]) -> None:
        self.pending = set()
        self.generics = []
        self.bound_dispatch = True
        self.dispatch_origins = set()
        self.validate_positional = False
        self[()] = func

    def __get__(self, instance, owner) -> Callable[..., RETURN]:
        return self if instance is None else types.MethodType(self, instance)

    def __call__(self, *args: Any, **kwargs: Any) -> RETURN:
        return super().__call__(*args, **kwargs)


class multimeta(type):
    """Convert all callables in namespace to multimethods."""

    class __prepare__(dict):
        def __init__(*args): ...

        def __setitem__(self, key, value):
            if callable(value):
                value = getattr(self.get(key), "register", multimethod)(value)
            super().__setitem__(key, value)
