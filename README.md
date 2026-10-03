[![image](https://img.shields.io/pypi/v/multimethod.svg)](https://pypi.org/project/multimethod/)
![image](https://img.shields.io/pypi/pyversions/multimethod.svg)
[![image](https://pepy.tech/badge/multimethod)](https://pepy.tech/project/multimethod)
![image](https://img.shields.io/pypi/status/multimethod.svg)
[![build](https://github.com/coady/multimethod/actions/workflows/build.yml/badge.svg)](https://github.com/coady/multimethod/actions/workflows/build.yml)
[![image](https://codecov.io/gh/coady/multimethod/branch/main/graph/badge.svg)](https://codecov.io/gh/coady/multimethod/)
[![CodeQL](https://github.com/coady/multimethod/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/coady/multimethod/actions/workflows/github-code-scanning/codeql)
[![CodSpeed Badge](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/coady/multimethod)
[![image](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)

Multimethod provides a decorator for adding multiple argument dispatching to functions. The decorator creates a multimethod object as needed, and registers the function with its annotations.

There are several multiple dispatch libraries on PyPI. This one aims for simplicity and speed. With caching of argument types, it should be the fastest pure Python implementation possible.

## Usage
There are a couple options which trade-off dispatch speed for flexibility.

Decorator | Speed | Dispatch | Arguments
--------- | ----- | -------- | ---------
[multimethod](#multimethod) | faster | cached lookup | positional only
[multidispatch](#multidispatch) | slower | binds to first signature + cached lookup | positional + keywords

Dispatching on simple types which use `issubclass` is cached. Advanced types which use `isinstance` require a linear scan.

### multimethod
```python
from multimethod import multimethod


@multimethod
def func(x: int, y: float): ...
```

`func` is now a `multimethod` which will delegate to the above function, when called with arguments of the specified types. Subsequent usage will register new types and functions to the existing multimethod of the same name.

```python
@multimethod
def func(x: float, y: int): ...
```

Alternatively, functions can be explicitly registered in the same style as [functools.singledispatch](https://docs.python.org/3/library/functools.html#functools.singledispatch). Some static type checkers enforce that each name is defined only once.

```python
@func.register
def _(x: bool, y: bool): ...


@func.register(object, bool)
@func.register(bool, object)
def _(x, y):  # stackable without annotations
    ...
```

Multimethods are implemented as mappings from signatures to functions, and can be introspected as such.

```python
method[type, ...]  # get registered function
method[type, ...] = func  # register function by explicit types
```

Multimethods support any types that satisfy the `issubclass` relation, including abstract base classes in `collections.abc`. Note `typing` aliases do not support `issubclass` consistently, and are no longer needed for subscripts. Using ABCs instead is recommended. Subscripted generics are supported by custom `isinstance` checks:
* `Mapping[...]` - the first key-value pair is checked
* `tuple[...]` - all args are checked
* `Iterable[...]` - the first arg is checked
* `type[...]` - `issubclass` of type
* `Literal[...]` - equality and type match
* `Callable[[...], ...]` - parameter types are contravariant, return type is covariant

Naturally checking subscripts is slower, but the implementation is optimized, cached, and bypassed if no subscripts are in use in the parameter. Empty iterables match any subscript, but don't special-case how the types are normally resolved.

Dispatch resolution details:
* If an exact match isn't registered, the next closest method is called (and cached).
* If there are ambiguous methods - or none - a custom `TypeError` is raised.
* Keyword-only parameters may be annotated, but won't affect dispatching.
* A skipped annotation is equivalent to `: object`.
* If no types are specified, it will inherently match all arguments.

#### TypeVar unification

Repeated `TypeVar`s in one overload share one candidate-local binding. A candidate is considered only when every occurrence unifies to the same concrete runtime type. A failed binding is discarded before another candidate is tried, so it never leaves constraints behind.

* An unconstrained `T` is first checked against `object`; repeated occurrences must then have exactly the same runtime type.
* A `bound` is checked normally, but repeated occurrences still require the same concrete type.
* Every constraint is checked with ordinary subclass rules, after which the matching concrete runtime type must be identical.
* `bool` is a subtype of `int`, so `bool` satisfies an `int` bound or constraint. Unifying one occurrence to `bool` and another to `int` still fails.
* Parent and child instances are similarly accepted by the bound only; repeated unconstrained variables do not coerce a child to its parent.
* Empty `tuple`, `list`, or `dict` positions provide no type evidence. The variable stays unbound there, and an empty parameterized container can still match a concrete parameterized overload.
* Nested variables in `tuple`, `list`, `dict`, and `type[...]` are inferred from their contents or class argument. `tuple[T, ...]` requires every element to unify with the same `T`.

```python
from typing import TypeVar

from multimethod import multimethod

T = TypeVar("T")

@multimethod
def merge(left, right):
    return "object"

@merge.register
def _(left: T, right: T):
    return "same"

@merge.register
def _(left: int, right: int):
    return "int"

merge(1, 2)          # int: concrete int,int is more specific than T=int
merge(True, False)   # same: bool,bool unifies but has no bool overload
merge(True, 1)       # int: the concrete int,int overload accepts bool
merge(1, 1.0)        # object: int and float do not unify
```

After a TypeVar candidate unifies, it is compared with ordinary concrete overloads using the resolved signature. If the candidates remain equally specific, the usual ambiguity error is reported rather than silently choosing the generic overload. TypeVar dispatch caches a content-derived key in addition to the runtime types, so `[0]`, `[False]`, and `[]` retain separate results.

### multidispatch
`multidispatch` is a wrapper to provide compatibility with `functools.singledispatch`. It requires a base implementation and use of the `register` method instead of namespace lookup. It also supports dispatching on keyword arguments.

### instance checks
`subtype` provisionally provides `isinstance` and `issubclass` checks for generic types. When called on a non-generic, it will return the origin type.

```python
from multimethod import subtype

cls = subtype(int | list[int])

for obj in (0, False, [0], [False], []):
    assert isinstance(obj, cls)
for obj in (0.0, [0.0], (0,)):
    assert not isinstance(obj, cls)

for subclass in (int, bool, list[int], list[bool]):
    assert issubclass(subclass, cls)
for subclass in (float, list, list[float], tuple[int]):
    assert not issubclass(subclass, cls)
```

If a type implements a custom `__instancecheck__`, it can opt-in to dispatch (without caching) by registering its metaclass and bases with `subtype.origins`. `parametric` provides a convenient constructor, which will match the base class, predicate functions, and check attributes.

```python
from multimethod import parametric

Coroutine = parametric(Callable, inspect.iscoroutinefunction)
IntArray = parametric(array, typecode="i")
```

### classes
`classmethod` and `staticmethod` may be used with a multimethod, but must be applied _last_, i.e., wrapping the final multimethod definition after all functions are registered. For class and instance methods, `cls` and `self` participate in the dispatch as usual. They may be left blank when using annotations, otherwise use `object` as a placeholder.

```python
class Cls:
    # @classmethod: only works here if there are no more functions
    @multimethod
    def meth(cls, arg: str): ...

    # @classmethod: can not be used with `register` because `_` is not the multimethod
    @meth.register
    def _(cls, arg: int): ...

    meth = classmethod(meth)  # done with registering
```

If a method spans multiple classes, then the namespace lookup can not work. The `register` method can be used instead.

```python
class Base:
    @multimethod
    def meth(self, arg: str): ...


class Subclass(Base):
    @Base.meth.register
    def _(self, arg: int): ...
```

If the base class can not be modified, the decorator - like any - can be called explicitly.

```python
class Subclass(Base):
    meth = multimethod(Base.meth)
    ...
```

`multimeta` creates a class with a special namespace which converts callables to multimethods, and registers duplicate callables with the original.

```python
class Cls(metaclass=multimeta): ...  # all methods are multimethods
```

## Installation
```console
pip install multimethod
```

## Tests
100% branch coverage.

```console
pytest [--cov]
```
