"""Static ambiguity check against small, hand-built registries.

Run:  python examples/registry_analysis.py

Each registry only *declares* signatures; the analyzer never dispatches, never
calls a registered function, never constructs user classes and never evaluates
user predicates.  Reports are grouped into the three supported verdicts:

* ambiguous - a concrete witness demonstrates two equally specific candidates;
* covered   - a third, more specific overload owns the whole intersection
              (already-resolved diamonds stay quiet);
* unknown   - complex ABCs, parametric predicates or future MI subclasses.
"""

import typing
from collections.abc import Hashable, Iterable

from multimethod import multimethod, parametric
from multimethod.analysis import analyze


def registry(name, *signatures):
    method = multimethod(lambda *args: None)
    del method[()]  # remove the incidental fallback so every key is intentional
    for index, types in enumerate(signatures):
        method[tuple(types)] = lambda *args, _i=index: _i
    print(f"\n### {name}")
    print(analyze(method))
    return method


print("==================== TRUE AMBIGUITY ====================")

# Two orthogonal refinements tie on values in the cross-product (0, 0).
registry(
    "two-argument cross-product: (object, int) vs (int, object)",
    [object, object], [object, int], [int, object],
)

# Literal sets overlap on exactly one safe value.
registry(
    "literal overlap on 'b'",
    [typing.Literal["a", "b"]], [typing.Literal["b", "c"]],
)


print("\n==================== COVERED (resolved, no warning) ====================")

# The same cross-product plus the diagonal arm (int, int) resolves it.
registry(
    "diamond resolved by third overload (int, int)",
    [object, object], [object, int], [int, object], [int, int],
)

# Single-inheritance chain: bool is a strict sub-region, never a tie.
registry(
    "normal inheritance object -> int -> bool",
    [object], [int], [bool],
)

# Known concrete containers with incompatible layouts do not overlap.
registry(
    "concrete containers with incompatible layouts: list[int] vs dict[str, int]",
    [list[int]], [dict[str, int]],
)


print("\n==================== CANNOT PROVE (unknown, with reasons) ====================")

# Iterable and Hashable share values (e.g. str) but no closed proof is possible.
registry(
    "complex ABCs: Iterable vs Hashable",
    [Iterable], [Hashable],
)

# Predicates can only be evaluated on instances, which the analyzer avoids.
registry(
    "parametric predicates (never called during analysis)",
    [parametric(int, lambda value: value > 0)],
    [parametric(int, lambda value: value < 0)],
)


class Left:
    def __new__(cls):  # proves the analyzer never instantiates a user class
        raise RuntimeError("user constructor must not run during analysis")


class Right:
    pass


registry(
    "future multiple-inheritance subclass: Left vs Right",
    [Left], [Right],
)
