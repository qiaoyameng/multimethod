import pytest

from multimethod import DispatchError, combined, multimethod, next_method


def test_combination_order():
    calls = []

    @combined
    def func(x: object):
        calls.append("primary-object")
        return "object"

    @func.primary
    def _(x: int):
        calls.append("primary-int")
        return "int"

    @func.before
    def _(x: object):
        calls.append("before-object")

    @func.before
    def _(x: int):
        calls.append("before-int")

    @func.after
    def _(x: object):
        calls.append("after-object")

    @func.after
    def _(x: int):
        calls.append("after-int")

    assert func(0) == "int"
    assert calls == [
        "before-int",  # befores run most specific first
        "before-object",
        "primary-int",  # primary-int does not call next_method
        "after-object",  # afters run least specific first
        "after-int",
    ]


def test_next_method():
    @combined
    def func(x: object):
        return "object"

    @func.primary
    def _(x: int):
        return next_method() + "+int"

    @func.primary
    def _(x: bool):
        return next_method() + "+bool"

    # next_method walks the candidate snapshot; re-dispatching would recurse
    assert func(True) == "object+int+bool"
    assert func(0) == "object+int"
    assert func("") == "object"


def test_no_next_method():
    @combined
    def func(x: object):
        return next_method()

    with pytest.raises(DispatchError, match="no next method"):
        func(0)

    with pytest.raises(DispatchError, match="no next method"):
        next_method()  # outside of any call


def test_next_method_twice():
    calls = []

    @combined
    def func(x: object):
        calls.append("object")
        return len(calls)

    @func.primary
    def _(x: int):
        return next_method(), next_method()

    assert func(0) == (1, 2)  # each call re-invokes the same next method
    assert calls == ["object", "object"]


def test_next_method_arguments():
    @combined
    def func(x: object):
        return "object"

    @func.primary
    def _(x: int):
        return next_method(x)  # attempting to change arguments is an error

    with pytest.raises(TypeError):
        func(0)


def test_around():
    calls = []

    @combined
    def func(x: object):
        calls.append("primary")
        return "primary"

    @func.around
    def _(x: object):
        calls.append("around-object")
        return next_method() + "!"

    @func.around
    def _(x: int):
        calls.append("around-int")
        return "[" + next_method() + "]"

    assert func(0) == "[primary!]"  # arounds run most specific first
    assert calls == ["around-int", "around-object", "primary"]


def test_around_short_circuit():
    calls = []

    @combined
    def func(x: object):
        calls.append("primary")
        return "primary"

    @func.before
    def _(x: object):
        calls.append("before")

    @func.after
    def _(x: object):
        calls.append("after")

    @func.around
    def _(x: int):
        return "short"

    assert func(0) == "short"  # befores, primary, and afters are all skipped
    assert calls == []
    assert func("") == "primary"
    assert calls == ["before", "primary", "after"]


def test_around_short_circuit_without_primary():
    @combined
    def func(x: str):
        return "str"

    @func.around
    def _(x: object):
        return next_method()

    assert func("") == "str"
    with pytest.raises(DispatchError, match="0 methods"):
        func(0)  # the around delegates, but no primary is applicable

    @func.around
    def _(x: int):
        return "short"

    assert func(0) == "short"  # short-circuiting needs no primary


def test_after_on_success_only():
    calls = []

    @combined
    def func(x: object):
        return "ok"

    @func.primary
    def _(x: int):
        raise ValueError("boom")

    @func.after
    def _(x: object):
        calls.append("after")

    assert func("") == "ok"
    assert calls == ["after"]
    with pytest.raises(ValueError):
        func(0)
    assert calls == ["after"]  # afters are skipped when the primary raises


def test_before_raises():
    calls = []

    @combined
    def func(x: object):
        calls.append("primary")
        return "ok"

    @func.before
    def _(x: int):
        raise ValueError("boom")

    @func.after
    def _(x: object):
        calls.append("after")

    assert func("") == "ok"
    with pytest.raises(ValueError):
        func(0)
    # neither primary nor afters ran for the int call
    assert calls == ["primary", "after"]


def test_no_applicable_primary():
    @combined
    def func(x: int):
        return "int"

    with pytest.raises(DispatchError, match="0 methods"):
        func("")


def test_ambiguous_primaries():
    @combined
    def func(x: object, y: object):
        return "object"

    @func.primary
    def _(x: int, y: object):
        return "int"

    @func.primary
    def _(x: object, y: str):
        return "str"

    assert func(0, 0) == "int"
    assert func("", "") == "str"
    assert func("", 0) == "object"
    with pytest.raises(DispatchError, match="2 methods"):
        func(0, "")  # incomparable candidates, regardless of registration order


def test_ambiguous_befores():
    @combined
    def func(x: object, y: object):
        return "ok"

    @func.before
    def _(x: int, y: object): ...

    @func.before
    def _(x: object, y: str): ...

    assert func(0, 0) == "ok"
    with pytest.raises(DispatchError, match="2 methods"):
        func(0, "")


def test_internal_type_error():
    @combined
    def func(x: object):
        return "ok"

    @func.primary
    def _(x: int):
        raise TypeError("internal")

    with pytest.raises(TypeError, match="internal"):  # not wrapped in DispatchError
        func(0)
    assert func("") == "ok"  # the chain is left intact


def test_reentrant_call():
    @combined
    def func(x: object):
        return "object"

    @func.primary
    def _(x: str):
        return "str"

    @func.primary
    def _(x: int):
        inner = func("")  # reentrant call with its own candidate snapshot
        return inner + "+" + next_method()

    assert func(0) == "str+object"


def test_reentrant_exception():
    @combined
    def inner(x: object):
        return "inner"

    @inner.primary
    def _(x: int):
        raise ValueError("boom")

    @combined
    def outer(x: object):
        return "outer"

    @outer.primary
    def _(x: int):
        with pytest.raises(ValueError):
            inner(0)  # unwinding the inner chain does not disturb this one
        return next_method() + "+int"

    assert outer(0) == "outer+int"


def test_no_next_method_in_before():
    @combined
    def func(x: object):
        return "ok"

    @func.before
    def _(x: int):
        next_method()

    with pytest.raises(DispatchError, match="no next method"):
        func(0)


def test_late_registration():
    calls = []

    @combined
    def func(x: object):
        return "object"

    assert func(0) == "object"

    @func.primary
    def _(x: int):
        return "int"

    @func.before
    def _(x: object):
        calls.append("before")

    assert func(0) == "int"  # cached chains are invalidated by registration
    assert calls == ["before"]


def test_explicit_types():
    calls = []

    @combined
    def func(x):
        return "object"

    @func.before(int)
    def _(x):
        calls.append("before-int")

    assert func(0) == "object"
    assert calls == ["before-int"]


def test_stacked_registration():
    calls = []

    @combined
    def func(x: int):
        return "int"

    @func.before
    def _(x: object):
        calls.append("before")

    @combined
    def func(x: str):
        return "str"

    assert func(0) == "int"
    assert func("") == "str"
    # qualifiers survive stacked registration, and apply to both calls
    assert calls == ["before", "before"]


def test_keywords():
    @combined
    def func(x: object, scale=1):
        return x * scale

    assert func(2, scale=3) == 6


def test_generic():
    @combined
    def func(x: list):
        return "list"

    @func.primary
    def _(x: list[int]):
        return next_method() + "[int]"

    assert func([0]) == "list[int]"
    assert func([""]) == "list"


def test_copy():
    calls = []

    @combined
    def func(x: object):
        return "object"

    @func.before
    def _(x: object):
        calls.append("before")

    clone = func.copy()
    assert isinstance(clone, combined)
    assert clone(0) == "object"
    assert calls == ["before"]


def test_multimethod_unchanged():
    @multimethod
    def func(x: object):
        return "object"

    @func.register
    def _(x: int):
        return "int"

    assert func(0) == "int"
    assert not hasattr(func, "befores")


def test_methods():
    class cls:
        @combined
        def method(self, x: object):
            return "object"

        @method.primary
        def _(self, x: int):
            return next_method() + "+int"

        @method.after
        def _(self, x: object):
            calls.append("after")

    calls = []
    obj = cls()
    assert obj.method(0) == "object+int"
    assert cls.method(obj, "") == "object"
    assert calls == ["after", "after"]


forward_calls = []


@combined
def forward_func(x: "forward_cls"):
    return "primary"


@forward_func.before
def _(x: "forward_cls"):
    forward_calls.append("before")


class forward_cls: ...


def test_forward_references():
    assert forward_func(forward_cls()) == "primary"
    assert forward_calls == ["before"]
