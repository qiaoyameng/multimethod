import pytest

from multimethod import DispatchError, NoNextMethod, combined, multimethod


def test_combination_order():
    events = []

    @combined
    def process(x):
        events.append("primary-object")
        return "object"

    @process.primary
    def _(x: int, *, next_method):
        events.append("primary-int")
        return "int:" + next_method()

    @process.before
    def _(x: int):
        events.append("before-int")

    @process.before
    def _(x: object):
        events.append("before-object")

    @process.after
    def _(x: int):
        events.append("after-int")

    @process.after
    def _(x: object):
        events.append("after-object")

    @process.around
    def _(x: int, *, next_method):
        events.append("around-int-enter")
        result = next_method()
        events.append("around-int-exit")
        return "wrapped:" + result

    assert process(1) == "wrapped:int:object"
    assert events == [
        "around-int-enter",
        "before-int",
        "before-object",
        "primary-int",
        "primary-object",
        "after-object",
        "after-int",
        "around-int-exit",
    ]
    assert process("") == "object"


def test_primary_register_aliases():
    @combined
    def func(x):
        return "object"

    @func.register
    def _(x: int):
        return "int"

    assert func(1) == "int"
    assert func(object()) == "object"


def test_around_short_circuit():
    events = []

    @combined
    def func(x):
        events.append("primary")
        return "primary"

    @func.before
    def _(x: int):
        events.append("before")

    @func.after
    def _(x: int):
        events.append("after")

    @func.around
    def _(x: int):
        events.append("around")
        return "short"

    assert func(1) == "short"
    assert events == ["around"]


def test_around_chain_specificity():
    events = []

    @combined
    def func(x):
        events.append("primary")
        return "base"

    @func.around
    def _(x: object, *, next_method):
        events.append("around-object-enter")
        result = next_method()
        events.append("around-object-exit")
        return "O:" + result

    @func.around
    def _(x: int, *, next_method):
        events.append("around-int-enter")
        result = next_method()
        events.append("around-int-exit")
        return "I:" + result

    assert func(1) == "I:O:base"
    assert events == [
        "around-int-enter",
        "around-object-enter",
        "primary",
        "around-object-exit",
        "around-int-exit",
    ]


def test_next_method_chain():
    @combined
    def func(x):
        return "object"

    @func.primary
    def _(x: bool, *, next_method):
        return "bool:" + next_method()

    @func.primary
    def _(x: int, *, next_method):
        return "int:" + next_method()

    assert func(True) == "bool:int:object"
    assert func(1) == "int:object"
    assert func(1.0) == "object"


def test_next_method_repeatable():
    calls = []

    @combined
    def func(x):
        calls.append("object")
        return "object"

    @func.primary
    def _(x: int, *, next_method):
        calls.append("int")
        return (next_method(), next_method())

    assert func(1) == ("object", "object")
    assert calls == ["int", "object", "object"]


def test_next_method_no_successor():
    @combined
    def func(x, *, next_method):
        return next_method()

    with pytest.raises(NoNextMethod):
        func(1)

    assert issubclass(NoNextMethod, DispatchError)


def test_next_method_rejects_arguments():
    @combined
    def func(x):
        return "object"

    @func.primary
    def _(x: int, *, next_method):
        return next_method(1)

    with pytest.raises(TypeError, match="takes no arguments"):
        func(1)

    @func.primary
    def _(x: str, *, next_method):
        return next_method(x="")

    with pytest.raises(TypeError, match="takes no arguments"):
        func("")


def test_next_method_uses_snapshot():
    seen = []

    @combined
    def func(x):
        seen.append(type(x))
        return "object"

    @func.primary
    def _(x: int, *, next_method):
        seen.append("int")
        return next_method()

    assert func(1) == "object"
    assert seen == ["int", int]


def test_ambiguous_diamond():
    class A: ...

    class B: ...

    class AB(A, B): ...

    @combined
    def func(x):
        return "object"

    @func.primary
    def _(x: A):
        return "A"

    @func.primary
    def _(x: B):
        return "B"

    with pytest.raises(DispatchError):
        func(AB())


def test_ambiguity_independent_of_registration_order():
    class A: ...

    class B: ...

    class AB(A, B): ...

    @combined
    def first(x):
        return "object"

    @first.primary
    def _(x: A):
        return "A"

    @first.primary
    def _(x: B):
        return "B"

    @combined
    def second(x):
        return "object"

    @second.primary
    def _(x: B):
        return "B"

    @second.primary
    def _(x: A):
        return "A"

    with pytest.raises(DispatchError):
        first(AB())
    with pytest.raises(DispatchError):
        second(AB())


def test_ambiguous_hooks():
    class A: ...

    class B: ...

    class AB(A, B): ...

    @combined
    def before_func(x):
        return "object"

    @before_func.before
    def _(x: A):
        pass

    @before_func.before
    def _(x: B):
        pass

    @combined
    def after_func(x):
        return "object"

    @after_func.after
    def _(x: A):
        pass

    @after_func.after
    def _(x: B):
        pass

    @combined
    def around_func(x):
        return "object"

    @around_func.around
    def _(x: A):
        pass

    @around_func.around
    def _(x: B):
        pass

    for func in (before_func, after_func, around_func):
        with pytest.raises(DispatchError):
            func(AB())


def test_after_skipped_on_primary_exception():
    events = []

    @combined
    def func(x):
        events.append("primary")
        raise ValueError("boom")

    @func.after
    def _(x: object):
        events.append("after")

    with pytest.raises(ValueError, match="boom"):
        func(1)
    assert events == ["primary"]


def test_before_exception_unwinds_around():
    events = []

    @combined
    def func(x):
        events.append("primary")
        return "primary"

    @func.before
    def _(x: int):
        raise RuntimeError("before boom")

    @func.after
    def _(x: int):
        events.append("after")

    @func.around
    def _(x: int, *, next_method):
        events.append("around-enter")
        try:
            return next_method()
        except RuntimeError:
            events.append("around-caught")
            return "recovered"

    assert func(1) == "recovered"
    assert events == ["around-enter", "around-caught"]


def test_internal_typeerror_not_wrapped():
    @combined
    def func(x):
        return 1 + ""

    with pytest.raises(TypeError, match="unsupported operand"):
        func(1)


def test_reentrant_chains_isolated():
    events = []

    @combined
    def func(x, *, depth=0):
        events.append(("enter", depth))
        if depth == 0:
            func(x, depth=1)
        events.append(("leave", depth))
        return depth

    @func.before
    def _(x, *, depth=0):
        events.append(("before", depth))

    @func.after
    def _(x, *, depth=0):
        events.append(("after", depth))

    assert func(1) == 0
    assert events == [
        ("before", 0),
        ("enter", 0),
        ("before", 1),
        ("enter", 1),
        ("leave", 1),
        ("after", 1),
        ("leave", 0),
        ("after", 0),
    ]


def test_failed_call_does_not_corrupt_later_chains():
    @combined
    def func(x):
        if x < 0:
            raise RuntimeError("negative")
        return "ok"

    @func.after
    def _(x: object):
        pass

    assert func(1) == "ok"
    with pytest.raises(RuntimeError):
        func(-1)
    assert func(2) == "ok"


def test_no_primary_dispatch_error():
    @combined
    def func(x: str):
        return "str"

    with pytest.raises(DispatchError):
        func(1)


def test_generic_specificity_ordering():
    events = []

    @combined
    def func(xs):
        events.append("primary")
        return sum(xs)

    @func.before
    def _(xs: list[int]):
        events.append("before-list-int")

    @func.before
    def _(xs: list):
        events.append("before-list")

    @func.before
    def _(xs: object):
        events.append("before-object")

    @func.after
    def _(xs: list[int]):
        events.append("after-list-int")

    @func.after
    def _(xs: list):
        events.append("after-list")

    assert func([1, 2, 3]) == 6
    assert events == [
        "before-list-int",
        "before-list",
        "before-object",
        "primary",
        "after-list",
        "after-list-int",
    ]


def test_explicit_types_registration():
    @combined
    def func(x):
        return "object"

    @func.before(int)
    def before_int(x):
        pass

    @func.after(object)
    def after_object(x):
        pass

    assert func(1) == "object"


def test_descriptor_binding():
    class Service:
        @combined
        def handle(self, x):
            return ("object", x)

        @handle.primary
        def _(self, x: int, *, next_method):
            return ("int", next_method())

    service = Service()
    assert service.handle(1) == ("int", ("object", 1))
    assert service.handle("") == ("object", "")


def test_plain_multimethod_unchanged():
    @multimethod
    def func(x):
        return "object"

    @func.register
    def _(x: int):
        return "int"

    assert not hasattr(func, "before")
    assert func(1) == "int"
    assert func("") == "object"


def test_equally_specific_same_function_collapses():
    @combined
    def func(x):
        return "object"

    @func.primary
    def _(x: int | str):
        return "union"

    assert func(1) == "union"
    assert func("") == "union"


def test_copy_is_independent():
    @combined
    def func(x):
        return "object"

    @func.primary
    def _(x: int):
        return "int"

    clone = func.copy()

    @clone.primary
    def _(x: str):
        return "str"

    assert clone("") == "str"
    assert func("") == "object"
    assert func(1) == "int"


def test_kwargs_passed_through():
    @combined
    def func(x, *, flag=False):
        return flag

    @func.before
    def _(x: int, *, flag=False):
        pass

    assert func(1, flag=True) is True


def test_around_calls_core_multiple_times():
    primary_calls = []

    @combined
    def func(x):
        primary_calls.append(x)
        return x

    @func.around
    def _(x: int, *, next_method):
        first = next_method()
        second = next_method()
        return first + second

    assert func(3) == 6
    assert primary_calls == [3, 3]
