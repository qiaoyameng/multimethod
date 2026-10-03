import types

import pytest

from multimethod import DispatchError, multimeta, multimethod, signature


def signatures(method):
    return {key for key in method if isinstance(key, signature)}


# two-level inheritance with explicit multimethods (no metaclass)
class base:
    @multimethod
    def method(self, x: int):
        return "base int"

    @multimethod
    def method(self, x: object):
        return "base object"


class sub(base):
    @multimethod
    def method(self, x: str):
        return "sub str"


def test_two_levels():
    obj = sub()
    assert obj.method("") == "sub str"  # own overload
    assert obj.method(0) == "base int"  # inherited overload
    assert obj.method(0.0) == "base object"
    assert base().method(0) == "base int"
    assert base().method("") == "base object"  # subclass overload not visible in base


def test_local_registry():
    assert sub.method is not base.method
    assert signatures(sub.method) > signatures(base.method)
    # registering on the subclass does not mutate the base registry
    sub.method.register(object, float)(lambda self, x: "sub float")
    assert sub().method(0.0) == "sub float"
    assert base().method(0.0) == "base object"
    base.method.clean()
    sub.method.clean()


def test_override_same_signature():
    class override(base):
        @multimethod
        def method(self, x: int):
            return "override int"

    assert override().method(0) == "override int"
    assert override().method("") == "base object"
    assert base().method(0) == "base int"  # base keeps its own implementation


# siblings
def test_siblings():
    class left(base):
        @multimethod
        def method(self, x: bytes):
            return "left bytes"

    class right(base):
        @multimethod
        def method(self, x: complex):
            return "right complex"

    assert left().method(b"") == "left bytes"
    assert right().method(0j) == "right complex"
    assert left().method(0j) == "base object"  # sibling overloads not visible
    assert right().method(b"") == "base object"
    assert signatures(base.method) == {signature([object, int]), signature([object, object])}


# diamond inheritance
class diamond_base(metaclass=multimeta):
    def method(self, x: object):
        return "base object"

    def method(self, x: int):
        return "base int"


class left(diamond_base):
    def method(self, x: str):
        return "left str"


class right(diamond_base):
    def method(self, x: str):
        return "right str"


class diamond(left, right):
    def method(self, x: bytes):
        return "diamond bytes"


def test_diamond():
    obj = diamond()
    assert obj.method(b"") == "diamond bytes"
    assert obj.method("") == "left str"  # public rule: first in MRO order wins
    assert obj.method(0) == "base int"
    assert obj.method(0.0) == "base object"
    # the shared ancestor's registry is collected only once
    assert signatures(diamond.method) == {
        signature([object, bytes]),
        signature([object, str]),
        signature([object, int]),
        signature([object, object]),
    }
    assert diamond().method("") == "left str"  # cached resolution is stable


def test_super():
    obj = diamond()
    assert super(diamond, obj).method("") == "left str"
    assert super(left, obj).method("") == "right str"
    assert super(right, obj).method("") == "base object"


def test_access_patterns():
    assert isinstance(diamond.method, multimethod)  # class access
    assert isinstance(diamond().method, types.MethodType)  # instance access
    assert diamond().method("") == diamond.method(diamond(), "")


# cache isolation: subclass calls must not pollute the parent's cache or registry
def test_cache_isolation():
    class parent(metaclass=multimeta):
        def method(self, x: object):
            return "parent object"

    class child(parent):
        def method(self, x: str):
            return "child str"

    kid = child()
    assert kid.method("") == "child str"  # fills child's cache
    assert kid.method(0) == "parent object"
    child_types = {key for key in child.method if not isinstance(key, signature)}
    assert child_types  # child cache is populated
    parent_types = {key for key in parent.method if not isinstance(key, signature)}
    assert not parent_types  # parent cache untouched
    assert parent().method(0) == "parent object"
    assert parent().method("") == "parent object"  # not "child str"
    assert signature([object, str]) not in signatures(parent.method)


# snapshot semantics: post-creation base registrations are not visible in subclasses
def test_snapshot():
    class parent(metaclass=multimeta):
        def method(self, x: object):
            return "parent object"

    class child(parent):
        def method(self, x: int):
            return "child int"

    assert child().method(0) == "child int"  # fill child's cache before parent changes

    @parent.method.register
    def _(self, x: frozenset):
        return "parent frozenset"

    assert parent().method(frozenset()) == "parent frozenset"
    assert child().method(frozenset()) == "parent object"  # falls back, snapshot unchanged
    assert parent().method(0) == "parent object"  # parent unaffected by child's cache


# a parent cache populated before the subclass is created is not copied
def test_snapshot_with_populated_cache():
    class parent(metaclass=multimeta):
        def method(self, x: object):
            return "parent object"

    assert parent().method(0) == "parent object"  # fill parent's cache

    class child(parent):
        def method(self, x: str):
            return "child str"

    assert all(isinstance(key, signature) for key in child.method)  # no cache copied
    assert child().method("") == "child str"
    assert child().method(0) == "parent object"
    assert parent().method(0) == "parent object"


# explicitly aliasing the base multimethod keeps it shared, without self-merging
def test_explicit_alias():
    class parent:
        @multimethod
        def method(self, x: object):
            return "parent object"

    class child(parent):
        method = parent.method

    assert child.method is parent.method
    assert child().method(0) == parent().method(0) == "parent object"


# pending forward references are inherited (module-level so the hint resolves)
class fwd_base(metaclass=multimeta):
    def method(self, other: "fwd_base"):
        return "base"


class fwd_child(fwd_base):  # created while `fwd_base.method` is still pending
    def method(self, other: int):
        return "child int"


def test_pending_forward_reference():
    assert fwd_child().method(fwd_base()) == "base"
    assert fwd_child().method(0) == "child int"
    assert fwd_base().method(fwd_base()) == "base"
    with pytest.raises(DispatchError):
        fwd_base().method(0)


# multimeta subclassing multimeta across three levels
def test_three_levels():
    class grand(metaclass=multimeta):
        def method(self, x: object):
            return "grand"

    class parent(grand):
        def method(self, x: int):
            return "parent"

    class child(parent):
        def method(self, x: str):
            return "child"

    obj = child()
    assert obj.method("") == "child"
    assert obj.method(0) == "parent"
    assert obj.method(0.0) == "grand"
    assert parent().method(0) == "parent"
    assert parent().method("") == "grand"  # falls back to inherited overload
    assert grand().method(0) == "grand"
