import pytest

from multimethod import DispatchError, multidispatch, multimeta, multimethod, signature


def signatures(method):
    return {key for key in method if isinstance(key, signature)}


class base:
    @multimethod
    def func(self, x: int):
        return "base int"

    @multimethod
    def func(self, x: object):
        return "base object"


class sub(base):
    @multimethod
    def func(self, x: str):
        return "sub str"


class sibling(base):
    @multimethod
    def func(self, x: bytes):
        return "sibling bytes"


class grandchild(sub):
    @multimethod
    def func(self, x: list):
        return "grandchild list"


def test_two_level_inheritance():
    assert sub().func(1) == "base int"
    assert sub().func("") == "sub str"
    assert sub().func(0.0) == "base object"
    assert grandchild().func(1) == "base int"
    assert grandchild().func("") == "sub str"
    assert grandchild().func([]) == "grandchild list"
    assert grandchild().func(0.0) == "base object"


def test_base_registry_untouched():
    assert base().func(1) == "base int"
    assert base().func("") == "base object"  # not "sub str"
    assert base().func(b"") == "base object"  # not "sibling bytes"
    assert len(signatures(base.func)) == 2


def test_siblings_are_independent():
    assert sibling().func(b"") == "sibling bytes"
    assert sibling().func("") == "base object"  # sub's str overload not visible
    assert sub().func(b"") == "base object"  # sibling's bytes overload not visible
    assert sibling.func is not sub.func
    assert sibling.func is not base.func


def test_same_signature_override():
    class override(base):
        @multimethod
        def func(self, x: int):
            return "override int"

    assert override().func(1) == "override int"
    assert override().func(0.0) == "base object"
    assert base().func(1) == "base int"


def test_subclass_register_is_local():
    class local(base): ...

    @local.func.register
    def _(self, x: complex):
        return "local complex"

    assert local().func(0j) == "local complex"
    assert base().func(0j) == "base object"
    assert sub().func(0j) == "base object"


def test_lazy_materialization():
    class lazy(base): ...

    assert "func" not in lazy.__dict__
    assert lazy().func(1) == "base int"
    assert "func" in lazy.__dict__
    assert lazy.func is lazy.__dict__["func"]
    assert lazy.func is not base.func
    assert base.func is base.__dict__["func"]


def test_cache_isolation():
    # child calls fill the child's cache; the parent's registry and cache are unaffected
    assert sub().func(1) == "base int"
    assert sub().func("") == "sub str"
    assert any(not isinstance(key, signature) for key in sub.func)  # cache populated
    assert base().func("") == "base object"
    assert base().func(1) == "base int"
    # parent calls fill the parent's cache; the child is unaffected
    assert base().func(0.0) == "base object"
    assert sub().func(0.0) == "base object"
    assert sub().func("") == "sub str"


def test_post_inheritance_registration_is_snapshot():
    class snap_base:
        @multimethod
        def func(self, x: int):
            return "base int"

        @multimethod
        def func(self, x: object):
            return "base object"

    class snap(snap_base): ...

    assert snap().func(1) == "base int"  # materialize the subclass registry

    @snap_base.func.register
    def _(self, x: complex):
        return "late complex"

    assert snap_base().func(0j) == "late complex"  # visible on the parent
    assert snap().func(0j) == "base object"  # snapshot: not visible on the child


def test_super():
    class sup(base):
        @multimethod
        def func(self, x: object):
            return "sub object"

        @multimethod
        def func(self, x: int):
            return "sub int", super().func(x)

    assert sup().func(1) == ("sub int", "base int")
    assert sup().func(0.0) == "sub object"
    assert base().func(0.0) == "base object"


def test_access_patterns():
    assert isinstance(sub.func, multimethod)  # class access
    assert sub.func(sub(), 1) == "base int"  # explicit instance passing
    bound = sub().func  # method retrieval
    assert bound("") == "sub str"
    assert bound.__self__ is not None
    assert sub.func is sub.func  # stable identity after materialization


# multiple inheritance


class left(base):
    @multimethod
    def func(self, x: str):
        return "left str"


class right(base):
    @multimethod
    def func(self, x: bytes):
        return "right bytes"


class diamond(left, right): ...


def test_diamond_merges_mro():
    class left_local(base):
        @multimethod
        def func(self, x: str):
            return "left str"

    class right_local(base):
        @multimethod
        def func(self, x: bytes):
            return "right bytes"

    class local_diamond(left_local, right_local): ...

    assert local_diamond().func(1) == "base int"
    assert local_diamond().func("") == "left str"
    assert local_diamond().func(b"") == "right bytes"
    assert local_diamond().func(0.0) == "base object"
    # the shared ancestor is collected once: 2 base + str + bytes signatures
    assert len(signatures(local_diamond.func)) == 4


def test_diamond_module_level():
    assert diamond().func(1) == "base int"
    assert diamond().func("") == "left str"
    assert diamond().func(b"") == "right bytes"
    assert diamond().func(0.0) == "base object"


def test_diamond_with_own_overloads():
    class bottom(left, right):
        @multimethod
        def func(self, x: float):
            return "bottom float"

    assert bottom().func(1.0) == "bottom float"
    assert bottom().func("") == "left str"
    assert bottom().func(b"") == "right bytes"
    assert bottom().func(1) == "base int"


def test_conflicting_signatures_follow_mro():
    class left_conflict(base):
        @multimethod
        def func(self, x: int):
            return "left"

    class right_conflict(base):
        @multimethod
        def func(self, x: int):
            return "right"

    class mro_left(left_conflict, right_conflict): ...

    class mro_right(right_conflict, left_conflict): ...

    assert mro_left().func(1) == "left"  # nearest class in MRO wins
    assert mro_right().func(1) == "right"
    assert left_conflict().func(1) == "left"
    assert right_conflict().func(1) == "right"


def test_multimeta_inheritance():
    class meta_base(metaclass=multimeta):
        def func(self, x: int):
            return "base int"

    class meta_sub(meta_base):
        def func(self, x: str):
            return "sub str"

    assert meta_sub().func(1) == "base int"
    assert meta_sub().func("") == "sub str"
    with pytest.raises(DispatchError):
        meta_base().func("")


# forward references: defined after the classes, resolved on first call


class fwd_base:
    @multimethod
    def func(self, x: "fwd_ref"):
        return "fwd"


class fwd_sub(fwd_base):
    @multimethod
    def func(self, x: int):
        return "int"


class fwd_ref: ...


def test_pending_forward_references_are_inherited():
    assert fwd_sub().func(fwd_ref()) == "fwd"
    assert fwd_sub().func(1) == "int"
    assert fwd_base().func(fwd_ref()) == "fwd"
    with pytest.raises(DispatchError):
        fwd_base().func(1)


def test_inherit_after_parent_cache_filled():
    class cached_base:
        @multimethod
        def func(self, x: int):
            return "int"

        @multimethod
        def func(self, x: object):
            return "object"

    assert cached_base().func(1) == "int"  # fill the parent's dispatch cache

    class cached_sub(cached_base):
        @multimethod
        def func(self, x: str):
            return "str"

    class cached_lazy(cached_base): ...

    assert cached_sub().func(1) == "int"
    assert cached_sub().func("") == "str"
    assert cached_lazy().func(1) == "int"
    assert cached_lazy().func("") == "object"
    assert cached_base().func("") == "object"


def test_detached_descriptor_access():
    assert base.func.__get__(None, None) is base.func
    assert base.func.__get__(None, int) is base.func  # unrelated owner: no materialization


def test_multidispatch_inheritance():
    class md_base:
        @multidispatch
        def func(self, x: int):
            return "int"

    class md_sub(md_base): ...

    assert md_sub().func(1) == "int"
    assert md_sub.func is not md_base.func

    @md_sub.func.register
    def _(self, x: str):
        return "str"

    assert md_sub().func("") == "str"
    assert md_base().func("") == "int"  # base implementation still matches
