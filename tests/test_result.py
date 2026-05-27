from hypothesis import assume, given
from hypothesis import strategies as st

from src import Err, Ok, Result


@st.composite
def results(draw):
    val = draw(st.integers() | st.text() | st.lists(st.integers()))
    if draw(st.booleans()):
        return Ok(val)
    return Err(val)


ok_values = st.integers() | st.text() | st.lists(st.integers())
err_values = st.text()


class TestResultLaws:
    @given(ok_values)
    def test_ok_unwrap(self, v):
        assert Ok(v).unwrap() == v

    @given(err_values)
    def test_err_unwrap_raises(self, e):
        try:
            Err(e).unwrap()
            assert False, "Expected RuntimeError"
        except RuntimeError:
            pass

    @given(ok_values)
    def test_ok_unwrap_or(self, v):
        assert Ok(v).unwrap_or(42) == v

    @given(err_values)
    def test_err_unwrap_or_default(self, e):
        assert Err(e).unwrap_or(42) == 42

    @given(ok_values)
    def test_ok_is_ok(self, v):
        assert Ok(v).is_ok()
        assert not Ok(v).is_err()

    @given(err_values)
    def test_err_is_err(self, e):
        assert Err(e).is_err()
        assert not Err(e).is_ok()

    @given(ok_values)
    def test_ok_ok_method(self, v):
        assert Ok(v).ok() == v

    @given(err_values)
    def test_err_ok_method(self, e):
        assert Err(e).ok() is None

    @given(ok_values)
    def test_ok_err_method(self, v):
        assert Ok(v).err() is None

    @given(err_values)
    def test_err_err_method(self, e):
        assert Err(e).err() == e

    @given(ok_values)
    def test_map_identity(self, v):
        assert Ok(v).map(lambda a: a) == Ok(v)

    @given(st.integers())
    def test_map_double(self, v):
        assert Ok(v).map(lambda x: x * 2) == Ok(v * 2)

    @given(err_values)
    def test_map_skips_err(self, e):
        r = Err(e).map(lambda x: x * 2)
        assert r.is_err()
        assert r.err() == e

    @given(st.integers())
    def test_and_then_chain(self, v):
        r = Ok(v).and_then(lambda x: Ok(x + 1)).and_then(lambda x: Ok(x * 2))
        assert r == Ok((v + 1) * 2)

    @given(st.integers())
    def test_and_then_short_circuit(self, v):
        r = Ok(v).and_then(lambda x: Err("fail")).and_then(lambda x: Ok(999))
        assert r.is_err()
        assert r.err() == "fail"

    @given(st.integers())
    def test_or_else_recovers(self, v):
        r = Err("fail").or_else(lambda e: Ok(v))
        assert r == Ok(v)

    @given(st.integers())
    def test_or_else_skips_ok(self, v):
        r = Ok(v).or_else(lambda e: Ok(999))
        assert r == Ok(v)

    @given(ok_values)
    def test_bool_ok(self, v):
        assert bool(Ok(v)) is True

    @given(err_values)
    def test_bool_err(self, e):
        assert bool(Err(e)) is False

    @given(ok_values, err_values)
    def test_map_err_skips_ok(self, v, e):
        assert Ok(v).map_err(lambda _: e) == Ok(v)

    @given(err_values, st.integers())
    def test_map_err_transforms(self, e, x):
        assert Err(e).map_err(lambda _: x).err() == x

    @given(ok_values)
    def test_expect_ok(self, v):
        assert Ok(v).expect("msg") == v
