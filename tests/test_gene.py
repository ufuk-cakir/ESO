import random
import pytest

from eso.ga.gene import Gene


class DummyRNG:
    """Helper to make random outputs predictable in tests."""

    def __init__(self, heights, positions):
        self._heights = heights[:]  # pop sequence
        self._positions = positions[:]

    def randint(self, a, b):  # type: ignore
        # decide whether to return from heights or positions stream
        if self._heights:
            return self._heights.pop(0)
        return self._positions.pop(0)


def test_max_height_minus_one_raises():
    with pytest.raises(ValueError):
        Gene(
            spec_height=10, min_position=0, max_position=9, min_height=1, max_height=-1
        )


def test_negative_min_and_max_position_and_height_normalize():
    g = Gene(
        spec_height=5, min_position=-1, max_position=-1, min_height=-1, max_height=3
    )
    # min_position → 0, max_position → spec_height
    assert 0 <= g.band_position <= 5 - g.band_height
    assert 1 <= g.band_height <= 3


def test_random_gene_uses_random_ints(monkeypatch):
    # force randint to first pick height=2 then position=1
    drng = DummyRNG(heights=[2], positions=[1])
    monkeypatch.setattr(random, "randint", drng.randint)

    g = Gene(spec_height=10, min_position=0, max_position=9, min_height=1, max_height=5)
    assert g.band_height == 2
    assert g.band_position == 1


def test_fixed_height_random_position(monkeypatch):
    # height fixed at 4; next randint returns pos=2
    drng = DummyRNG(heights=[], positions=[2])
    monkeypatch.setattr(random, "randint", drng.randint)

    g = Gene(
        spec_height=10,
        min_position=1,
        max_position=8,
        min_height=1,
        max_height=5,
        band_position=None,
        band_height=4,
    )
    assert g.band_height == 4
    assert 1 <= g.band_position <= 8 - 4
    assert g.band_position == 2


def test_fixed_position_random_height(monkeypatch):
    # position fixed at 3; next randint returns height=2
    drng = DummyRNG(heights=[2], positions=[])
    monkeypatch.setattr(random, "randint", drng.randint)

    g = Gene(
        spec_height=10,
        min_position=0,
        max_position=9,
        min_height=1,
        max_height=5,
        band_position=3,
        band_height=None,
    )
    assert g.band_position == 3
    # upper_height = min(max_position - band_position, max_height) = min(9-3,5)=5
    assert 1 <= g.band_height <= 5
    assert g.band_height == 2


def test_set_gene_valid_and_invalid():
    # valid
    g = Gene(10, 0, 9, 1, 5, band_position=2, band_height=3)
    assert (g.band_position, g.band_height) == (2, 3)

    # invalid: sum exceeds max_position
    with pytest.raises(ValueError):
        Gene(10, 0, 5, 1, 5, band_position=3, band_height=3)


def test_getters_and_setters():
    g = Gene(10, 0, 9, 1, 5, band_position=1, band_height=2)
    assert g.get_band_position() == 1
    assert g.get_band_height() == 2

    # setting position out of range
    with pytest.raises(AssertionError):
        g.set_band_position(-1)
    with pytest.raises(AssertionError):
        g.set_band_position(9)  # 9 + height(2) = 11 > max_position(9)

    # setting height out of range
    with pytest.raises(AssertionError):
        g.set_band_height(0)
    with pytest.raises(AssertionError):
        g.set_band_height(10)  # > max_height


def test_str_and_repr():
    g = Gene(5, 0, 4, 1, 3, band_position=1, band_height=2)
    assert str(g) == "(1, 2)"
    assert repr(g) == "Gene(1, 2)"


def test_gene():
    """
    test function.
    Returns:
        None
    """

    # Fixed gene (band_position and band_height given)
    gene_2 = Gene(
        spec_height=200,
        min_position=0,
        max_position=128,
        min_height=1,
        max_height=10,
        band_position=80,
        band_height=7,
    )
    print(f"Gene 2: {gene_2}")

    # Partially random gene (given band_position)
    gene_3 = Gene(
        spec_height=200,
        min_position=0,
        max_position=128,
        min_height=1,
        max_height=10,
        band_position=100,
    )
    print(f"Gene 3: {gene_3}")

    # Partially random gene (given band_height)
    gene_4 = Gene(
        spec_height=200,
        min_position=0,
        max_position=128,
        min_height=1,
        max_height=10,
        band_height=9,
    )
    print(f"Gene 4: {gene_4}")

    # Should error die silently ?
    # Should we simply fail ?
    # Should we alert the user
    # Right now we are generating a random/partially random gene when some
    # are not valid
    # Could also raise ValueErrors.

    print("\n======================================")
    # Problematic cases
    print("Problematic cases")

    # band_position < 0
    # gene_7 = Gene(spec_height=200, min_position=0, max_position=128,
    #              min_height=1, max_height=10, band_position=-123)
    # print(f"Gene 7: {gene_7}")

    # band_position > max_position
    # gene_8 = Gene(spec_height=200, min_position=0, max_position=128,
    #              min_height=1, max_height=10, band_position=200)

    # print(f"Gene 8: {gene_8}")

    # band_height < 0
    # gene_9 = Gene(spec_height=200, min_position=0, max_position=128,
    #              min_height=1, max_height=10, band_position=90, band_height=-10)
    # print(f"Gene 9: {gene_9}")

    # gene_10 = Gene(spec_height=200, min_position=0, max_position=128,
    #               min_height=1, max_height=10, band_height=-4)
    # print(f"Gene 10: {gene_10}")

    # band_position < 0 and band_height < 0
    # gene_11 = Gene(spec_height=200, min_position=0, max_position=128,
    #               min_height=1, max_height=10, band_position=-120, band_height=-8)
    # print(f"Gene 11: {gene_11}")

    # band_position < 0 and band_height > 0
    # gene_12 = Gene(spec_height=200, min_position=0,
    #               max_position=128, min_height=1,
    #               max_height=10, band_position=-100, band_height=5)

    # band_height > max_height
    # gene_13 = Gene(spec_height=200, min_position=0, max_position=128,
    #               min_height=1, max_height=10, band_height= 11)
