import pytest
from nanopyx.core.io.unique_names import intToRoman, get_unique_name, _already_used, names


def test_int_to_roman():
    # Test some known values
    assert intToRoman(1) == "I"
    assert intToRoman(4) == "IV"
    assert intToRoman(9) == "IX"
    assert intToRoman(10) == "X"
    assert intToRoman(40) == "XL"
    assert intToRoman(50) == "L"
    assert intToRoman(90) == "XC"
    assert intToRoman(100) == "C"
    assert intToRoman(400) == "CD"
    assert intToRoman(500) == "D"
    assert intToRoman(900) == "CM "
    assert intToRoman(1000) == "M"


def test_int_to_roman_range():
    # Test all values in the range 1-10
    for i in range(1, 11):
        assert intToRoman(i) is not None


def test_get_unique_name():
    _already_used.clear()  # Clear the global list before testing
    unique_names = set()
    for _ in range(100):  # Generate 100 unique names
        name = get_unique_name()
        assert name not in unique_names
        unique_names.add(name)
