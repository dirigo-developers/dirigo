"""Test inclusive integer ranges with strictly increasing bounds.

Verify that endpoints are integers, excluding booleans, and that equal or
reversed bounds are rejected. Membership and cardinality include both endpoints,
and bounds remain immutable after construction.
"""

from dataclasses import FrozenInstanceError

import pytest

from dirigo.units import IntRange


@pytest.mark.parametrize(
    ("minimum", "maximum", "expected_count"),
    [
        (0, 1, 2),
        (1, 5, 5),
        (-5, -2, 4),
        (-256, 255, 512),
    ],
)
def test_range_counts_inclusive_integers(
    minimum: int,
    maximum: int,
    expected_count: int,
) -> None:
    """Preserve the supplied bounds and count every integer including both endpoints."""
    bounds = IntRange(minimum, maximum)

    assert bounds.min == minimum
    assert bounds.max == maximum
    assert bounds.count == expected_count


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [
        (1.0, 3),
        (1, 3.0),
        ("1", 3),
        (1, "3"),
        (None, 3),
        (1, None),
    ],
)
def test_noninteger_endpoints_are_rejected(
    minimum: object,
    maximum: object,
) -> None:
    """Reject noninteger endpoints on either side without coercing their values."""
    with pytest.raises(TypeError, match="endpoints must be integers"):
        IntRange(minimum, maximum)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [
        (False, 1),
        (True, 1),
        (0, False),
        (0, True),
        (False, True),
        (True, False),
    ],
)
def test_boolean_endpoints_are_rejected(
    minimum: int,
    maximum: int,
) -> None:
    """Reject boolean endpoints before checking whether bounds are strictly ordered."""
    with pytest.raises(TypeError, match="endpoints must be integers"):
        IntRange(minimum, maximum)


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [
        (0, 0),
        (4, 4),
        (-4, -4),
        (1, 0),
        (0, -1),
        (-2, -3),
    ],
)
def test_endpoints_must_be_strictly_increasing(
    minimum: int,
    maximum: int,
) -> None:
    """Reject singleton ranges and reversed bounds for positive and negative values."""
    with pytest.raises(ValueError, match="max must be greater than min"):
        IntRange(minimum, maximum)


@pytest.mark.parametrize(
    ("minimum", "maximum", "value", "expected"),
    [
        (-2, 2, -3, False),
        (-2, 2, -2, True),
        (-2, 2, 0, True),
        (-2, 2, 2, True),
        (-2, 2, 3, False),
        (4, 5, 3, False),
        (4, 5, 4, True),
        (4, 5, 5, True),
        (4, 5, 6, False),
    ],
)
def test_membership_includes_both_endpoints(
    minimum: int,
    maximum: int,
    value: int,
    expected: bool,
) -> None:
    """Include endpoints and interior values while excluding values outside bounds."""
    bounds = IntRange(minimum, maximum)

    assert (value in bounds) is expected


@pytest.mark.parametrize("endpoint", ["min", "max"])
def test_endpoints_are_immutable(endpoint: str) -> None:
    """Neither endpoint can be reassigned after construction."""
    bounds = IntRange(2, 5)

    with pytest.raises(FrozenInstanceError):
        setattr(bounds, endpoint, 0)

    assert bounds.min == 2
    assert bounds.max == 5


@pytest.mark.parametrize("value", ["1", object(), True, False])
def test_membership_rejects_nonintegers_and_bools(value: int) -> None:
    bounds = IntRange(2, 6)

    with pytest.raises(TypeError, match="membership requires an integer"):
        value in bounds  # ruff: ignore[useless-comparison]
