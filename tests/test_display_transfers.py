"""Test linear and gamma transfer functions used to build display lookup tables."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from dirigo.display import Gamma, Linear, TransferFunction


def test_linear_preserves_normalized_intensities() -> None:
    """Map every normalized intensity to the same display intensity."""
    values = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)

    result = Linear()(values)

    np.testing.assert_array_equal(result, values)


def test_gamma_defaults_to_point_four_five() -> None:
    """Use the previous display implementation's default gamma exponent."""
    assert Gamma().gamma == 0.45


@pytest.mark.parametrize(
    ("gamma", "expected"),
    [
        (0.5, [0.0, 0.25, 0.5, 0.75, 1.0]),
        (1.0, [0.0, 0.0625, 0.25, 0.5625, 1.0]),
        (2.0, [0.0, 0.00390625, 0.0625, 0.31640625, 1.0]),
    ],
)
def test_gamma_applies_the_power_law(
    gamma: float,
    expected: list[float],
) -> None:
    """Brighten, preserve, or darken intermediate values using the exponent."""
    values = np.array(
        [0.0, 0.0625, 0.25, 0.5625, 1.0],
        dtype=np.float64,
    )

    result = Gamma(gamma=gamma)(values)

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("transfer", [Linear(), Gamma(), Gamma(gamma=2.0)])
def test_transfer_domain_ends_at_one(transfer: TransferFunction) -> None:
    """Use one as the saturation endpoint for linear and gamma mappings."""
    assert transfer.input_max == 1.0


@pytest.mark.parametrize("transfer", [Linear(), Gamma(), Gamma(gamma=2.0)])
def test_transfer_preserves_input_shape_and_values(
    transfer: TransferFunction,
) -> None:
    """Accept read-only arrays without changing their values or output layout."""
    values = np.array(
        [[0.0, 0.1, 0.3], [0.6, 0.9, 1.0]],
        dtype=np.float64,
    )
    original = values.copy()
    values.flags.writeable = False

    result = transfer(values)

    assert result.shape == values.shape
    assert result.dtype == np.dtype(np.float64)
    assert np.isfinite(result).all()
    assert ((result >= 0.0) & (result <= 1.0)).all()
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("gamma", [1, 0.45, 2.0])
def test_gamma_accepts_positive_numeric_exponents(gamma: float) -> None:
    """Accept finite positive integer and floating-point exponents."""
    assert Gamma(gamma=gamma).gamma == gamma


@pytest.mark.parametrize("gamma", [True, False, None, "0.45"])
def test_gamma_rejects_invalid_parameter_types(gamma) -> None:  # ruff: ignore[missing-type-function-argument]
    """Reject booleans and nonnumeric exponents without coercing them."""
    with pytest.raises(TypeError, match="gamma must be a number"):
        Gamma(gamma=gamma)


@pytest.mark.parametrize(
    "gamma",
    [0.0, -0.5, float("nan"), float("inf"), float("-inf")],
)
def test_gamma_rejects_nonpositive_or_nonfinite_exponents(
    gamma: float,
) -> None:
    """Require a finite positive exponent to preserve a defined mapping at zero."""
    with pytest.raises(ValueError, match="gamma must be finite and positive"):
        Gamma(gamma=gamma)


def test_gamma_parameter_is_immutable() -> None:
    """Prevent parameter changes after a transfer func enters a settings snapshot."""
    transfer = Gamma(gamma=0.45)

    with pytest.raises(FrozenInstanceError):
        transfer.gamma = 2.0  # ty: ignore[invalid-assignment]

    assert transfer.gamma == 0.45
