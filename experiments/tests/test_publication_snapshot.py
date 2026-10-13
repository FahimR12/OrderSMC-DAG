import numpy as np
import pytest

from ordersmc_experiments.publication_snapshot import plot_ranges


def test_plot_ranges_accepts_only_roundoff_outside_extrema():
    observed=np.array([.25806451612903225])
    before=observed.copy()
    ranges=plot_ranges(observed+1e-16,observed,observed)
    assert (ranges>=0).all()
    assert np.array_equal(observed,before)
    with pytest.raises(ValueError,match="outside"):
        plot_ranges([.3],[.1],[.2])
