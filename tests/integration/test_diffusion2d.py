"""
Tests for functionality checks in class SolveDiffusion2D
"""

import numpy
import pytest

from diffusion2d import SolveDiffusion2D


def test_initialize_physical_parameters():
    """
    Checks function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()
    solver.initialize_domain(w=20.0, h=20.0, dx=5.0, dy=5.0)
    solver.initialize_physical_parameters(d=5.0, T_cold=400.0, T_hot=600.0)
    assert solver.dt == pytest.approx(1.25, rel=1e-6)


def test_set_initial_condition():
    """
    Checks function SolveDiffusion2D.get_initial_function
    """
    solver = SolveDiffusion2D()
    solver.initialize_domain(w=20.0, h=20.0, dx=2.0, dy=2.0)
    solver.initialize_physical_parameters(d=5.0, T_cold=400.0, T_hot=600.0)
    actual_result = solver.set_initial_condition()

    expected_result = 400.0 * numpy.ones((10, 10))
    for i in range(10):
        for j in range(10):
            p2 = (i * 2.0 - 5.0) ** 2 + (j * 2.0 - 5.0) ** 2
            if p2 < 4:
                expected_result[i, j] = 600.0

    assert numpy.allclose(actual_result, expected_result)
