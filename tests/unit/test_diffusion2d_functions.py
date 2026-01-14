"""
Tests for functions in class SolveDiffusion2D
"""

import numpy
import pytest

from diffusion2d import SolveDiffusion2D


def test_initialize_domain():
    """
    Check function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()
    solver.initialize_domain(w=20.0, h=20.0, dx=0.5, dy=0.5)
    assert solver.nx == 40
    assert solver.ny == 40


def test_initialize_physical_parameters():
    """
    Checks function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()
    solver.dx = 0.5
    solver.dy = 0.5
    solver.initialize_physical_parameters(d=3.0, T_cold=200.0, T_hot=800.0)
    assert solver.dt == pytest.approx(0.02083333333, rel=1e-6)


def test_set_initial_condition():
    """
    Checks function SolveDiffusion2D.get_initial_function
    """
    solver = SolveDiffusion2D()
    solver.nx = 5
    solver.ny = 5
    solver.dx = 3
    solver.dy = 3
    solver.T_cold = 200.0
    solver.T_hot = 800.0

    expected_result = numpy.array(
        [
            [
                200.0,
                200.0,
                200.0,
                200.0,
                200.0,
            ],
            [
                200.0,
                200.0,
                200.0,
                200.0,
                200.0,
            ],
            [
                200.0,
                200.0,
                800.0,
                200.0,
                200.0,
            ],
            [
                200.0,
                200.0,
                200.0,
                200.0,
                200.0,
            ],
            [
                200.0,
                200.0,
                200.0,
                200.0,
                200.0,
            ],
        ]
    )

    assert numpy.array_equal(solver.set_initial_condition(), expected_result)
