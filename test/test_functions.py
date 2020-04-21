from aotools import functions
import numpy


def test_gaussian2d():
    gaussian = functions.gaussian2d(10, 3, 10.)
    assert gaussian.shape == (10, 10)


def test_gaussian2d_array_centre_point():
    gaussian = functions.gaussian2d((10, 8), (3, 2), 10., numpy.asarray((4, 3)))
    assert gaussian.shape == (10, 8)


def test_gaussian2d_2d():
    gaussian = functions.gaussian2d((10, 8), (3, 2), 10., (4, 3))
    assert gaussian.shape == (10, 8)
