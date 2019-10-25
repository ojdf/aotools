from aotools import functions
import numpy


def test_gaussian2d():
    gaussian = functions.gaussian2d(10, 3, 10.)
    assert gaussian.shape == (10, 10)


def test_gaussian2d_2d():
    gaussian = functions.gaussian2d((10, 8), (3, 2), 10., (4, 3))
    print(gaussian.shape)
    assert gaussian.shape == (10, 8)
