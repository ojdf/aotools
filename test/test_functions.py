from aotools import functions
import numpy


def test_gaussian2d():
    gaussian = functions.gaussian2d(10, 3, 10.)
    assert gaussian.shape == (10, 10)


def test_gaussian2d_2d():
    gaussian = functions.gaussian2d((10, 8), (3, 2), 10., (4, 3))
    print(gaussian.shape)
    assert gaussian.shape == (10, 8)


def test_encircledEnergy():
    data = numpy.random.rand(32, 32)
    ee50d = functions.encircledEnergy(data)
    print(ee50d)
    assert type(ee50d) == float


def test_encircledEnergy_func():
        data = numpy.random.rand(32, 32)
        x, y = functions.encircledEnergy(data, eeDiameter=False)
        print(y.min(), y.max())
        assert len(x) == len(y)
        assert numpy.max(y) <= 1
        assert numpy.min(y) >= 0
