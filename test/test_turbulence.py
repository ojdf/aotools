from aotools import turbulence
import numpy


def test_r0fromSlopes():
    slopes = numpy.random.random((2, 100, 2))
    wavelength = 500e-9
    subapDiam = 0.5
    r0 = turbulence.r0_from_slopes(slopes, wavelength, subapDiam)
    print(type(r0))


def test_slopeVarfromR0():
    r0 = 0.1
    wavelength = 500e-9
    subapDiam = 0.5
    variance = turbulence.slope_variance_from_r0(r0, wavelength, subapDiam)
    assert type(variance) == float
