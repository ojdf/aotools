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

def test_equivalent_layers():
    h = numpy.arange(0,25000,250)
    p = numpy.ones(len(h)) * 100e-17
    L = 5
    h_el, p_el = turbulence.equivalent_layers(h, p, L)
    assert type(h_el) and type(p_el) == numpy.ndarray
    assert len(h_el) and len(p_el) == L

def test_optimal_grouping():
    h = numpy.arange(0,25000,250)
    p = numpy.ones(len(h)) * 100e-17
    L = 5
    R = 1
    h_el, p_el = turbulence.optimal_grouping(R, L, h, p)
    assert type(h_el) and type(p_el) == numpy.ndarray
    assert len(h_el) and len(p_el) == L

def test_GCTM():
    h = numpy.arange(0,25000,250)
    p = numpy.ones(len(h)) * 100e-17
    L = 5
    h_el, p_el = turbulence.GCTM(h, p, L)
    assert type(h_el) and type(p_el) == numpy.ndarray
    assert len(h_el) and len(p_el) == L

