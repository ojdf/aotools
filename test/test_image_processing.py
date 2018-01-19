from aotools import image_processing
import numpy


def test_r0fromSlopes():
    slopes = numpy.random.random((2, 100, 2))
    wavelength = 500e-9
    subapDiam = 0.5
    r0 = image_processing.r0fromSlopes(slopes, wavelength, subapDiam)
    print(type(r0))


def test_slopeVarfromR0():
    r0 = 0.1
    wavelength = 500e-9
    subapDiam = 0.5
    variance = image_processing.slopeVarfromR0(r0, wavelength, subapDiam)
    assert type(variance) == float


def test_image_contrast():
    image = numpy.random.random((20, 20))
    contrast = image_processing.image_contrast(image)
    assert type(contrast) == float


def test_rms_contrast():
    image = numpy.random.random((20, 20))
    contrast = image_processing.rms_contrast(image)
    assert type(contrast) == float


def test_encircled_energy():
    data = numpy.random.rand(32, 32)
    ee50d = image_processing.encircled_energy(data)
    print(ee50d)
    assert type(ee50d) == float


def test_encircled_energy_func():
    data = numpy.random.rand(32, 32)
    x, y = image_processing.encircled_energy(data, eeDiameter=False)
    print(y.min(), y.max())
    assert len(x) == len(y)
    assert numpy.max(y) <= 1
    assert numpy.min(y) >= 0


def test_azimuthal_average():
    data = numpy.random.rand(32, 32)
    azi = image_processing.azimuthal_average(data)
    print(azi.shape)
    assert azi.shape == (16,)
