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