from aotools import image_processing
import numpy


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


def test_encircledEnergy():
    data = numpy.random.rand(32, 32)
    ee50d = image_processing.encircled_energy(data)
    print(ee50d)
    assert type(ee50d) == float


def test_encircledEnergy_func():
    data = numpy.random.rand(32, 32)
    x, y = image_processing.encircled_energy(data, eeDiameter=False)
    print(y.min(), y.max())
    assert len(x) == len(y)
    assert numpy.max(y) <= 1
    assert numpy.min(y) >= 0


def test_aziAvg():
    data = numpy.random.rand(32, 32)
    azi = image_processing.azimuthal_average(data)
    print(azi.shape)
    assert azi.shape == (16,)