import numpy
from aotools import interpolation


def test_zoom():
    image = numpy.random.random((10, 10))
    zoomed_image = interpolation.zoom(image, (50, 50))
    assert zoomed_image.shape == (50, 50)


def test_zoom_with_order():
    image = numpy.random.random((10, 10))
    zoomed_image = interpolation.zoom(image, (50, 50), order=5)
    assert zoomed_image.shape == (50, 50)


def test_zoom_with_complex():
    image = numpy.zeros((10, 10), dtype=numpy.complex64)
    zoomed_image = interpolation.zoom(image, (50, 50))
    assert zoomed_image.shape == (50, 50)


def test_bin():
    image = numpy.random.random((50, 50))
    binned_image = interpolation.binImgs(image, 5)
    assert binned_image.shape == (10, 10)


def test_bin_3D():
    image = numpy.random.random((5, 50, 50))
    binned_image = interpolation.binImgs(image, 5)
    assert binned_image.shape == (5, 10, 10)


def test_zoom_rbs():
    image = numpy.random.random((10, 10))
    zoomed_image = interpolation.zoom_rbs(image, (50, 50))
    assert zoomed_image.shape == (50, 50)


def test_zoom_rbs_with_order():
    image = numpy.random.random((10, 10))
    zoomed_image = interpolation.zoom_rbs(image, (50, 50), order=2)
    assert zoomed_image.shape == (50, 50)


def test_zoom_rbs_complex():
    image = numpy.zeros((10, 10), dtype=numpy.complex64)
    zoomed_image = interpolation.zoom_rbs(image, (50, 50), order=2)
    assert zoomed_image.shape == (50, 50)
