from aotools import image_processing
import numpy


def test_centreOfGravity_single():
    img = numpy.random.random((10, 10))
    com = image_processing.centreOfGravity(img, 0.1)
    assert(com.shape[0]) == 2


def test_centreOfGravity_many():
    img = numpy.random.random((5, 10, 10))
    com = image_processing.centreOfGravity(img, 0.1)
    assert(com.shape[0] == 2)
    assert(com.shape[1] == 5)


def test_brightestPxl_single():
    img = numpy.random.random((10, 10))
    com = image_processing.brightestPxl(img, 0.3)
    assert(com.shape[0] == 2)


def test_brightestPxl_many():
    img = numpy.random.random((5, 10, 10))
    com = image_processing.brightestPxl(img, 0.1)
    assert(com.shape[0] == 2)
    assert(com.shape[1] == 5)


def test_quadCell_single():
    img = numpy.random.random((2, 2))
    com = image_processing.quadCell(img)
    assert(com.shape[0] == 2)


def test_quadCell_many():
    img = numpy.random.random((5, 2, 2))
    com = image_processing.quadCell(img)
    assert(com.shape[0] == 2)
    assert(com.shape[1] == 5)


def test_convolution():
    im = numpy.random.random((10, 10))
    ref = numpy.random.random((10, 10))
    corr = image_processing.corrConvolve(im, ref)
    assert(corr.shape == im.shape)


def test_correlation_single():
    im = numpy.random.random((10, 10))
    ref = numpy.random.random((10, 10))
    com = image_processing.correlation(im, ref, 0.3)
    assert(com.shape[0] == 2)


def test_correlation_many():
    im = numpy.random.random((5, 10, 10))
    ref = numpy.random.random((10, 10))
    com = image_processing.correlation(im, ref, 0.3)
    assert (com.shape[0] == 2)
    assert(com.shape[1] == 5)


