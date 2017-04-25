from aotools import image_processing
import numpy


def test_centreOfGravity_single():
    img = numpy.random.random((10, 10))
    com = image_processing.centreOfGravity(img, 0.1)

def test_centreOfGravity_many():
    img = numpy.random.random((5, 10, 10))
    com = image_processing.centreOfGravity(img, 0.1)
    assert(com.shape[1] == 5)


def test_brightestPxl():
    img = numpy.random.random((10, 10))
    com = image_processing.brightestPxl(img, 0.3)

def test_quadCell_single():
    img = numpy.random.random((2, 2))
    com = image_processing.quadCell(img)

def test_quadCell_many():
    img = numpy.random.random((5, 2, 2))
    com = image_processing.quadCell(img)
    assert(com.shape[1] == 5)
