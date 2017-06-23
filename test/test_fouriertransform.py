from aotools import fouriertransform
import numpy


def test_ft():
    data = numpy.random.random((100))
    ft_data = fouriertransform.ft(data, 0.1)
    assert ft_data.shape == data.shape


def test_ift():
    data = numpy.random.random((100))
    ift_data = fouriertransform.ift(data, 1.)
    assert ift_data.shape == data.shape


def test_ft2():
    data = numpy.zeros((10, 10))
    ft_data = fouriertransform.ft2(data, 1.)
    assert ft_data.shape == data.shape


def test_ift2():
    data = numpy.zeros((10, 10))
    ift_data = fouriertransform.ift2(data, 1.)
    assert ift_data.shape == data.shape
