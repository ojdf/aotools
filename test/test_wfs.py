from aotools import wfs, circle
import numpy


def test_findActiveSubaps():
    subapertures = 10
    mask = circle(4, 10)
    threshold = .6
    active_subapertures = wfs.findActiveSubaps(subapertures, mask, threshold)
    assert active_subapertures.shape == (52, 2)


def test_findActiveSubaps_with_returnFill():
    subapertures = 10
    mask = circle(4, 10)
    threshold = .6
    active_subapertures, fill_factors = wfs.findActiveSubaps(subapertures, mask, threshold, returnFill=True)
    assert active_subapertures.shape == (52, 2)
    assert len(fill_factors) == 52


def test_make_subaps_2d():
    data = numpy.random.random((10, 2, 52))
    mask = circle(4, 10)
    sub_apertures_2d = wfs.make_subaps_2d(data, mask)
    assert sub_apertures_2d.shape == (10, 2, 10, 10)


def test_computeFillFactor():
    mask = circle(49, 100)
    sub_aperture_positions = numpy.array(([[10, 10], [10, 0]]))
    sub_aperture_spacing = 10
    fill_factor = wfs.computeFillFactor(mask, sub_aperture_positions, sub_aperture_spacing)
    assert len(fill_factor) == 2