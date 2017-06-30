from aotools.turbulence import atmos_conversions
import numpy


def test_cn2_to_seeing():
    cn2 = 5e-13
    seeing = atmos_conversions.cn2_to_seeing(cn2)
    assert type(seeing) == float


def test_cn2_to_r0():
    cn2 = 5e-13
    r0 = atmos_conversions.cn2_to_r0(cn2)
    assert type(r0) == float


def test_r0_to_seeing():
    r0 = 0.1
    seeing = atmos_conversions.r0_to_seeing(r0)
    assert type(seeing) == float


def test_conversion_consistency():
    cn2 = 5e-13
    r0 = atmos_conversions.cn2_to_r0(cn2)
    seeing_1 = atmos_conversions.r0_to_seeing(r0)
    seeing_2 = atmos_conversions.cn2_to_seeing(cn2)
    assert seeing_1 == seeing_2


def test_coherenceTime():
    cn2 = numpy.array((5e-13, 1e-14))
    v = numpy.array((10, 20))
    tau0 = atmos_conversions.coherenceTime(cn2, v)
    print(tau0, type(tau0))
    assert type(tau0) == float


def test_isoplanaticAngle():
    cn2 = numpy.array((5e-13, 1e-14))
    h = numpy.array((0., 10000.))
    isoplanatic_angle = atmos_conversions.isoplanaticAngle(cn2, h)
    assert type(isoplanatic_angle) == float
