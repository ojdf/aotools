import numpy
from aotools.turbulence import phasescreen
from aotools import circle, opticalpropagation


def test_propagation_round_trip():
    screen = phasescreen.ft_phase_screen(0.16, 512, 4.2 / 512, 100, 0.01)

    # Input E Field
    E = numpy.exp(1j * screen)

    prop1 = opticalpropagation.angularSpectrum(E, 500e-9, 4.2 / 512, 4.2 / 512, 10000.)
    prop2 = opticalpropagation.angularSpectrum(prop1, 500e-9, 4.2 / 512, 4.2 / 512, -10000.)

    assert numpy.allclose(E, prop2)


def test_propagation_split_propagation():
    screen = phasescreen.ft_phase_screen(0.16, 512, 4.2 / 512, 100, 0.01)

    # Input E Field
    E = numpy.exp(1j * screen)

    prop1 = opticalpropagation.angularSpectrum(E, 500e-9, 4.2 / 512, 4.2 / 512, 5000.)
    prop2 = opticalpropagation.angularSpectrum(prop1, 500e-9, 4.2 / 512, 4.2 / 512, 5000.)
    prop3 = opticalpropagation.angularSpectrum(E, 500e-9, 4.2 / 512, 4.2 / 512, 10000.)

    assert numpy.allclose(prop2, prop3)


def test_propagation_conserves_intensity():
    screen = phasescreen.ft_phase_screen(0.16, 512, 4.2 / 512, 100, 0.01)

    # Input E Field
    E = numpy.exp(1j * screen)

    Em = E * circle(150, 512)
    sum1 = (abs(Em) ** 2).sum()
    prop1 = opticalpropagation.angularSpectrum(Em, 500e-9, 4.2 / 512, 4.2 / 512, 10000.)
    sum2 = (abs(prop1) ** 2).sum()

    assert numpy.allclose(sum1, sum2)


def test_one_step_fresnel():
    screen = phasescreen.ft_phase_screen(0.16, 512, 4.2 / 512, 100, 0.01)

    # Input E Field
    E = numpy.exp(1j * screen)

    E_out = opticalpropagation.oneStepFresnel(E, 500e-9, 4.2/512, 10000)


def test_two_step_fresnel():
    screen = phasescreen.ft_phase_screen(0.16, 512, 4.2 / 512, 100, 0.01)

    # Input E Field
    E = numpy.exp(1j * screen)

    E_out = opticalpropagation.twoStepFresnel(E, 500e-9, 4.2 / 512, 4.2 / 512, 10000)


def fresnel_comparison():
    screen = phasescreen.ft_phase_screen(0.16, 512, 4.2 / 512, 100, 0.01)

    # Input E Field
    E = numpy.exp(1j * screen)

    E_one = opticalpropagation.oneStepFresnel(E, 500e-9, 4.2 / 512, 10000)
    E_two = opticalpropagation.twoStepFresnel(E, 500e-9, 4.2 / 512, 4.2 / 512, 10000)

    assert numpy.allclose(E_one, E_two)


def test_lensAgainst():
    screen = phasescreen.ft_phase_screen(0.16, 512, 4.2 / 512, 100, 0.01)

    # Input E Field
    E = numpy.exp(1j * screen)

    E_out = opticalpropagation.lensAgainst(E, 500e-9, 4.2/512, 0.3)
