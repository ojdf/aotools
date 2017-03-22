'''
Note that these tests only cover angular spectrum propagation!
'''
from __future__ import print_function

import unittest

import numpy

import logging
from aotools.turbulence import opticalpropagation, phasescreen
from aotools import circle

logging.basicConfig()

log = logging.getLogger(__name__)

log.setLevel(2)

class TestPropagation(unittest.TestCase):
    def test_propagation(self):

        log.info('Make phase screen for propagation')
        scrn = phasescreen.ft_phase_screen(0.16, 512, 4.2/512, 100, 0.01)
        
        # Input E Field
        E = numpy.exp(1j*scrn)

        log.info('Propagate phase screen 10 km and back, should be no difference')
        prop1 = opticalpropagation.angularSpectrum(E, 500e-9, 4.2/512, 4.2/512, 10000.)
        prop2 = opticalpropagation.angularSpectrum(prop1, 500e-9, 4.2/512, 4.2/512, -10000.)
        self.assertEqual(E.all(), prop2.all())
        self.assertTrue(numpy.allclose(E, prop2))

        log.info('Propagate phase screen 5 km twice, compare to single 10 km prop')
        prop1 = opticalpropagation.angularSpectrum(E, 500e-9, 4.2/512, 4.2/512, 5000.)
        prop2 = opticalpropagation.angularSpectrum(prop1, 500e-9, 4.2/512, 4.2/512, 5000.)
        prop3 = opticalpropagation.angularSpectrum(E, 500e-9, 4.2/512, 4.2/512, 10000.)
        self.assertEqual(prop2.all(), prop3.all())
        self.assertTrue(numpy.allclose(prop2, prop3))

        log.info('Make sure the total intensity sum inside an aperture is preserved')
        # Mask E field with circular aperture
        Em = E * circle(150,512)
        sum1 = (abs(Em)**2).sum()
        prop1 = opticalpropagation.angularSpectrum(Em, 500e-9, 4.2/512, 4.2/512, 10000.)
        sum2 = (abs(prop1)**2).sum()
        self.assertAlmostEqual(sum1, sum2)

if __name__ == "__main__":
    t = TestPropagation()
    t.test_propagation()
