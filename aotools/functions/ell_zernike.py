#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from aotools.functions import zernike

'''
Normalised "Zernike" polynomials for the elliptical aperture.
Based on Virendra N. Mahajan and Guang-ming Dai, "Orthonormal polynomials in wavefront analysis: analytical solution," J. Opt. Soc. Am. A 24, 2994-3016 (2007).
'''

class ZernikeEllipticalaperture:
    def __init__(self, nterms, npix, a, b, ell_aperture=True, coeff=None):
        self.nterms = nterms    #maximum radial order
        self.npix = npix    #number of pixels for the map
        self.a = a          #major semi-axis (normalised)
        self.b = b          #minor semi-axis (normalised)
        self.ell_aperture = ell_aperture  # Specify whether to use the elliptical aperture. False leads to a standard circular aperture
        self.coeff = coeff  # Coefficients for the Zernike modes (optional). Default random.
        self.ell_aperture_mask = self.GenerateEllipticalAperture()
        self.circular_zern = self.GetCircZernikeValue()
        self.E = self.CalculateEllipticalZernike()

    def GetCircZernikeValue(self):
        '''
        Get values of circular (standard) Zernike polynomials.

        Return: zern_value
        '''
        zern_value = zernike.zernikeArray(self.nterms, self.npix)
        zern_value = np.array(zern_value / np.linalg.norm(zern_value)).squeeze()
        return zern_value

    def CalculateEllipticalZernike(self):
        '''
        Calculate elliptical orthonormal polynomials as a product of the conversion matrix and circular zernike polynomials.

        Return: E
        '''

        Z = self.circular_zern
        M = self.M_matrix()
        E = []  # Initialize a list to store E arrays for each l

        for i in range(1, self.nterms + 1):
            E_l = np.zeros(Z[0].shape)  # Initialize E with the same shape as Z[0]
            for j in range(1, i + 1):
                E_l += M[i - 1, j - 1] * Z[j - 1]
            E.append(E_l)

        E = np.array(E)
        if self.ell_aperture == True:
            E[:, np.logical_not(self.ell_aperture_mask)] = 0
        return E

    def M_matrix(self):
        '''
        Calculate the conversion matrix.

        Return: M
        '''
        C = self.C_zern()
        regularization = 1e-6  # Small positive constant to regularize
        C += regularization * np.eye(C.shape[0])

        Q = np.linalg.cholesky(C)
        QT = np.transpose(Q)
        M = np.linalg.inv(QT)
        return M

    def C_zern(self):
        '''
        Calculate C_zern matrix which is a symmetric matrix of inner products of Zernike polynomials
        over the domain of a noncircular pupil (elliptical).

        Return: C
        '''
        # Initialize the C matrix
        C = np.zeros((self.nterms, self.nterms))
        # Calculate the area of each grid cell
        dx = (2 * self.a) / 10000
        dy = (2 * self.b) / 10000

        for i in range(self.nterms):
            for j in range(i, self.nterms):
                product_Zern = np.dot(self.circular_zern[i], self.circular_zern[j]) * dx * dy
                C[i, j] += np.sum(product_Zern)
                if i != j:
                    C[j, i] = C[i, j]

        return C

    def GenerateEllipticalAperture(self):

        x, y = np.meshgrid(np.linspace(-1, 1, self.npix), np.linspace(-1, 1, self.npix))
        normalized_distance = (x / self.a) ** 2 + (y / self.b) ** 2
        aperture = (normalized_distance <= 1).astype(float)
        return aperture

    def EllZernikeMap(self, coeff=None):
        '''
        Generate a wavefront map on an elliptical pupil.
        Parameters:
        coeff (array): coefficients for the polynomial terms. Must be the same length as the number of elliptical modes.

        Return: phi
        '''
        xx, yy = np.meshgrid(np.linspace(-1, 1, self.npix), np.linspace(-1, 1, self.npix))
        E_ell = np.zeros((xx.size, self.nterms))

        for k in range(self.nterms):
            E_ell[:, k] = np.ravel(self.E[k])

        if coeff is None:
            coeff = np.random.random(self.nterms)

        if len(coeff) != self.nterms:
            raise ValueError(f"Coefficient array must have length {self.nterms}, but got {len(coeff)}.")

        phi = np.dot(E_ell, coeff)
        phi = phi.reshape(xx.shape)

        return phi
