"""
An implementation of the "infinite phase screen", as deduced by Francois Assemat and Richard W. Wilson, 2006.
"""

from scipy.special import gamma, kv
from scipy import linalg
import numpy
from numpy import pi

from . import phasescreen

class PhaseScreen(object):
    """
    Parameters:
        nSize (int): Size of initial phase screen (NxN)
        pxlScale(float): Size of each phase pixel in metres
        r0 (float): fried parameter (metres)
        L0 (float): Outer scale (metres)
        nCol (int): Number of columns to use to continue screen
    """
    
    def __init__(self, nSize, pxlScale, r0, L0, nCol=2):
        
        self.nSize = nSize
        self.pxlScale = pxlScale
        self.r0 = r0
        self.L0 = L0
        self.nCol = nCol
        
        self.A_mat = None
        self.B_mat = None

        self.makeAMatrix()
        self.makeBMatrix()
        self.makeInitialScrn()

    def makeXZSeperation(self):
        
        # First, find matrix of seperations between all points.
        r_xz = numpy.zeros((self.nSize, self.nCol*self.nSize))
        
        for i in range(self.nSize):
            for n in range(self.nCol):
                for j in range(self.nSize):
                    
                    # Assume first of used columns is zero point in x
                    Z_x = n * self.pxlScale  
                    Z_y = j * self.pxlScale
                    
                    # Adding only in x so new column always has same X pos
                    X_x = self.nCol * self.pxlScale
                    X_y = i * self.pxlScale
                    
                    dx = X_x - Z_x
                    dy = X_y - Z_y
                    
                    r = numpy.sqrt(dx**2 + dy**2)
                    xCoord = i
                    yCoord = n*self.nSize + j
                    r_xz[xCoord, yCoord] = r
                    # print("Point ({}) = {}".format((xCoord, yCoord), r))
                    
        return r_xz
        
    def makeXZCovMat(self):
        r_xz = self.makeXZSeperation()
        
        self.cov_xz = phaseCovariance(r_xz, self.r0, self.L0)
       

    def makeZZSeperation(self):
        # First, find matrix of seperations between all points.
        r_zz = numpy.zeros((self.nCol*self.nSize, self.nCol*self.nSize))
        
        for ni in range(self.nCol):
            for i in range(self.nSize):
                for nj in range(self.nCol):
                    for j in range(self.nSize):
                        # Assume first of used columns is zero point in x
                        Z1_x = nj * self.pxlScale  
                        Z1_y = j * self.pxlScale
                        
                        # Adding only in x so new column always has same X pos
                        Z2_x = ni * self.pxlScale
                        Z2_y = i * self.pxlScale
                        
                        dx = Z2_x - Z1_x
                        dy = Z2_y - Z1_y
                        
                        r = numpy.sqrt(dx**2 + dy**2)
                        xCoord = ni * self.nSize + i
                        yCoord = nj * self.nSize + j
                        r_zz[xCoord, yCoord] = r
                        # print("Point ({}) = {}".format((xCoord, yCoord), r))
                    
        return r_zz
        
    def makeZZCovMat(self):
        r_zz = self.makeZZSeperation()
        
        self.cov_zz = phaseCovariance(r_zz, self.r0, self.L0)

    def makeAMatrix(self):
        
        self.makeXZCovMat()
        self.makeZZCovMat()
        
        # Difference inversion methods, not sure which is best
        cf = linalg.cho_factor(self.cov_zz)
        inv_cov_zz = linalg.cho_solve(cf, numpy.identity(self.cov_zz.shape[0]))
        # inv_cov_zz = numpy.linalg.pinv(self.cov_zz)#, 0.001)
        self.A_mat = self.cov_xz.dot(inv_cov_zz) 


    def makeXXSeperation(self):
        # First, find matrix of seperations between all points.
        r_xx = numpy.zeros((self.nSize, self.nSize))
        
        for i in range(self.nSize):
            for j in range(self.nSize):
                    
                # Assume first of used columns is zero point in x
                X1_x = 0 
                X1_y = j * self.pxlScale
                
                # Adding only in x so new column always has same X pos
                X2_x = 0
                X2_y = i * self.pxlScale
                
                dx = X2_x - X1_x
                dy = X2_y - X1_y
                
                r = numpy.sqrt(dx**2 + dy**2)
                xCoord = i
                yCoord = j
                r_xx[xCoord, yCoord] = r
                # print("Point ({}) = {}".format((xCoord, yCoord), r))
                    
        return r_xx

    def makeXXCovMatrix(self):
        
        r_xx = self.makeXXSeperation()
        
        self.cov_xx = phaseCovariance(r_xx, self.r0, self.L0)

        
    def makeZXSeperation(self):
        # First, find matrix of seperations between all points.
        r_xz = numpy.zeros((self.nCol*self.nSize, self.nSize))
        
        for n in range(self.nCol):
            for i in range(self.nSize):
                for j in range(self.nSize):
                    
                    # Assume first of used columns is zero point in x
                    X_x = self.nCol * self.pxlScale  
                    X_y = j * self.pxlScale
                    
                    # Adding only in x so new column always has same X pos
                    Z_x = n * self.pxlScale
                    Z_y = i * self.pxlScale
                    
                    dx = Z_x - X_x
                    dy = Z_y - X_y
                    
                    r = numpy.sqrt(dx**2 + dy**2)
                    xCoord = n * self.nSize + i
                    yCoord = j
                    r_xz[xCoord, yCoord] = r
                    # print("Point ({}) = {}".format((xCoord, yCoord), r))
                    
        return r_xz
        
    def makeZXCovMatrix(self):
        r_xz = self.makeZXSeperation()
        
        self.cov_zx = phaseCovariance(r_xz, self.r0, self.L0)

        
    def makeBMatrix(self):
        
        self.makeXXCovMatrix()
        self.makeZXCovMatrix()
        
        if self.A_mat is None:
            self.makeAMatrix()

        # Can make initial BBt matrix first
        self._BBt = self.cov_xx - self.A_mat.dot(self.cov_zx)
        
        # Then do SVD to get B matrix
        self._u, self._W, ut = numpy.linalg.svd(self._BBt)
        
        self._L_mat = numpy.zeros((self.nSize, self.nSize))
        numpy.fill_diagonal(self._L_mat, numpy.sqrt(self._W))
        
        # Now use sqrt(eigenvalues) to get B matrix
        self.B_mat = self._u.dot(self._L_mat) 
        
    
    def makeInitialScrn(self):
        """
        Makes the initial screen to extend 
        """
        
        self.scrn = phasescreen.ft_phase_screen(
                self.r0, self.nSize, self.pxlScale, self.L0, 1e-10
                )
            
    
    def addRow(self, nRows=1):
        """
        Adds new rows to the phse screen and removes old ones.
        
        Parameters:
            nRows (int): Number of rows to add
        """
        for row in range(nRows):
        # Get a vector of values with gaussian stats
            beta = numpy.random.normal(size=self.nSize)
            
            # Get last two rows of previous screen
            Z = self.scrn[-self.nCol:].flatten()
            
            # Find new values
            X = self.A_mat.dot(Z) + self.B_mat.dot(beta)
            
            self.scrn = numpy.roll(self.scrn, -1, axis=0)
            self.scrn[-1] = X
            

def phaseCovariance(r, r0, L0):
    """
    Calculate the phase covariance between two points seperated by `r`, 
    in turbulence with a given `r0 and `L0`.
    Uses equation 5 from Assemat and Wilson, 2006.
    
    Parameters:
        r (float, ndarray): Seperation between points in metres (can be ndarray)
        r0 (float): Fried parameter of turbulence in metres
        L0 (float): Outer scale of turbulence in metres
    """
    # Make sure everything is a float to avoid nasty surprises in division!
    r = numpy.float32(r)
    r0 = float(r0)
    L0 = float(L0)
    
    # Get rid of any zeros
    r += 1e-40
    
    A = (L0/r0)**(5./3) 
    
    B1 = (2**(1./6)) * gamma(11./6)/(pi**(8./3))
    B2 = ((24./5) * gamma(6./5))**(5./6)
    
    C = (((2 * pi * r)/L0) ** (5./6)) * kv(5./6, (2 * pi * r)/L0)
    
    cov = A * B1 * B2 * C
    
    return cov

    
if __name__ == "__main__":
    
    
    scrn = PhaseScreen(64, 4./64, 0.2, 50, nCol=4)
    
    from matplotlib import pyplot
    pyplot.ion()
    pyplot.figure()
    pyplot.imshow(scrn.scrn)
    pyplot.colorbar()
    for i in range(100):
        scrn.addRow(5)
        pyplot.clf()
        pyplot.imshow(scrn.scrn)
        pyplot.draw()
        pyplot.pause(0.00001)
        
        
    
    