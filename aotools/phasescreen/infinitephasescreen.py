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
    A "Phase Screen" for use in AO simulation. 
    
    This represents the phase addition light experiences when passing through atmospheric 
    turbulence. Unlike other phase screen generation techniques that translate a large static 
    screen, this method keeps a small section of phase, and extends it as neccessary for as many 
    steps as required. This can significantly reduce memory consuption at the expense of more 
    processing power required.
    
    The technique is described in a paper by Assemat and Wilson, 2006. It essentially assumes that
    there are two matrices, "A" and "B", that can be used to extend an existing phase screen.
    A single row or column of new phase can be represented by 
    
        X = A.Z + B.b
    
    where X is the new phase vector, Z is some number of columns of the existing screen, 
    and b is a random vector with gaussian statistics.
    
    This object calculates the A and B matrices using an expression of the phase covariance when it
    is initialised. Calculating A is straightforward through the relationship:
        
        A =  Cov_xz . (Cov_zz)^(-1).
    
    B is less trivial.
    
        BB^t = Cov_xx - A.Cov_zx
        
    (where B^t is the transpose of B) is a symmetric matrix, hence B can be expressed as 
    
        B = UL, 
    
    where U and L are obtained from the svd for BB^t
    
        U, w, U^t = svd(BB^t)
    
    L is a diagonal matrix where the diagonal elements are w^(1/2).    
     
    On initialisation an initial phase screen is calculated using an FFT based method.
    When 'addRows' is called, a new vector of phase is added to the phase screen using `nCols`
    columns of previous phase. Assemat & Wilson claim that two columns are adequate for good
    atmospheric statistics. The phase in the screen data is always accessed as `<phasescreen>.scrn`.
    
    Parameters:
        nSize (int): Size of phase screen (NxN)
        pxlScale(float): Size of each phase pixel in metres
        r0 (float): fried parameter (metres)
        L0 (float): Outer scale (metres)
        nCol (int, optional): Number of columns to use to continue screen, default is 2
    """
    
    def __init__(self, nSize, pxlScale, r0, L0, nCol=2):
                           
        self.nSize = nSize
        self.pxlScale = pxlScale
        self.r0 = r0
        self.L0 = L0
        self.nCol = nCol

        self.makeAMatrix()
        self.makeBMatrix()
        self.makeInitialScrn()

    def makeXZSeperation(self):
        """
        Calculates a matrix where each element is the seperation in 
        metres between points in the new phase data and the existing data.
                
        Return:
            ndarray: Array of seperations
        """
        
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
        """
        Uses the seperation between new and existing phase points to 
        calculate the theoretical covariance between them
        """
        r_xz = self.makeXZSeperation()
        
        self.cov_xz_forwards = phaseCovariance(r_xz, self.r0, self.L0)
                
        # Make the covariance matrix for adding elements in the other direction.
        # This is the same, except the position of each of the columns is reversed
        self.cov_xz_backwards = numpy.zeros_like(self.cov_xz_forwards)
        
        totalSize = self.nCol * self.nSize
        for col in range(self.nCol):            
            self.cov_xz_backwards[:, col * self.nSize: (col+1) * self.nSize] \
                    = self.cov_xz_forwards[:, totalSize-(col+1)*self.nSize: totalSize-col * self.nSize]

    def makeZZSeperation(self):
        """
        Calculates a matrix where each element is the seperation in 
        metres between points in the the existing data.
                
        Return:
            ndarray: Array of seperations
        """
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
        """
        Uses the seperation between the existing phase points to calculate 
        the theoretical covariance between them
        """
        r_zz = self.makeZZSeperation()
        
        self.cov_zz = phaseCovariance(r_zz, self.r0, self.L0)


    def makeAMatrix(self):
        """
        Calculates the "A" matrix, that uses the existing data to find a new 
        component of the new phase vector. This is for propagating in axis 0.
        """
        self.makeXZCovMat()
        self.makeZZCovMat()
        
        # Different inversion methods, not sure which is best
        cf = linalg.cho_factor(self.cov_zz)
        inv_cov_zz = linalg.cho_solve(cf, numpy.identity(self.cov_zz.shape[0]))
        # inv_cov_zz = numpy.linalg.pinv(self.cov_zz)#, 0.001)    
        
        self.A_mat_forwards = self.cov_xz_forwards.dot(inv_cov_zz)
        self.A_mat_backwards = self.cov_xz_backwards.dot(inv_cov_zz) 


    def makeXXSeperation(self): 
        """
        Calculates a matrix where each element is the seperation in metres between 
        points in the new phase data points.
        
        Return:
            ndarray: Array of seperations
        """
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
        """
        Uses the seperation between the new phase points to calculate the theoretical 
        covariance between them
        """
        r_xx = self.makeXXSeperation()
        
        self.cov_xx = phaseCovariance(r_xx, self.r0, self.L0)

        
    def makeZXSeperation(self):
        """
        Calculates a matrix where each element is the seperation in metres between points 
        in the existing phase data and the new data.
                
        Return:
            ndarray: Array of seperations
        """
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
        """
        Uses the seperation between the existing and new phase points to calculate the
        theoretical covariance between them
        """
        r_xz = self.makeZXSeperation()
        
        self.cov_zx_forwards = phaseCovariance(r_xz, self.r0, self.L0)
        
        # Make the covariance matrix for adding elements in the other direction.
        # This is the same, except the position of each of the columns is reversed
        self.cov_zx_backwards = numpy.zeros_like(self.cov_zx_forwards)
        
        totalSize = self.nCol * self.nSize
        for col in range(self.nCol):            
            self.cov_zx_backwards[col * self.nSize: (col+1) * self.nSize] \
                    = self.cov_zx_forwards[totalSize-(col+1)*self.nSize: totalSize-col * self.nSize]
        
    def makeBMatrix(self):
        """
        Calculates the "B" matrix, that turns a random vector into a component of the new phase. 
        
        Finds a B matrix for the case of generating new data on each side of the phase screen.
        """
        self.makeXXCovMatrix()
        self.makeZXCovMatrix()

        # Make B matrix for each axis
        
        # Axis 0, forwards
        self.B_mat_forwards = self.makeSingleBMatrix(
                self.cov_xx, self.cov_zx_forwards, self.A_mat_forwards)
        
        # Axis 0, backwards
        self.B_mat_backwards = self.makeSingleBMatrix(
                self.cov_xx, self.cov_zx_backwards, self.A_mat_backwards)

        
    def makeSingleBMatrix(self, cov_xx, cov_zx, A_mat):
        """
        Makes the B matrix for a single direction
        
        Parameters:
            cov_xx: Matrix of XX covariance
            cov_zx: Matrix of ZX covariance
            A_mat: Corresponding A matrix
        """
        # Can make initial BBt matrix first
        BBt = cov_xx - A_mat.dot(cov_zx)
        
        # Then do SVD to get B matrix
        u, W, ut = numpy.linalg.svd(BBt)
        
        L_mat = numpy.zeros((self.nSize, self.nSize))
        numpy.fill_diagonal(L_mat, numpy.sqrt(W))
        
        # Now use sqrt(eigenvalues) to get B matrix
        B_mat = u.dot(L_mat) 
        
        return B_mat
    
    def makeInitialScrn(self):
        """
        Makes the initial screen usign FFT method that can be extended 
        """
        
        self.scrn = phasescreen.ft_phase_screen(
                self.r0, self.nSize, self.pxlScale, self.L0, 1e-10
                )
    
    def addRow(self, nRows=1, axis=0, direction=-1):
        """
        Adds new rows to the phase screen and removes old ones.
        
        Parameters:
            nRows (int): Number of rows to add
            axis (int): Axis to add new rows (can be 0 (default) or 1)
            direction (int): Direction to add row (-1 (default) or 1)
        """

        # Find parameters based on axis to add to and direction
        if direction == -1:
            # Forward direction
            A_mat = self.A_mat_forwards
            B_mat = self.B_mat_forwards
            
            # Coords to cut out to get existing phase
            x1_z = -self.nCol
            x2_z = None
            y1_z = 0
            y2_z = None
            
            # Coords to add in new phase
            newCoord = -1
                
        elif direction == 1:
            # Backwards
            A_mat = self.A_mat_backwards
            B_mat = self.B_mat_backwards
            
            # Coords to cut out to get existing phase
            x1_z = 0
            x2_z = self.nCol
            y1_z = 0
            y2_z = None
            
            # Coords to add in new phase
            newCoord = 0

        else:
            raise ValueError("Direction: {} not valid".format(direction))
        
        if axis not in [0, 1]:
            raise ValueError("Axis: {} not valid".format(axis))
        
        # Transpose if adding to axis 1
        if axis == 1:
            self.scrn = self.scrn.T
               
        for row in range(nRows):

            # Get a vector of values with gaussian stats
            beta = numpy.random.normal(size=self.nSize)
            
            # Get last two rows of previous screen
            Z = self.scrn[x1_z: x2_z, y1_z: y2_z].flatten()
                
            # Find new values
            X = A_mat.dot(Z) + B_mat.dot(beta)
            
            self.scrn = numpy.roll(self.scrn, direction, axis=0)
            
            self.scrn[newCoord] = X
            
        # Transpose back again 
        if axis == 1:
            self.scrn = self.scrn.T

    def __repr__(self):
        return str(self.scrn)
        
        
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
    for i in range(20):
        scrn.addRow(5)
        pyplot.clf()
        pyplot.imshow(scrn.scrn)
        pyplot.draw()
        pyplot.pause(0.00001)
        
    for i in range(20):
        scrn.addRow(5, direction=1)
        pyplot.clf()
        pyplot.imshow(scrn.scrn)
        pyplot.draw()
        pyplot.pause(0.00001)
        
    for i in range(20):
        scrn.addRow(5, axis=1)
        pyplot.clf()
        pyplot.imshow(scrn.scrn)
        pyplot.draw()
        pyplot.pause(0.00001)
        
    for i in range(20):
        scrn.addRow(5, direction=1, axis=1)
        pyplot.clf()
        pyplot.imshow(scrn.scrn)
        pyplot.draw()
        pyplot.pause(0.00001)
        
        
    
    