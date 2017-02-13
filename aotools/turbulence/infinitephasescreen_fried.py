"""
An implementation of the "infinite phase screen", as deduced by Francois Assemat and Richard W. Wilson, 2006.
"""

from scipy.special import gamma, kv
from scipy import linalg
from scipy.interpolate import interp2d
import numpy
from numpy import pi

from . import phasescreen


def find_allowed_size(nx_size):
    n = 0
    while (2 ** n + 1) < nx_size:
        n += 1

    nx_size = 2 ** n + 1
    return nx_size


class PhaseScreen(object):
    """
    A "Phase Screen" for use in AO simulation using the Fried method.
    
    This represents the phase addition light experiences when passing through atmospheric 
    turbulence. Unlike other phase screen generation techniques that translate a large static 
    screen, this method keeps a small section of phase, and extends it as neccessary for as many 
    steps as required. This can significantly reduce memory consuption at the expense of more 
    processing power required.
    
    The technique is described in a paper by Assemat and Wilson, 2006 and expanded upon by Fried, 2008.
    It essentially assumes that there are two matrices, "A" and "B",
    that can be used to extend an existing phase screen.
    A single row or column of new phase can be represented by 
    
        X = A.Z + B.b
    
    where X is the new phase vector, Z is some data from the existing screen,
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

    The Z data is taken from points in a "stencil" defined by Fried that samples the entire screen.
    An additional "reference point" is also considered, that

    On initialisation an initial phase screen is calculated using an FFT based method.
    When 'addRows' is called, a new vector of phase is added to the phase screen.
    
    Parameters:
        nx_size (int): Size of phase screen (NxN)
        pixel_scale(float): Size of each phase pixel in metres
        r0 (float): fried parameter (metres)
        L0 (float): Outer scale (metres)
        random_seed (int, optional): seed for the random number generator
        stencil_length_factor (int, optional): How much longer is the stencil than the desired phase? default is 4
    """
    
    def __init__(self, nx_size, pixel_scale, r0, L0, random_seed=None, stencil_length_factor=4):


        self.requested_nx_size = nx_size
        self.nx_size = find_allowed_size(nx_size)
        print("New size: {}".format(self.nx_size))
        self.pixel_scale = pixel_scale
        self.r0 = r0
        self.L0 = L0
        self.stencil_length_factor = stencil_length_factor
        self.stencil_length = stencil_length_factor * self.nx_size


        if random_seed is not None:
            numpy.random.seed(random_seed)

        # Coordinate of Fried's "reference point" that stops the screen diverging
        self.reference_coord = (1, 1)

        self.set_X_coords()
        self.set_stencil_coords()

        self.calc_seperations()
        self.make_covmats()

        self.makeAMatrix()
        self.makeBMatrix()
        self.makeInitialScrn()

    def set_X_coords(self):
        """
        Sets the coords of X, the new phase vector.

        """

        self.X_coords = numpy.zeros((self.nx_size, 2))
        self.X_coords[:, 0] = -1
        self.X_coords[:, 1] = numpy.arange(self.nx_size)
        self.X_positions = self.X_coords * self.pixel_scale

    def set_stencil_coords(self):
        """
        Sets the Z coordinates, sections of the phase screen that will be used to create new phase

        """
        self.stencil = numpy.zeros((self.stencil_length, self.nx_size))

        max_n = 1
        while True:
            if 2**(max_n-1) + 1 >= self.nx_size:
                max_n -= 1
                break
            max_n += 1

        for n in range(0, max_n+1):
            col = int((2**(n - 1)) + 1)
            n_points = (2**(max_n - n)) + 1

            coords = numpy.round(numpy.linspace(0, self.nx_size-1, n_points)).astype('int32')
            self.stencil[col-1][coords] = 1

        # Now fill in tail of stencil
        for n in range(1, self.stencil_length_factor+1):
            col = n * self.nx_size - 1
            self.stencil[col, self.nx_size//2] = 1

        self.stencil_coords = numpy.array(numpy.where(self.stencil==1)).T
        self.stencil_positions = self.stencil_coords * self.pixel_scale

        self.n_stencils = len(self.stencil_coords)

    def calc_seperations(self):
        """
        Calculates the seperations between the phase points in the stencil and the new phase vector
        """
        positions = numpy.append(self.stencil_positions, self.X_positions, axis=0)
        self.seperations = numpy.zeros((len(positions), len(positions)))

        for i, (x1, y1) in enumerate(positions):
            for j, (x2, y2) in enumerate(positions):

                delta_x = x2 - x1
                delta_y = y2 - y1

                delta_r = numpy.sqrt(delta_x**2 + delta_y**2)

                self.seperations[i, j] = delta_r

    def make_covmats(self):
        """
        Makes the covariance matrices required for adding new phase
        """
        self.cov_mat = phaseCovariance(self.seperations, self.r0, self.L0)

        self.cov_mat_zz = self.cov_mat[:self.n_stencils, :self.n_stencils]
        self.cov_mat_xx = self.cov_mat[self.n_stencils:, self.n_stencils:]
        self.cov_mat_zx = self.cov_mat[:self.n_stencils, self.n_stencils:]
        self.cov_mat_xz = self.cov_mat[self.n_stencils:, :self.n_stencils]

    def makeAMatrix(self):
        """
        Calculates the "A" matrix, that uses the existing data to find a new 
        component of the new phase vector.
        """
        # Different inversion methods, not sure which is best
        cf = linalg.cho_factor(self.cov_mat_zz)
        inv_cov_zz = linalg.cho_solve(cf, numpy.identity(self.cov_mat_zz.shape[0]))

        self.A_mat = self.cov_mat_xz.dot(inv_cov_zz)

    def makeBMatrix(self):
        """
        Calculates the "B" matrix, that turns a random vector into a component of the new phase.
        """
        # Can make initial BBt matrix first
        BBt = self.cov_mat_xx - self.A_mat.dot(self.cov_mat_zx)

        # Then do SVD to get B matrix
        u, W, ut = numpy.linalg.svd(BBt)

        L_mat = numpy.zeros((self.nx_size, self.nx_size))
        numpy.fill_diagonal(L_mat, numpy.sqrt(W))

        # Now use sqrt(eigenvalues) to get B matrix
        self.B_mat = u.dot(L_mat)

    def makeInitialScrn(self):
        """
        Makes the initial screen usign FFT method that can be extended 
        """
        
        self._scrn = phasescreen.ft_phase_screen(
                self.r0, self.stencil_length, self.pixel_scale, self.L0, 1e-10
                )

        self._scrn = self._scrn[:, :self.nx_size]

    def addRow(self):
        """
        Adds new rows to the phase screen and removes old ones.
        
        Parameters:
            nRows (int): Number of rows to add
            axis (int): Axis to add new rows (can be 0 (default) or 1)
        """

        random_data = numpy.random.normal(0, 1, size=self.nx_size)

        stencil_data = self._scrn[(self.stencil_coords[:, 0], self.stencil_coords[:, 1])]

        reference_value = self._scrn[self.reference_coord]

        new_row = self.A_mat.dot(stencil_data - reference_value) + self.B_mat.dot(random_data) + reference_value
        new_row.shape = (1, self.nx_size)

        self._scrn = numpy.append(new_row, self._scrn, axis=0)[:self.stencil_length, :self.nx_size]

        return self.scrn

    @property
    def scrn(self):
        return self._scrn[:self.requested_nx_size, :self.requested_nx_size]



        
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
    
    B1 = (2**(-5./6)) * gamma(11./6)/(pi**(8./3))
    B2 = ((24./5) * gamma(6./5))**(5./6)
    
    C = (((2 * pi * r)/L0) ** (5./6)) * kv(5./6, (2 * pi * r)/L0)
    
    cov = A * B1 * B2 * C
    
    return cov
    
    
if __name__ == "__main__":
    

    from matplotlib import pyplot

    screen = PhaseScreen(64, 8./32, 0.2, 40)

    pyplot.ion()
    pyplot.imshow(screen.stencil)

    pyplot.figure()
    pyplot.imshow(screen.scrn)
    pyplot.colorbar()
    for i in range(100):
        screen.addRow()

        pyplot.clf()
        pyplot.imshow(screen.scrn)
        pyplot.colorbar()
        pyplot.draw()
        pyplot.pause(0.01)

    