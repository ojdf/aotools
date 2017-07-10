import numpy
from . import pupil


def gaussian2d(size, width, amplitude=1., cent=None):
    '''
    Generates 2D gaussian distribution


    Args:
        size (tuple, float): Dimensions of Array to place gaussian (y, x)
        width (tuple, float): Width of distribution.
                                Accepts tuple for x and y values in order (y, x).
        amplitude (float): Amplitude of guassian distribution
        cent (tuple): Centre of distribution on grid in order (y, x).
    '''

    try:
        ySize = size[0]
        xSize = size[1]
    except (TypeError, IndexError):
        xSize = ySize = size

    try:
        yWidth = float(width[0])
        xWidth = float(width[1])
    except (TypeError, IndexError):
        xWidth = yWidth = float(width)

    if not cent:
        xCent = xSize/2.
        yCent = ySize/2.
    else:
        yCent = cent[0]
        xCent = cent[1]

    X, Y = numpy.meshgrid(range(0, xSize), range(0, ySize))

    image = amplitude * numpy.exp(
        -(((xCent - X) / xWidth) ** 2 + ((yCent - Y) / yWidth) ** 2) / 2)

    return image


def aziAvg(data):
    """
    Measure the azimuthal average of a 2d array

    Args:
        data (ndarray): A 2-d array of data

    Returns:
        ndarray: A 1-d vector of the azimuthal average
    """

    size = data.shape[0]
    avg = numpy.empty(int(size / 2), dtype="float")
    for i in range(int(size / 2)):
        ring = pupil.circle(i + 1, size) - pupil.circle(i, size)
        avg[i] = (ring * data).sum() / (ring.sum())

    return avg


def encircledEnergy(data,
                    fraction=0.5, center=None,
                    eeDiameter=True):
    """
        Return the encircled energy diameter for a given fraction
        (default is ee50d).
        Can also return the encircled energy function.
        Translated and extended from YAO.

        Parameters:
            data : 2-d array
            fraction : energy fraction for diameter calculation
                default = 0.5
            center : default = center of image
            eeDiameter : toggle option for return.
                If False returns two vectors: (x, ee(x))
                Default = True
        Returns:
            Encircled energy diameter
            or
            2 vectors: diameters and encircled energies

    """
    dim = data.shape[0] // 2
    center = [dim, dim]
    xc = center[0]
    yc = center[1]
    e = 1.9
    npt = 20
    rad = numpy.linspace(0, dim**(1. / e), npt)**e
    ee = numpy.empty(rad.shape)

    for i in range(npt):
        pup = pupil.circle(rad[i],
                           int(dim) * 2,
                           circle_centre=(xc + 0.5,
                                          yc + 0.5),
                           origin='corner')
        rad[i] = numpy.sqrt(numpy.sum(pup) * 4 / numpy.pi)  # diameter
        ee[i] = numpy.sum(pup * data)

    rad = numpy.append(0, rad)
    ee = numpy.append(0, ee)
    ee /= numpy.sum(data)
    xi = numpy.linspace(0, dim, int(2 * dim))
    yi = numpy.interp(xi, rad, ee)

    if eeDiameter is False:
        return xi, yi
    else:
        ee50d = float(xi[numpy.argmin(numpy.abs(yi - fraction))])
        return ee50d
