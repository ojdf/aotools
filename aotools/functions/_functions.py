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
