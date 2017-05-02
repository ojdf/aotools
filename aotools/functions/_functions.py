import numpy


def gaussian2d(size, width, amplitude=1., cent=None):
    '''
    Generates 2D gaussian distribution


    Args:
        size (tuple, float): Dimensions of Array to place gaussian
        width (tuple, float): Width of distribution.
                                Accepts tuple for x and y values.
        amplitude (float): Amplitude of guassian distribution
        cent (tuple): Centre of distribution on grid.
    '''

    try:
        xSize = size[0]
        ySize = size[1]
    except (TypeError, IndexError):
        xSize = ySize = size

    try:
        xWidth = float(width[0])
        yWidth = float(width[1])
    except (TypeError, IndexError):
        xWidth = yWidth = float(width)

    if not cent:
        xCent = size[0] / 2.
        yCent = size[1] / 2.
    else:
        xCent = cent[0]
        yCent = cent[1]

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
    avg = numpy.empty(size / 2, dtype="float")
    for i in range(size / 2):
        ring = circle(i + 1, size) - circle(i, size)
        avg[i] = (ring * data).sum() / (ring.sum())

    return avg
