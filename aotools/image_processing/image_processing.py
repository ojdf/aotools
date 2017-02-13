def r0fromSlopes(slopes, wavelength, subapDiam):
    """
    Measures the value of R0 from a set of WFS slopes.

    Uses the equation in Saint Jaques, 1998, PhD Thesis, Appendix A to calculate the value of atmospheric seeing parameter, r0, that would result in the variance of the given slopes.

    Parameters:
        slopes (ndarray): A 3-d set of slopes in radians, of shape (dimension, nSubaps, nFrames)
        wavelength (float): The wavelegnth of the light observed
        subapDiam (float) The diameter of each sub-aperture

    Returns:
        float: An estimate of r0 for that dataset.

    """
    slopeVar = slopes.var(axis=(-1))

    r0 = ((0.162 * (wavelength ** 2) * subapDiam ** (-1. / 3)) / slopeVar) ** (3. / 5)

    r0 = r0.mean()

    return r0


def slopeVarfromR0(r0, wavelength, subapDiam):
    """Returns the expected slope variance for a given r0 ValueError

    Uses the equation in Saint Jaques, 1998, PhD Thesis, Appendix A to calculate the slope variance resulting from a value of r0.

    """

    slope_var = 0.162 * (wavelength ** 2) * r0 ** (-5. / 3) * subapDiam ** (-1. / 3)

    return slope_var


def image_contrast(image):
    """
    Calculates the 'Michelson' contrast.

    Uses a method by Michelson (Michelson, A. (1927). Studies in Optics. U. of Chicago Press.), to calculate the contrast ratio of an image. Uses the formula:
        (img_max - img_min)/(img_max + img_min)

    Parameters:
        image (ndarray): Image array

    Returns:
        float: Contrast value
    """

    contrast = (image.max() - image.min()) / (image.max() + image.min())

    return contrast


def rms_contrast(image):
    """
    Calculates the RMS contrast - basically the standard deviation of the image

    Parameters:
        image (ndarray): Image array

    Returns:
        float: Contrast value
    """

    image /= image.max()

    return image.std()
