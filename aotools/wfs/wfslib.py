"""
A library of functions which may be of use to analyse WFS data
"""

def r0fromSlopes(slopes, wavelength, subapDiam):
    """
    Measures the value of R0 from a set of WFS slopes.
    
    Uses the equation in Saint Jaques, 1998, PhD Thesis, Appendix A to calculate the value of atmospheric seeing parameter, r0, that would result in the variance of the given slopes. The slopes should represent only the X or Y directions.

    Parameters:
        slopes (ndarray): A 3-d set of slopes in radians, of shape (dimension, nSubaps, nFrames)
        wavelength (float): The wavelegnth of the light observed
        subapDiam (float) The diameter of each sub-aperture

    Returns:
        float: An estimate of r0 for that dataset.

    """
    onedSlopes = (slopes[0]**2 + slopes[1]**2)**0.5
    sVar = onedSlopes.var()
    r0 = ((0.162*(wavelength**2) * subapDiam**(-1./3)) / sVar)**(3./5)
    
    return r0