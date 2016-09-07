import numpy

def calcSlopeTemporalPowerSpectrum(slope_data):
    """
    Calculates the temporal power spectra of the loaded centroid data.

    Calculates the Fourier transform over the number frames of the centroid
    data, then returns the square of the  mean of all sub-apertures, for x
    and y. This is a temporal power spectra of the slopes, and should adhere
    to a -11/3 power law for the slopes in the wind direction, and -14/3 in
    the direction tranverse to the wind direction. See Conan, 1995 for more.

    The phase slope data should be split into X and Y components, with all X data first, then Y (or visa-versa).

    Parameters:
        slope_data (ndarray): 2-d array of shape (nFrames, nCentroids)

    Returns:
        ndarray: The temporal power spectra of the data for X and Y components.
    """

    nFrames = slope_data.shape[0]
    totalSubaps = slope_data.shape[1]/2

    # Only take half result, as FFT mirrors
    tPS = numpy.fft.fft(slope_data, axis=0)[:nFrames/2]

    # Split TPS into X and Y slopes.
    meanTPSx = tPS[:, :totalSubaps].mean(1)
    meanTPSy = tPS[:, totalSubaps:].mean(1)

    return numpy.array([abs(meanTPSx)**2, abs(meanTPSy)**2])
