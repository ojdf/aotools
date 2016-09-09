"""
Turbulence gradient temporal power spectra calculation and plotting

:author: Andrew Reeves
:date: September 2016
"""

import numpy
from matplotlib import pyplot


def calc_slope_temporalps(slope_data):
    """
    Calculates the temporal power spectra of the loaded centroid data.

    Calculates the Fourier transform over the number frames of the centroid
    data, then returns the square of the  mean of all sub-apertures, for x
    and y. This is a temporal power spectra of the slopes, and should adhere
    to a -11/3 power law for the slopes in the wind direction, and -14/3 in
    the direction tranverse to the wind direction. See Conan, 1995 for more.

    The phase slope data should be split into X and Y components, with all X data first, then Y (or visa-versa).

    Parameters:
        slope_data (ndarray): 2-d array of shape (..., nFrames, nCentroids)

    Returns:
        ndarray: The temporal power spectra of the data for X and Y components.
    """

    n_frames = slope_data.shape[-2]

    # Only take half result, as FFT mirrors
    tps = abs(numpy.fft.fft(slope_data, axis=-2)[..., :n_frames/2, :])**2

    # Find mean across all sub-aps
    mean_tps = (abs(tps)**2).mean(-1)

    return mean_tps


def get_tps_time_axis(frame_rate, n_frames):
    """
    Parameters:
        frame_rate (float): Frame rate of detector observing slope gradients (Hz)
        n_frames: (int): Number of frames used for temporal power spectrum

    Returns:
        ndarray: Time values for temporal power spectra plots
    """

    t_vals = numpy.fft.fftfreq(n_frames, 1./frame_rate)[:n_frames/2.]

    return t_vals



def plot_tps(slope_data, frame_rate):
    """
    Generates a plot of the temporal power spectrum/a for a data set of phase gradients

    Parameters:
        slope_data (ndarray):  2-d array of shape (..., nFrames, nCentroids)
        frame_rate (float): Frame rate of detector observing slope gradients (Hz)

    Returns:

    """
    n_frames = slope_data.shape[-2]

    tps = calc_slope_temporalps(slope_data)

    t_axis_data = get_tps_time_axis(frame_rate, n_frames)

    fig = pyplot.figure()
    ax = fig.add_subplot(111)

    # plot each power spectrum
    for i_ps, ps in enumerate(tps):
        ax.semilogy(t_axis_data, ps, label="Spectrum {}".format(i_ps))

    ax.legend()

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (arbitrary units)")

    pyplot.show()

def fit_tps(tps, axis):
    """
    Parameters:
        tps: The temporal power spectrum to fit
        axis: fit parallel or perpendicular to wind direction

    Returns:

    """

