from aotools import turbulence
import numpy


def test_calc_slope_temporalps():
    slopes = numpy.random.random((1000, 104))
    mean_spectra, error_spectra = turbulence.calc_slope_temporalps(slopes)
    assert len(mean_spectra) == 500
    assert len(error_spectra) == 500


def test_get_tps_time_axis():
    frame_rate = 100
    n_frames = 1000
    t_values = turbulence.get_tps_time_axis(frame_rate, n_frames)
    assert len(t_values) == 500


# def test_fit_tps():
#     frames = 1000
#     frame_rate = 100
#     sub_aperture_diameter = 0.5
#     slopes = numpy.random.random((frames, 104))
#     temporal_power_spectra, error = turbulence.calc_slope_temporalps(slopes)
#     t_axis = turbulence.get_tps_time_axis(frame_rate, frames)
#     turbulence.fit_tps(temporal_power_spectra, t_axis, sub_aperture_diameter)
