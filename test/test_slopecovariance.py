import time

import numpy

import aotools

def test_slopecovmat_init():
    threads = 1

    n_wfs = 3
    telescope_diameter = 8.
    nx_subaps = 10

    n_layers = 3
    layer_altitudes = numpy.linspace(0, 20000, n_layers)
    layer_r0s = [1] * n_layers
    layer_L0s = [25.] * n_layers

    asterism_radius = 10

    subap_diameters = [telescope_diameter / nx_subaps] * n_wfs
    pupil_masks = [aotools.circle(nx_subaps / 2., nx_subaps)] * n_wfs
    gs_altitudes = [90000] * n_wfs
    gs_positions = [
        [asterism_radius, 0],
        [numpy.sin(numpy.pi / 3.) * asterism_radius, numpy.cos(numpy.pi / 3.) * asterism_radius],
        [numpy.sin(numpy.pi / 3.) * asterism_radius, -numpy.cos(numpy.pi / 3.) * asterism_radius],
        [-numpy.sin(numpy.pi / 3.) * asterism_radius, numpy.cos(numpy.pi / 3.) * asterism_radius],
        [-numpy.sin(numpy.pi / 3.) * asterism_radius, -numpy.cos(numpy.pi / 3.) * asterism_radius],
        [-asterism_radius, 0]]
    wfs_magnifications = [1.] * n_wfs
    pupil_offsets = [[0, 0]] * n_wfs
    wfs_rotations = [0] * n_wfs
    wfs_wavelengths = [550e-9] * n_wfs

    cov_mat = aotools.CovarianceMatrix(n_wfs, pupil_masks, telescope_diameter, subap_diameters, gs_altitudes, gs_positions,
                                    wfs_wavelengths,
                                    n_layers, layer_altitudes, layer_r0s, layer_L0s, threads)

def test_slopecovmat_makecovmat():
    threads = 1

    n_wfs = 3
    telescope_diameter = 8.
    nx_subaps = 10

    n_layers = 3
    layer_altitudes = numpy.linspace(0, 20000, n_layers)
    layer_r0s = [1] * n_layers
    layer_L0s = [25.] * n_layers

    asterism_radius = 10

    subap_diameters = [telescope_diameter / nx_subaps] * n_wfs
    pupil_masks = [aotools.circle(nx_subaps / 2., nx_subaps)] * n_wfs
    gs_altitudes = [90000] * n_wfs
    gs_positions = [
        [asterism_radius, 0],
        [numpy.sin(numpy.pi / 3.) * asterism_radius, numpy.cos(numpy.pi / 3.) * asterism_radius],
        [numpy.sin(numpy.pi / 3.) * asterism_radius, -numpy.cos(numpy.pi / 3.) * asterism_radius],
        [-numpy.sin(numpy.pi / 3.) * asterism_radius, numpy.cos(numpy.pi / 3.) * asterism_radius],
        [-numpy.sin(numpy.pi / 3.) * asterism_radius, -numpy.cos(numpy.pi / 3.) * asterism_radius],
        [-asterism_radius, 0]]
    wfs_magnifications = [1.] * n_wfs
    pupil_offsets = [[0, 0]] * n_wfs
    wfs_rotations = [0] * n_wfs
    wfs_wavelengths = [550e-9] * n_wfs

    cov_mat = aotools.CovarianceMatrix(n_wfs, pupil_masks, telescope_diameter, subap_diameters, gs_altitudes, gs_positions,
                                    wfs_wavelengths,
                                    n_layers, layer_altitudes, layer_r0s, layer_L0s, threads)

    covariance_matrix = cov_mat.make_covariance_matrix()


def test_covtomorecon():
    threads = 1

    n_wfs = 3
    telescope_diameter = 8.
    nx_subaps = 10

    n_layers = 3
    layer_altitudes = numpy.linspace(0, 20000, n_layers)
    layer_r0s = [1] * n_layers
    layer_L0s = [25.] * n_layers

    asterism_radius = 10

    subap_diameters = [telescope_diameter / nx_subaps] * n_wfs
    pupil_masks = [aotools.circle(nx_subaps / 2., nx_subaps)] * n_wfs
    gs_altitudes = [90000] * n_wfs
    gs_positions = [
        [asterism_radius, 0],
        [numpy.sin(numpy.pi / 3.) * asterism_radius, numpy.cos(numpy.pi / 3.) * asterism_radius],
        [numpy.sin(numpy.pi / 3.) * asterism_radius, -numpy.cos(numpy.pi / 3.) * asterism_radius],
        [-numpy.sin(numpy.pi / 3.) * asterism_radius, numpy.cos(numpy.pi / 3.) * asterism_radius],
        [-numpy.sin(numpy.pi / 3.) * asterism_radius, -numpy.cos(numpy.pi / 3.) * asterism_radius],
        [-asterism_radius, 0]]
    wfs_magnifications = [1.] * n_wfs
    pupil_offsets = [[0, 0]] * n_wfs
    wfs_rotations = [0] * n_wfs
    wfs_wavelengths = [550e-9] * n_wfs

    cov_mat = aotools.CovarianceMatrix(n_wfs, pupil_masks, telescope_diameter, subap_diameters, gs_altitudes, gs_positions,
                                    wfs_wavelengths,
                                    n_layers, layer_altitudes, layer_r0s, layer_L0s, threads)

    cov_mat.make_covariance_matrix()

    tomo_recon = cov_mat.make_tomographic_reconstructor()


def test_slopecovmat_makecovmat_multithreaded():
    threads = 2

    n_wfs = 3
    telescope_diameter = 8.
    nx_subaps = 10

    n_layers = 3
    layer_altitudes = numpy.linspace(0, 20000, n_layers)
    layer_r0s = [1] * n_layers
    layer_L0s = [25.] * n_layers

    asterism_radius = 10

    subap_diameters = [telescope_diameter / nx_subaps] * n_wfs
    pupil_masks = [aotools.circle(nx_subaps / 2., nx_subaps)] * n_wfs
    gs_altitudes = [90000] * n_wfs
    gs_positions = [
        [asterism_radius, 0],
        [numpy.sin(numpy.pi / 3.) * asterism_radius, numpy.cos(numpy.pi / 3.) * asterism_radius],
        [numpy.sin(numpy.pi / 3.) * asterism_radius, -numpy.cos(numpy.pi / 3.) * asterism_radius],
        [-numpy.sin(numpy.pi / 3.) * asterism_radius, numpy.cos(numpy.pi / 3.) * asterism_radius],
        [-numpy.sin(numpy.pi / 3.) * asterism_radius, -numpy.cos(numpy.pi / 3.) * asterism_radius],
        [-asterism_radius, 0]]
    wfs_magnifications = [1.] * n_wfs
    pupil_offsets = [[0, 0]] * n_wfs
    wfs_rotations = [0] * n_wfs
    wfs_wavelengths = [550e-9] * n_wfs

    cov_mat = aotools.CovarianceMatrix(n_wfs, pupil_masks, telescope_diameter, subap_diameters, gs_altitudes, gs_positions,
                                    wfs_wavelengths,
                                    n_layers, layer_altitudes, layer_r0s, layer_L0s, threads)
    covariance_matrix_multithread = cov_mat.make_covariance_matrix()

    # Check consistant with single thread implementation

    cov_mat.threads = 1
    covariance_matrix = cov_mat.make_covariance_matrix()

    assert numpy.array_equal(covariance_matrix, covariance_matrix_multithread)


if __name__ == "__main__":
    N = 1
    threads = 2

    n_wfs = 6
    telescope_diameter = 8.
    nx_subaps = 10

    n_layers = 3
    layer_altitudes = numpy.linspace(0, 20000, n_layers)
    layer_r0s = [1] * n_layers
    layer_L0s = [25.] * n_layers

    asterism_radius = 10

    subap_diameters = [telescope_diameter / nx_subaps] * n_wfs
    pupil_masks = [aotools.circle(nx_subaps / 2., nx_subaps)] * n_wfs
    gs_altitudes = [90000] * n_wfs
    gs_positions = [
        [asterism_radius, 0],
        [numpy.sin(numpy.pi / 3.) * asterism_radius, numpy.cos(numpy.pi / 3.) * asterism_radius],
        [numpy.sin(numpy.pi / 3.) * asterism_radius, -numpy.cos(numpy.pi / 3.) * asterism_radius],
        [-numpy.sin(numpy.pi / 3.) * asterism_radius, numpy.cos(numpy.pi / 3.) * asterism_radius],
        [-numpy.sin(numpy.pi / 3.) * asterism_radius, -numpy.cos(numpy.pi / 3.) * asterism_radius],
        [-asterism_radius, 0]]
    wfs_magnifications = [1.] * n_wfs
    pupil_offsets = [[0, 0]] * n_wfs
    wfs_rotations = [0] * n_wfs
    wfs_wavelengths = [550e-9] * n_wfs

    t1 = time.time()
    for i in range(N):
        print("\nIteration {}".format(i))
        cov_mat = aotools.CovarianceMatrix(n_wfs, pupil_masks, telescope_diameter, subap_diameters, gs_altitudes, gs_positions,
                                    wfs_wavelengths,
                                    n_layers, layer_altitudes, layer_r0s, layer_L0s, threads)
        cov_mat.make_covariance_matrix()

    t2 = time.time()

    time_taken = (t2 - t1) / N
    covmat_per_sec = 1. / time_taken
    print("Time for 1 Covariance Matrix: {}s".format(time_taken))
    print("Covariance Matrics per second: {} cps".format(covmat_per_sec))

    from matplotlib import pyplot
    pyplot.imshow(cov_mat.covariance_matrix)
    pyplot.show()