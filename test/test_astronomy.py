from aotools import astronomy, circle


def test_photons_per_mag():
    mask = circle(2, 5)
    photons = astronomy.photons_per_mag(5.56, mask, 0.5, 0.3, 10)
    assert type(photons) == float


def test_flux_to_magnitude():
    magnitude = astronomy.flux_to_magnitude(52504716., 'V')
    assert type(magnitude) == float


def test_magnitude_to_flux():
    flux = astronomy.magnitude_to_flux(5.56, 'V')
    assert type(flux) == float


def test_magnitude_to_flux_and_flux_to_magnitude():
    flux = astronomy.magnitude_to_flux(5.56, 'V')
    magnitude = astronomy.flux_to_magnitude(flux, 'V')
    assert magnitude == 5.56
