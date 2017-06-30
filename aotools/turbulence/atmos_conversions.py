import numpy

def cn2_to_seeing(cn2,lamda=500.E-9):
    """
    Calculates the seeing angle from the integrated Cn2 value

    Parameters:
        cn2 (float): integrated Cn2 value in m^2/3
        lamda : wavelength

    Returns:
        seeing angle in arcseconds
    """
    r0 = cn2_to_r0(cn2,lamda)
    seeing = r0_to_seeing(r0,lamda)
    return seeing

def cn2_to_r0(cn2,lamda=500.E-9):
    """
    Calculates r0 from the integrated Cn2 value

    Parameters:
        cn2 (float): integrated Cn2 value in m^2/3
        lamda : wavelength

    Returns:
        r0 in cm
    """
    r0=(0.423*(2*numpy.pi/lamda)**2*cn2)**(-3./5.)
    return r0

def r0_to_seeing(r0,lamda=500.E-9):
    """
    Calculates the seeing angle from r0

    Parameters:
        r0 (float): Freid's parameter in cm
        lamda : wavelength

    Returns:
        seeing angle in arcseconds
    """
    return (0.98*lamda/r0)*180.*3600./numpy.pi

def coherenceTime(cn2,v,lamda=500.E-9):
    """
    Calculates the coherence time from profiles of the Cn2 and wind velocity

    Parameters:
        cn2 (array): Cn2 profile in m^2/3
        v (array): profile of wind velocity, same altitude scale as cn2 
        lamda : wavelength

    Returns:
        coherence time in seconds
    """
    Jv = (cn2*(v**(5./3.))).sum()
    tau0 = float((Jv**(-3./5.))*0.057*lamda**(6./5.))
    return tau0

def isoplanaticAngle(cn2,h,lamda=500.E-9):
    """
    Calculates the isoplanatic angle from the Cn2 profile

    Parameters:
        cn2 (array): Cn2 profile in m^2/3
        h (Array): Altitude levels of cn2 profile in m
        lamda : wavelength

    Returns:
        isoplanatic angle in arcseconds
    """
    Jh = (cn2*(h**(5./3.))).sum()
    iso = float(0.057*lamda**(6./5.)*Jh**(-3./5.)  *180.*3600./numpy.pi)
    return iso