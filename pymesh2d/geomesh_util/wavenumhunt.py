import numpy as np

def wavenumhunt(T, h):
    """
    Compute wavelength (L) and wavenumber (k) using Hunt (1979) approximation.

    Parameters
    ----------
    T : float or array_like
        Wave period [s]
    h : float or array_like
        Water depth [m]

    Returns
    -------
    L : float or ndarray
        Wavelength [m]
    k : float or ndarray
        Wavenumber [1/m]
    """
    D = np.array([0.6666666666, 0.3555555555, 0.1608465608,
                  0.0632098765, 0.0217540484, 0.0065407983])

    L0 = (9.81 * T**2) / (2 * np.pi)
    k0 = 2 * np.pi / L0
    k0h = k0 * h

    # Approximation of kh following Hunt (1979)
    poly = (D[0]*k0h**1 + D[1]*k0h**2 + D[2]*k0h**3 +
            D[3]*k0h**4 + D[4]*k0h**5 + D[5]*k0h**6)
    kh = k0h * np.sqrt(1 + (k0h * (1 + poly))**-1)

    k = kh / h
    L = 2 * np.pi / k

    return L, k
