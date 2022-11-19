__all__ = ['make_mchi_model', 'make_mchi_aligned_model', 'make_ftau_model']

import aesara.tensor as at
import aesara.tensor.slinalg as atl
import numpy as np
import pymc as pm
from scipy.special import binom

# reference frequency and mass values to translate linearly between the two
FREF = 2985.668287014743
MREF = 68.0

def rd(ts, f, gamma, Apx, Apy, Acx, Acy, Fp, Fc):
    """Generate a ringdown waveform as it appears in a detector.
    
    Arguments
    ---------
    
    ts : array_like
        The times at which the ringdown waveform should be evaluated.  
    f : real
        The frequency.
    gamma : real
        The damping rate.
    Apx : real
        The amplitude of the "plus" cosine-like quadrature.
    Apy : real
        The amplitude of the "plus" sine-like quadrature.
    Acx : real
        The amplitude of the "cross" cosine-like quadrature.
    Acy : real
        The amplitude of the "cross" sine-like quadrature.
    Fp : real
        The coefficient of the "plus" polarization in the detector.
    Fc : real
        The coefficient of the "cross" term in the detector.

    Returns
    -------

    Array of the ringdown waveform in the detector.
    """
    ct = at.cos(2*np.pi*f*ts)
    st = at.sin(2*np.pi*f*ts)
    decay = at.exp(-gamma*ts)
    p = decay*(Apx*ct + Apy*st)
    c = decay*(Acx*ct + Acy*st)
    return Fp*p + Fc*c

def chi_factors(chi, coeffs):
    log1mc = at.log1p(-chi)
    log1mc2 = log1mc*log1mc
    log1mc3 = log1mc2*log1mc
    log1mc4 = log1mc2*log1mc2
    v = at.stack([chi, at.as_tensor_variable(1.0), log1mc, log1mc2,
                  log1mc3, log1mc4])
    return at.dot(coeffs, v)

def get_snr(h, d, L):
    wh = atl.solve_lower_triangular(L, h)
    wd = atl.solve_lower_triangular(L, h)
    return at.dot(wh, wd) / at.sqrt(at.dot(wh, wh))

def compute_h_det_mode(t0s, ts, Fps, Fcs, fs, gammas, Apxs, Apys, Acxs, Acys):
    ndet = len(t0s)
    nmode = fs.shape[0]
    nsamp = ts[0].shape[0]

    t0s = at.as_tensor_variable(t0s).reshape((ndet, 1, 1))
    ts = at.as_tensor_variable(ts).reshape((ndet, 1, nsamp))
    Fps = at.as_tensor_variable(Fps).reshape((ndet, 1, 1))
    Fcs = at.as_tensor_variable(Fcs).reshape((ndet, 1, 1))
    fs = at.as_tensor_variable(fs).reshape((1, nmode, 1))
    gammas = at.as_tensor_variable(gammas).reshape((1, nmode, 1))
    Apxs = at.as_tensor_variable(Apxs).reshape((1, nmode, 1))
    Apys = at.as_tensor_variable(Apys).reshape((1, nmode, 1))
    Acxs = at.as_tensor_variable(Acxs).reshape((1, nmode, 1))
    Acys = at.as_tensor_variable(Acys).reshape((1, nmode, 1))

    return rd(ts - t0s, fs, gammas, Apxs, Apys, Acxs, Acys, Fps, Fcs)

def a_from_quadratures(Apx, Apy, Acx, Acy):
    A = 0.5*(at.sqrt(at.square(Acy + Apx) + at.square(Acx - Apy)) +
             at.sqrt(at.square(Acy - Apx) + at.square(Acx + Apy)))
    return A

def ellip_from_quadratures(Apx, Apy, Acx, Acy):
    A = a_from_quadratures(Apx, Apy, Acx, Acy)
    e = 0.5*(at.sqrt(at.square(Acy + Apx) + at.square(Acx - Apy)) -
             at.sqrt(at.square(Acy - Apx) + at.square(Acx + Apy))) / A
    return e

def Aellip_from_quadratures(Apx, Apy, Acx, Acy):
    # should be slightly cheaper than calling the two functions separately
    term1 = at.sqrt(at.square(Acy + Apx) + at.square(Acx - Apy))
    term2 = at.sqrt(at.square(Acy - Apx) + at.square(Acx + Apy))
    A = 0.5*(term1 + term2)
    e = 0.5*(term1 - term2) / A
    return A, e

def phiR_from_quadratures(Apx, Apy, Acx, Acy):
    return at.arctan2(-Acx + Apy, Acy + Apx)

def phiL_from_quadratures(Apx, Apy, Acx, Acy):
    return at.arctan2(-Acx - Apy, -Acy + Apx)

def flat_A_quadratures_prior(Apx_unit, Apy_unit, Acx_unit, Acy_unit):
    return 0.5*at.sum(at.square(Apx_unit) + at.square(Apy_unit) +
                      at.square(Acx_unit) + at.square(Acy_unit))

def spin_weighted_spherical_harmonic(s, l, m, theta):
    """Compute the spin weighted spherical harmonics, of a given spin weight,
    evaluated at certain polar angle, fixing the azimuthal angle to phi=0.

    Arguments
    ---------
    s : int
        spin weight (-2 for GWs)
    l : int
        total angular momentum number
    m : int
        azimuthal number, must have `|m| <= l`
    theta : float
        polar angle

    Returns
    -------
    swsh : float
        SWSH coefficient
    """
    prefac2 = (np.math.factorial(l + m)*np.math.factorial(l - m)*(2*l + 1))/\
              (np.math.factorial(l + s)*np.math.factorial(l - s)*4*np.pi)
    sin_th_2 = at.sin(theta/2)
    cot_th_2 = at.cos(theta/2) / sin_th_2
    itsum = 0.0
    for r in range(l - s + 1):
      itsum += binom(l-s, r)*binom(l+s, r+s-m)*(-1)**(l-r-s)*cot_th_2**(2*r+s-m)
    return (-1)**m * np.sqrt(prefac2) * sin_th_2**(2*l) * itsum 

def symmetric_swsh_pc(s, l, m, theta):
    """Effective plus and cross SWSH factors assuming equatorial symmetry.

    Returns
    -------
    Ylm_plus : float
        prefactor for the plus polarization
    Ylm_cross : float
        prefactor for the cross polarization
    """
    Ylm_plus_m = spin_weighted_spherical_harmonic(s, l, m, theta)
    Ylm_minus_m = spin_weighted_spherical_harmonic(s, l, m, np.pi - theta)
    # TODO: double check missing (-1)**s
    Ylm_plus = Ylm_plus_m + Ylm_minus_m
    Ylm_cross = Ylm_plus_m - Ylm_minus_m
    return Ylm_plus, Ylm_cross
    

def make_mchi_model(t0, times, strains, Ls, Fps, Fcs, f_coeffs, g_coeffs,
                    **kwargs):
    M_min = kwargs.pop("M_min")
    M_max = kwargs.pop("M_max")
    chi_min = kwargs.pop("chi_min")
    chi_max = kwargs.pop("chi_max")
    A_scale = kwargs.pop("A_scale")
    df_max = kwargs.pop("df_max")
    dtau_max = kwargs.pop("dtau_max")
    perturb_f = kwargs.pop("perturb_f", 0)
    perturb_tau = kwargs.pop("perturb_tau", 0)
    flat_A = kwargs.pop("flat_A", True)
    flat_A_ellip = kwargs.pop("flat_A_ellip", False)

    if flat_A and flat_A_ellip:
        raise ValueError("at most one of `flat_A` and `flat_A_ellip` can be "
                         "`True`")
    if (chi_min < 0) or (chi_max > 1):
        raise ValueError("chi boundaries must be contained in [0, 1)")

    ndet = len(t0)
    nt = len(times[0])
    nmode = f_coeffs.shape[0]

    ifos = kwargs.pop('ifos', np.arange(ndet))
    modes = kwargs.pop('modes', np.arange(nmode))

    coords = {
        'ifo': ifos,
        'mode': modes,
        'time_index': np.arange(nt)
    }

    with pm.Model(coords=coords) as model:
        pm.ConstantData('times', times, dims=['ifo', 'time_index'])
        pm.ConstantData('t0', t0, dims=['ifo'])
        pm.ConstantData('L', Ls, dims=['ifo', 'time_index', 'time_index'])

        M = pm.Uniform("M", M_min, M_max)
        chi = pm.Uniform("chi", chi_min, chi_max)

        Apx_unit = pm.Normal("Apx_unit", dims=['mode'])
        Apy_unit = pm.Normal("Apy_unit", dims=['mode'])
        Acx_unit = pm.Normal("Acx_unit", dims=['mode'])
        Acy_unit = pm.Normal("Acy_unit", dims=['mode'])

        df = pm.Uniform("df", -df_max, df_max, dims=['mode'])
        dtau = pm.Uniform("dtau", -dtau_max, dtau_max, dims=['mode'])

        Apx = pm.Deterministic("Apx", A_scale*Apx_unit, dims=['mode'])
        Apy = pm.Deterministic("Apy", A_scale*Apy_unit, dims=['mode'])
        Acx = pm.Deterministic("Acx", A_scale*Acx_unit, dims=['mode'])
        Acy = pm.Deterministic("Acy", A_scale*Acy_unit, dims=['mode'])

        A = pm.Deterministic("A", a_from_quadratures(Apx, Apy, Acx, Acy),
                             dims=['mode'])
        ellip = pm.Deterministic("ellip",
            ellip_from_quadratures(Apx, Apy, Acx, Acy),
            dims=['mode'])

        f0 = FREF*MREF/M
        f = pm.Deterministic("f",
            f0*chi_factors(chi, f_coeffs)*at.exp(df*perturb_f),
            dims=['mode'])
        gamma = pm.Deterministic("gamma",
            f0*chi_factors(chi, g_coeffs)*at.exp(-dtau*perturb_tau),
            dims=['mode'])
        tau = pm.Deterministic("tau", 1/gamma, dims=['mode'])
        Q = pm.Deterministic("Q", np.pi * f * tau, dims=['mode'])
        phiR = pm.Deterministic("phiR",
             phiR_from_quadratures(Apx, Apy, Acx, Acy),
             dims=['mode'])
        phiL = pm.Deterministic("phiL",
             phiL_from_quadratures(Apx, Apy, Acx, Acy),
             dims=['mode'])
        theta = pm.Deterministic("theta", -0.5*(phiR + phiL), dims=['mode'])
        phi = pm.Deterministic("phi", 0.5*(phiR - phiL), dims=['mode'])

        h_det_mode = pm.Deterministic("h_det_mode",
                compute_h_det_mode(t0, times, Fps, Fcs, f, gamma,
                                   Apx, Apy, Acx, Acy),
                dims=['ifo', 'mode', 'time_index'])
        h_det = pm.Deterministic("h_det", at.sum(h_det_mode, axis=1),
                                 dims=['ifo', 'time_index'])

        # Priors:

        # Flat in M-chi already

        # Amplitude prior
        if flat_A:
            # bring us back to flat-in-quadratures
            pm.Potential("flat_A_quadratures_prior",
                         flat_A_quadratures_prior(Apx_unit, Apy_unit,
                                                  Acx_unit, Acy_unit))
            # bring us to flat-in-A prior
            pm.Potential("flat_A_prior", -3*at.sum(at.log(A)))
        elif flat_A_ellip:
            # bring us back to flat-in-quadratures
            pm.Potential("flat_A_quadratures_prior",
                         flat_A_quadratures_prior(Apx_unit, Apy_unit,
                                                  Acx_unit, Acy_unit))
            # bring us to flat-in-A and flat-in-ellip prior
            pm.Potential("flat_A_ellip_prior", 
                         at.sum(-3*at.log(A) - at.log1m(at.square(ellip))))

        # Flat prior on the delta-fs and delta-taus

        # Likelihood:
        for i in range(ndet):
            key = ifos[i]
            if isinstance(key, bytes):
                # Don't want byte strings in our names!
                key = key.decode('utf-8')
            _ = pm.MvNormal(f"strain_{key}", mu=h_det[i,:], chol=Ls[i],
                            observed=strains[i], dims=['time_index'])
        
        return model
        
def make_mchi_aligned_model(t0, times, strains, Ls, Fps, Fcs, f_coeffs,
                            g_coeffs, **kwargs):
    M_min = kwargs.pop("M_min")
    M_max = kwargs.pop("M_max")
    chi_min = kwargs.pop("chi_min")
    chi_max = kwargs.pop("chi_max")
    cosi_min = kwargs.pop("cosi_min")
    cosi_max = kwargs.pop("cosi_max")
    A_scale = kwargs.pop("A_scale")
    df_max = kwargs.pop("df_max")
    dtau_max = kwargs.pop("dtau_max")
    perturb_f = kwargs.pop("perturb_f", 0)
    perturb_tau = kwargs.pop("perturb_tau", 0)
    flat_A = kwargs.pop("flat_A", True)

    if (cosi_min < -1) or (cosi_max > 1):
        raise ValueError("cosi boundaries must be contained in [-1, 1]")
    if (chi_min < 0) or (chi_max > 1):
        raise ValueError("chi boundaries must be contained in [0, 1)")
    
    ndet = len(t0)
    nt = len(times[0])
    nmode = f_coeffs.shape[0]

    ifos = kwargs.pop('ifos', np.arange(ndet))
    modes = kwargs.pop('modes')
    lms = [(int(l), int(m))for (p,l,m,n) in [x.decode('ascii') for x in modes]]

    coords = {
        'ifo': ifos,
        'mode': modes,
        'time_index': np.arange(nt)
    }

    with pm.Model(coords=coords) as model:
        pm.ConstantData('times', times, dims=['ifo', 'time_index'])
        pm.ConstantData('t0', t0, dims=['ifo'])
        pm.ConstantData('L', Ls, dims=['ifo', 'time_index', 'time_index'])

        M = pm.Uniform("M", M_min, M_max)
        chi = pm.Uniform("chi", chi_min, chi_max)

        cosi = pm.Uniform("cosi", cosi_min, cosi_max)
        iota = at.arccos(cosi)

        Ylm_plus_m = at.as_tensor_variable([spin_weighted_spherical_harmonic(-2, l, m, iota) for (l,m) in lms], ndim=1)
        Ylm_minus_m = at.as_tensor_variable([spin_weighted_spherical_harmonic(-2, l, m, np.pi-iota) for (l,m) in lms], ndim=1)
        Ylm_plus = Ylm_plus_m + Ylm_minus_m
        Ylm_cross = Ylm_plus_m - Ylm_minus_m

        Ax_unit = pm.Normal("Ax_unit", dims=['mode'])
        Ay_unit = pm.Normal("Ay_unit", dims=['mode'])

        df = pm.Uniform("df", -df_max, df_max, dims=['mode'])
        dtau = pm.Uniform("dtau", -dtau_max, dtau_max, dims=['mode'])

        A = pm.Deterministic("A",
            A_scale*at.sqrt(at.square(Ax_unit)+at.square(Ay_unit)),
            dims=['mode'])
        phi = pm.Deterministic("phi", at.arctan2(Ay_unit, Ax_unit),
            dims=['mode'])

        f0 = FREF*MREF/M
        f = pm.Deterministic('f',
            f0*chi_factors(chi, f_coeffs)*at.exp(df * perturb_f),
            dims=['mode'])
        gamma = pm.Deterministic('gamma',
             f0*chi_factors(chi, g_coeffs)*at.exp(-dtau * perturb_tau),
             dims=['mode'])
        tau = pm.Deterministic('tau', 1/gamma, dims=['mode'])
        Q = pm.Deterministic('Q', np.pi*f*tau, dims=['mode'])
        Ap = pm.Deterministic('Ap', Ylm_plus*A, dims=['mode'])
        Ac = pm.Deterministic('Ac', Ylm_cross*A, dims=['mode'])
        ellip = pm.Deterministic('ellip', Ac/Ap, dims=['mode'])

        Apx = Ylm_plus*A*at.cos(phi)
        Apy = Ylm_plus*A*at.sin(phi)
        Acx = -Ylm_cross*at.sin(phi)
        Acy = Ylm_cross*A*at.cos(phi)

        h_det_mode = pm.Deterministic("h_det_mode",
            compute_h_det_mode(t0, times, Fps, Fcs, f, gamma,
                               Apx, Apy, Acx, Acy),
            dims=['ifo', 'mode', 'time_index'])
        h_det = pm.Deterministic("h_det", at.sum(h_det_mode, axis=1),
                                 dims=['ifo', 'time_index'])

        # Priors:

        # Flat in M-chi already

        # Amplitude prior
        if flat_A:
            # first bring us to flat in quadratures
            pm.Potential("flat_A_quadratures_prior",
                         0.5*at.sum(at.square(Ax_unit) + at.square(Ay_unit)))
            # now to flat in A
            pm.Potential("flat_A_prior", -at.sum(at.log(A)))

        # Flat prior on the delta-fs and delta-taus

        # Likelihood
        for i in range(ndet):
            key = ifos[i]
            if isinstance(key, bytes):
                # Don't want byte strings in our names!
                key = key.decode('utf-8')
            _ = pm.MvNormal(f"strain_{key}", mu=h_det[i,:], chol=Ls[i],
                            observed=strains[i], dims=['time_index'])
        
        return model

def logit(p):
    return np.log(p) - np.log1p(-p)

def make_ftau_model(t0, times, strains, Ls, **kwargs):
    f_min = kwargs.pop("f_min")
    f_max = kwargs.pop("f_max")
    gamma_min = kwargs.pop("gamma_min")
    gamma_max = kwargs.pop("gamma_max")
    A_scale = kwargs.pop("A_scale")
    flat_A = kwargs.pop("flat_A", True)
    nmode = kwargs.pop("nmode", 1)

    ndet = len(t0)
    nt = len(times[0])

    ifos = kwargs.pop('ifos', np.arange(ndet))
    modes = kwargs.pop('modes', np.arange(nmode))

    coords = {
        'ifo': ifos,
        'mode': modes,
        'time_index': np.arange(nt)
    }

    with pm.Model(coords=coords) as model:
        pm.ConstantData('times', times, dims=['ifo', 'time_index'])
        pm.ConstantData('t0', t0, dims=['ifo'])
        pm.ConstantData('L', Ls, dims=['ifo', 'time_index', 'time_index'])

        f = pm.Uniform("f", f_min, f_max, dims=['mode'])
        gamma = pm.Uniform('gamma', gamma_min, gamma_max, dims=['mode'],
                           transform=pm.distributions.transforms.ordered)

        Ax_unit = pm.Normal("Ax_unit", dims=['mode'])
        Ay_unit = pm.Normal("Ay_unit", dims=['mode'])

        A = pm.Deterministic("A",
            A_scale*at.sqrt(at.square(Ax_unit)+at.square(Ay_unit)),
            dims=['mode'])
        phi = pm.Deterministic("phi", at.arctan2(Ay_unit, Ax_unit),
                               dims=['mode'])

        tau = pm.Deterministic('tau', 1/gamma, dims=['mode'])
        Q = pm.Deterministic('Q', np.pi*f*tau, dims=['mode'])

        Apx = A*at.cos(phi)
        Apy = A*at.sin(phi)

        h_det_mode = pm.Deterministic("h_det_mode",
            compute_h_det_mode(t0, times, np.ones(ndet), np.zeros(ndet),
                               f, gamma, Apx, Apy, np.zeros(nmode),
                               np.zeros(nmode)),
            dims=['ifo', 'mode', 'time_index'])
        h_det = pm.Deterministic("h_det", at.sum(h_det_mode, axis=1),
                                 dims=['ifo', 'time_index'])

        # Priors:

        # Flat in M-chi already

        # Amplitude prior
        if flat_A:
            # first bring us to flat in quadratures
            pm.Potential("flat_A_quadratures_prior",
                         0.5*at.sum(at.square(Ax_unit) + at.square(Ay_unit)))
            pm.Potential("flat_A_prior", -at.sum(at.log(A)))

        # Flat prior on the delta-fs and delta-taus

        # Likelihood
        for i in range(ndet):
            key = ifos[i]
            if isinstance(key, bytes):
                # Don't want byte strings in our names!
                key = key.decode('utf-8')
            _ = pm.MvNormal(f"strain_{key}", mu=h_det[i,:], chol=Ls[i],
                            observed=strains[i], dims=['time_index'])
        
        return model


