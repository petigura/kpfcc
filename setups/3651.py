#!/usr/bin/env python3

# Required packages for setup
import pandas as pd
import numpy as np
import radvel

# overwrite after load
starname = ''

# Define global planetary system and dataset parameters
nplanets = 1    # number of planets in the system
instnames = ['kpf']  # 1: HIRES (pre 2004), 2: HIRES (post-2004), 3: CORALIE, 4: Hamilton
ntels = len(instnames)       # number of instruments with unique velocity zero-points
fitting_basis = 'per tc secosw sesinw k'    # Fitting basis, see radvel.basis.BASIS_NAMES for available basis names
bjd0 = 2440000.
planet_letters = {1:'b'}

# Define prior centers (initial guesses) in a basis of your choice (need not be in the fitting basis)
anybasis_params = radvel.Parameters(nplanets,basis='per tc e w k')    # initialize Parameters object

anybasis_params['per1'] = radvel.Parameter(value=62.218000, vary=False)
anybasis_params['tc1'] = radvel.Parameter(value=2456643.835764, vary=False)
anybasis_params['e1'] = radvel.Parameter(value=0.596000)
anybasis_params['w1'] = radvel.Parameter(value=4.232423)
anybasis_params['k1'] = radvel.Parameter(value=15.900000)

anybasis_params['dvdt'] = radvel.Parameter(value=0.0, vary=False)        # slope
anybasis_params['curv'] = radvel.Parameter(value=0.0, vary=False)         # curvature

anybasis_params['gamma'] = radvel.Parameter(0.0, vary=True)
anybasis_params['jit'] = radvel.Parameter(value=0.3, vary=True)

# Convert input orbital parameters into the fitting basis
params = anybasis_params.basis.to_any_basis(anybasis_params,fitting_basis)

# Define GP hyperparameters as Parameter objects.
gp_per_mean = 20.64 # T_bar in Dai et al. (2017) [days]
gp_per_unc = 1.0
gp_explength_mean = 9.5*np.sqrt(2.) # sqrt(2)*tau in Dai et al. (2017) [days]
gp_explength_unc = 1.0*np.sqrt(2.)
params['gp_amp'] = radvel.Parameter(value=3.0)
params['gp_explength'] = radvel.Parameter(value=gp_explength_mean)
params['gp_per'] = radvel.Parameter(value=gp_per_mean)
params['gp_perlength'] = radvel.Parameter(value=0.5, vary=False)

# Define prior shapes and widths here.
priors = [radvel.prior.EccentricityPrior(nplanets),
          radvel.prior.Jeffreys('gp_amp', 0.01, 100.),
          radvel.prior.Gaussian('gp_explength', gp_explength_mean, gp_explength_unc),
          radvel.prior.Gaussian('gp_per', gp_per_mean, gp_per_unc),
          radvel.prior.HardBounds('gp_per', 5.0, 60.0)]

# abscissa for slope and curvature terms (should be near mid-point of time baseline)
# time_base = np.mean([np.min(data.time), np.max(data.time)])  

# optional argument that can contain stellar mass in solar units (mstar) and
# uncertainty (mstar_err). If not set, mstar will be set to nan.
stellar = dict(mstar=0.8837, mstar_err=0.0382)

k_injected = params['k1'].value
mpsini_injected = radvel.utils.Msini(params['k1'].value, params['per1'].value, stellar['mstar'],
                                    anybasis_params['e1'].value, Msini_units='earth')


