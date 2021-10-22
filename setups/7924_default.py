import radvel
import numpy as np
import pandas as pd

starname = '7924'
nplanets = 4
fitting_basis = 'per tc secosw sesinw k'
bjd0 = 0.
planet_letters = {1: 'b', 2: 'c', 3: 'd', 4: 'activity'}

# Define prior centers (initial guesses) in a basis of your choice (need not be in the fitting basis)
anybasis_params = radvel.Parameters(nplanets, basis='per tc e w k', 
                                    planet_letters=planet_letters) # initialize Parameters object
anybasis_params['per1'] = radvel.Parameter(value=5.397920)
anybasis_params['tc1'] = radvel.Parameter(value=2455937.378704)
anybasis_params['e1'] = radvel.Parameter(value=0.076)
anybasis_params['w1'] = radvel.Parameter(value=0.436332)
anybasis_params['k1'] = radvel.Parameter(value=3.600000)
anybasis_params['per2'] = radvel.Parameter(value=15.299000)
anybasis_params['tc2'] = radvel.Parameter(value=2455935.334200)
anybasis_params['e2'] = radvel.Parameter(value=0)
anybasis_params['w2'] = radvel.Parameter(value=0.471239)
anybasis_params['k2'] = radvel.Parameter(value=2.310000)
anybasis_params['per3'] = radvel.Parameter(value=24.451000)
anybasis_params['tc3'] = radvel.Parameter(value=2455935.334200)
anybasis_params['e3'] = radvel.Parameter(value=0)
anybasis_params['w3'] = radvel.Parameter(value=2.076942)
anybasis_params['k3'] = radvel.Parameter(value=1.650000)
anybasis_params['per4'] = radvel.Parameter(value=2819)
anybasis_params['tc4'] = radvel.Parameter(value=2456629.0)
anybasis_params['e4'] = radvel.Parameter(value=0.29)
anybasis_params['w4'] = radvel.Parameter(value=-1.14)
anybasis_params['k4'] = radvel.Parameter(value=2.01)


time_base = 2455935.334200
anybasis_params['dvdt'] = radvel.Parameter(value=0.0)
anybasis_params['curv'] = radvel.Parameter(value=0.0)
data = pd.read_csv('/Users/bjfulton/Dropbox/radvel_targets/data/vst7924.csv', usecols=(1, 2, 3), names=['time', 'mnvel', 'errvel'],
                   header=0, skiprows=1, dtype={'time': np.float64, 'mnvel': np.float64, 'errvel': np.float64})
data['tel'] = 'j'
data['time'] += 2440000
bin_t, bin_vel, bin_err, bin_tel = radvel.utils.bintels(data['time'].values, data['mnvel'].values, data['errvel'].values, data['tel'].values, binsize=0.1)
data = pd.DataFrame([], columns=['time', 'mnvel', 'errvel', 'tel'])
data['time'] = bin_t
data['mnvel'] = bin_vel
data['errvel'] = bin_err
data['tel'] = bin_tel

instnames = ['j']
ntels = len(instnames)
anybasis_params['gamma_j'] = radvel.Parameter(value=0.0, vary=False, linear=True)
anybasis_params['jit_j'] = radvel.Parameter(value=1.0)

params = anybasis_params.basis.to_any_basis(anybasis_params,fitting_basis)
mod = radvel.RVModel(params, time_base=time_base)

mod.params['per1'].vary = True
mod.params['tc1'].vary = True
mod.params['secosw1'].vary = True
mod.params['sesinw1'].vary = True
mod.params['per2'].vary = True
mod.params['tc2'].vary = True
mod.params['secosw2'].vary = True
mod.params['sesinw2'].vary = True
mod.params['per3'].vary = True
mod.params['tc3'].vary = True
mod.params['secosw3'].vary = True
mod.params['sesinw3'].vary = True
mod.params['dvdt'].vary = False
mod.params['curv'].vary = False
mod.params['jit_j'].vary = True

priors = [
          radvel.prior.EccentricityPrior(nplanets),
          radvel.prior.PositiveKPrior(nplanets),
          radvel.prior.Gaussian('per1', 5.397920, 0.00025),
          radvel.prior.Gaussian('per2', 15.299000, 0.0033),
          radvel.prior.Gaussian('per3', 24.451000, 0.017),

         ]

stellar = dict(mstar=0.7959, mstar_err=0.0317)


