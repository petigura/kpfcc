#!/usr/bin/env python

import sys
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from astropy import constants as c

np.random.seed(2)

import radvel
from radvel.plot import orbit_plots, mcmc_plots

def arguments():
    psr = ArgumentParser(
        description="Simulate observations of a planet given a set of observation timestamps.",
    )

    psr.add_argument(metavar='schedule_path', dest='schedule_path',
                    action='store', default=None, type=str,
                    help="Path to observation timestamp csv")
    psr.add_argument(metavar='radvel_setup', dest='radvel_setup',
                    action='store', default=None, type=str,
                    help="Path to radvel setup file describing the simulated planetary system.")
    psr.add_argument(metavar='nobs', dest='nobs',
                    action='store', default=None, type=int,
                    help="Number of observations.")

    args = psr.parse_args()

    return args

def load_module_from_file(module_name, module_path):
    """Loads a python module from the path of the corresponding file.

    Args:
        module_name (str): namespace where the python module will be loaded,
            e.g. ``foo.bar``
        module_path (str): path of the python file containing the module
    Returns:
        A valid module object
    Raises:
        ImportError: when the module can't be loaded
        FileNotFoundError: when module_path doesn't exist
    """
    if sys.version_info[0] == 3 and sys.version_info[1] >= 5:
        import importlib.util
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    elif sys.version_info[0] == 3 and sys.version_info[1] < 5:
        import importlib.machinery
        loader = importlib.machinery.SourceFileLoader(module_name, module_path)
        module = loader.load_module()

    return module

def synthesize_data(args, P, mod):
    times = np.sort(np.loadtxt(args.schedule_path))[0:args.nobs]

    rvs = mod(times) + np.random.normal(loc=0, scale=P.params['jit'].value, size=len(times))
    errvel = np.zeros_like(rvs) + P.params['jit'].value

    data = pd.DataFrame([])
    data['time'] = times
    data['mnvel'] = rvs
    data['errvel'] = errvel
    P.data = data

    return P

def create_timeseries(args):
    P = load_module_from_file('planet', args.radvel_setup)

    mod = radvel.RVModel(P.params, time_base=2459380)
    P = synthesize_data(args, P, mod)
    P.starname = "{:s}_{:s}_{:.2f}_{:03d}".format(os.path.basename(args.schedule_path.replace('.csv', '')),
                                              os.path.basename(args.radvel_setup.replace('.py', '')),
                                              P.params['k1'].value, args.nobs)
    like = radvel.likelihood.RVLikelihood(mod, P.data.time, 
                                          P.data.mnvel, P.data.errvel)

    post = radvel.posterior.Posterior(like)
    post.priors = P.priors

    post.outdir = P.starname

    system_name = P.starname
    outdir = os.path.join('./', system_name)
    args.outputdir = outdir

    if not os.path.isdir(args.outputdir):
        os.mkdir(args.outputdir)

    return P, post

def fit_timeseries(post):
    post.chains = radvel.mcmc(post, nwalkers=20, nrun=3000, ensembles=6, headless=True)
    print(post)

    return post
    
def get_msigma(P, post):
    # Convert chains into synth basis
    chains = post.chains
    mstar = np.zeros(len(chains)) + P.stellar['mstar']
    args.outputdir = P.starname
    conf_base = os.path.basename(P.starname)

    synthchains = post.chains.copy()
    for par in post.params.keys():
        if not post.params[par].vary:
            synthchains[par] = post.params[par].value

    synthchains = post.params.basis.to_synth(synthchains)

    outcols = []
    for i in np.arange(1, P.nplanets + 1, 1):
        # Grab parameters from the chain
        def _has_col(key):
            cols = list(synthchains.columns)
            return cols.count('{}{}'.format(key, i)) == 1

        def _get_param(key):
            if _has_col(key):
                return synthchains['{}{}'.format(key, i)]
            else:
                return P.params['{}{}'.format(key, i)].value

        def _set_param(key, value):
            chains['{}{}'.format(key, i)] = value

        def _get_colname(key):
            return '{}{}'.format(key, i)

        per = _get_param('per')
        k = _get_param('k')
        e = _get_param('e')

        mpsini = radvel.utils.Msini(k, per, mstar, e, Msini_units='earth')
        _set_param('mpsini', mpsini)
        outcols.append(_get_colname('mpsini'))

        mtotal = mstar + (mpsini * c.M_earth.value) / c.M_sun.value      # get total star plus planet mass
        a = radvel.utils.semi_major_axis(per, mtotal)               # changed from mstar to mtotal
        
        _set_param('a', a)
        outcols.append(_get_colname('a'))

        musini = (mpsini * c.M_earth) / (mstar * c.M_sun)
        _set_param('musini', musini)
        outcols.append(_get_colname('musini'))

    # Get quantiles and update posterior object to median
    # values returned by MCMC chains
    quantiles = chains.quantile([0.159, 0.5, 0.841])
    csvfn = os.path.join(args.outputdir, conf_base+'_derived_quantiles.csv')
    quantiles.to_csv(csvfn, columns=outcols)

    sigmas = quantiles.loc[0.5] / (quantiles.loc[0.5] - quantiles.loc[0.159])

    # saved derived paramters to posterior file
    postfile = os.path.join(args.outputdir,
                            '{}_post_obj.pkl'.format(conf_base))
    post.derived = quantiles[outcols]
    post.writeto(postfile)

    csvfn = os.path.join(args.outputdir, conf_base+'_derived.csv.bz2')
    chains.to_csv(csvfn, columns=outcols, compression='bz2')

    print("Derived parameters:", outcols)
    return sigmas

def write_msigma(args, sigmas):
    outfile = './sigmas_{}.csv'.format(os.path.basename(args.radvel_setup).replace('.py', ''))
    if not os.path.exists(outfile):
        f = open(outfile, 'w')
        print("run,nobs,baseline,sigma_k,sigma_mp", file=f)
    else:
        f = open(outfile, 'a')

    run = args.outputdir
    print("{},{:d},{:0.2f},{:0.4f},{:0.4f}".format(run, args.nobs, args.baseline, 
                                                sigmas['k1'], sigmas['mpsini1']), file=f)
    f.close()

def orbit_plot(args, post, P):
    run = args.outputdir
    saveto = args.outputdir + "/{}.pdf".format(args.outputdir)
    conf_base = os.path.basename(P.starname)

    RVPlot = orbit_plots.MultipanelPlot(post, saveplot=saveto)
    setattr(RVPlot, 'status', dict())
    RVPlot.status['derive'] = {}
    RVPlot.status['derive']['chainfile'] = os.path.join(args.outputdir, conf_base+'_derived.csv.bz2')
    RVPlot.plot_multipanel()

if __name__ == '__main__':
    args = arguments()
    P, post = create_timeseries(args)
    chains = fit_timeseries(post)
    sigmas = get_msigma(P, post)
    args.baseline = P.data['time'].max() - P.data['time'].min()
    write_msigma(args, sigmas)
    orbit_plot(args, post, P)
