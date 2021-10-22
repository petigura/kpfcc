#!/usr/bin/env python

import sys
import os
from argparse import ArgumentParser
import copy

import numpy as np
from numpy.random import RandomState
from scipy.signal import periodogram
import pandas as pd
import pylab as pl
from astropy import constants as c
from radvel.likelihood import GPLikelihood
from radvel.model import RVModel, Parameter, Parameters

np.random.seed(1234)
random = RandomState(1234)

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
    psr.add_argument('--gp', dest='gp',
                    action='store_true', default=False,
                    help="Include and model correlated noise in simulated RVs?")
    psr.add_argument('--noiseplot', dest='noise',
                    action='store_true', default=False,
                    help="Include plot of correlated noise sources (GPs).")

    args = psr.parse_args()

    return args

def plot_gp_like(like, ci):
    """
    Plot a single Gaussian Process Likleihood object in the current Axes, 
    including Gaussian Process uncertainty bands.

    Args:
        like (radvel.GPLikelihood): radvel.GPLikelihood object. The model
            plotted will be generated from `like.params`.
        orbit_model4data (numpy array): 
        ci (int): index to use when choosing a color to plot from 
            radvel.plot.default_colors. This is only used if the
            Likelihood object being plotted is not in the list of defaults.
            Increments by 1 if it is used.

    Returns: current (possibly changed) value of the input `ci`

    """
    ax = pl.gca()

    if isinstance(like, radvel.likelihood.GPLikelihood):

        xpred = np.linspace(np.min(like.x), np.max(like.x), num=int(3e3))
        gpmu, stddev = like.predict(xpred)

        gp_orbit_model = like.model(xpred)

        gp_mean4data, _ = like.predict(like.x)

        color = radvel.plot.default_colors[ci]
        ci += 1

        ax.fill_between(xpred-2450000, gpmu+gp_orbit_model-stddev, gpmu+gp_orbit_model+stddev, 
                        color=color, alpha=0.5, lw=0
                        )
        ax.plot(xpred-2450000, gpmu, 'o-', rasterized=False, lw=0.1)
        ax.plot(xpred-2450000, gp_orbit_model, 'b-', rasterized=False, lw=0.2)

    # if not self.yscale_auto: 
    #     scale = np.std(self.rawresid+self.rvmod)
    #     ax.set_ylim(-self.yscale_sigma * scale, self.yscale_sigma * scale)

    ax.set_ylabel('RV [{ms:}]'.format(**radvel.plot.latex), weight='bold')
    ticks = ax.yaxis.get_majorticklocs()
    ax.yaxis.set_ticks(ticks[1:])

    return ax


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

def synthesize_data(args, P, mod, gp=False):
    alltimes = np.sort(np.loadtxt(args.schedule_path))
    times = alltimes[0:args.nobs]

    random = RandomState(1234)
    rvs = mod(times) + random.normal(loc=0, scale=P.params['jit'].value, size=len(times))
    errvel = np.zeros_like(rvs) + P.params['jit'].value

    if args.gp:
        # standard rotation GP that we will include in fit
        modtimes = np.linspace(2459300, 2459900, 1000)
        htimes = np.arange(2459300, 2459400, 0.005)

        random = RandomState(1234)
        gplike = GPLikelihood(mod, modtimes, np.zeros_like(modtimes)\
            + random.normal(loc=0, scale=mod.params['gp_amp'].value, size=len(modtimes)), np.zeros_like(modtimes)+0.3)
        gp, _ = gplike.predict(times)
        rvs += gp

        plgp, _ = gplike.predict(modtimes)
        if args.noise:
            fig = pl.figure(figsize=(6, 11.5))
            ax = pl.subplot(4,1,1)
            fgp, _ = gplike.predict(htimes)
            ax = plot_gp_like(gplike, 0)
            ax.annotate("Rotation", xy=(0.8, 0.9), xycoords='axes fraction')

            # granulation GP that we will not model
            par2 = Parameters(num_planets=0)
            par2['gp_amp'] = Parameter(value=0.5)
            par2['gp_explength'] = Parameter(value = 1.0)
            par2['gp_per'] = Parameter(value = 1/24)
            par2['gp_perlength'] = Parameter(value=0.3)
            mod2 = RVModel(par2, time_base=2459380)
            random = RandomState(1234)
            gplike = GPLikelihood(mod2, modtimes, np.zeros_like(modtimes)\
                + random.normal(loc=0, scale=par2['gp_amp'].value, size=len(modtimes)), np.zeros_like(modtimes)+0.3)
            gp, _ = gplike.predict(times)
            rvs += gp

            ax = pl.subplot(4, 1, 2)
            random = RandomState(1234)
            plgp2, _ = gplike.predict(modtimes)
            plgp += plgp2
            if args.noise:
                fgp2, _ = gplike.predict(htimes)
                fgp += fgp2

                ax.plot(htimes-2450000, fgp2, 'b-', lw=0.5)
                ax.set_ylabel('RV [{ms:}]'.format(**radvel.plot.latex), weight='bold')
                ax.annotate("Granulation", xy=(0.8, 0.9), xycoords='axes fraction')
                ax.set_xlim(9300, 9302)
                ax.set_ylim(-1, 1)

            ax = pl.subplot(4, 1, 3)
            ax.plot(modtimes-2450000, plgp, 'b-', lw=0.5)
            ax.set_ylabel('RV [{ms:}]'.format(**radvel.plot.latex), weight='bold')
            ticks = ax.yaxis.get_majorticklocs()
            ax.yaxis.set_ticks(ticks[1:])
            ax.set_xlabel('BJD - 2450000')
            ax.annotate("Total", xy=(0.8, 0.9), xycoords='axes fraction')

            ax = pl.subplot(4, 1, 4)
            f, pxx = periodogram(fgp, 1/(np.diff(htimes)[0]*86400), nfft=100000)
            ax.plot(f, pxx/1e6, '-', color='0.7', lw=0.2)
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlabel('Frequency [$Hz$]')
            ax.set_ylabel('PSD [({ms:})$^{{2}} \mu Hz^{{-1}}$]'.format(**radvel.plot.latex), weight='bold')
            ax.set_xlim(3e-8, 3e-3)
            ax.set_ylim(1e-9, 10)

            pl.savefig('noise_{}.pdf'.format(os.path.basename(args.radvel_setup).replace('.py', '')))

    if not args.gp:
        pars = mod.params.keys()
        for par in pars:
            if par.startswith('gp_'):
                mod.params[par].vary = False

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
    P.params['k1'].value = 0.1
    P.params['secosw1'].value = 0.0
    P.params['sesinw1'].value = 0.0
    if 'gp_per' in P.params.keys():
        P.params['gp_per'].value = P.params['gp_per'].value + 5
    mod = radvel.RVModel(P.params, time_base=2459380)

    if args.gp:
            like = radvel.likelihood.GPLikelihood(mod, P.data.time, 
                                          P.data.mnvel, P.data.errvel)
    else:
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
    post = radvel.fitting.maxlike_fitting(post, verbose=True)
    post.chains = radvel.mcmc(post, nwalkers=25, nrun=30000, ensembles=8, headless=True)
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

    sigmas = quantiles.loc[0.5] - quantiles.loc[0.159]

    # saved derived paramters to posterior file
    postfile = os.path.join(args.outputdir,
                            '{}_post_obj.pkl'.format(conf_base))
    post.derived = quantiles[outcols]
    post.writeto(postfile)

    csvfn = os.path.join(args.outputdir, conf_base+'_derived.csv.bz2')
    chains.to_csv(csvfn, columns=outcols, compression='bz2')

    print("Derived parameters:", outcols)
    return sigmas

def write_msigma(args, P, sigmas):
    outfile = './sigmas_{}.csv'.format(os.path.basename(args.radvel_setup).replace('.py', ''))
    if not os.path.exists(outfile):
        f = open(outfile, 'w')
        print("run,nobs,baseline,sigma_k,sigma_mp", file=f)
    else:
        f = open(outfile, 'a')

    run = args.outputdir
    print("{},{:d},{:0.2f},{:0.4f},{:0.4f}".format(run, args.nobs, args.baseline, 
                                                   P.k_injected/sigmas['k1'],
                                                   P.mpsini_injected/sigmas['mpsini1']), file=f)
    f.close()

def orbit_plot(args, post, P):
    run = args.outputdir
    saveto = args.outputdir + "/{}.pdf".format(args.outputdir)
    conf_base = os.path.basename(P.starname)

    if args.gp:
        RVPlot = orbit_plots.GPMultipanelPlot(post, saveplot=saveto, separate_orbit_gp=True)
    else:
        RVPlot = orbit_plots.MultipanelPlot(post, saveplot=saveto)
    RVPlot.status = dict()
    RVPlot.status['derive'] = {}
    RVPlot.status['derive']['chainfile'] = os.path.join(args.outputdir, conf_base+'_derived.csv.bz2')
    RVPlot.plot_multipanel()

if __name__ == '__main__':
    args = arguments()
    P, post = create_timeseries(args)
    chains = fit_timeseries(post)
    sigmas = get_msigma(P, post)
    args.baseline = P.data['time'].max() - P.data['time'].min()
    write_msigma(args, P, sigmas)
    orbit_plot(args, post, P)
