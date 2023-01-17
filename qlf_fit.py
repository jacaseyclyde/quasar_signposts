#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A one-line description or name.

A longer description that spans multiple lines.  Explain the purpose of the
file and provide a short list of the key classes/functions it contains.  This
is the docstring shown when some does 'import foo;foo?' in IPython, so it
should be reasonably useful and informative.

Created on Wed Oct  5 17:16:07 2022

@author: jacaseyclyde

OPTIONS ------------------------------------------------------------------
A description of each option that can be passed to this script

ARGUMENTS -------------------------------------------------------------
A description of each argument that can or must be passed to this script

"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# stdlib imports -------------------------------------------------------
import argparse
import pickle
import sys

# Third-party imports -----------------------------------------------
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import truncnorm

import pandas as pd
import h5py
import yaml

import astropy.units as u
from astropy.cosmology import WMAP9 as cosmo

import emcee
import multiprocess as mp
import schwimmbad

import matplotlib.pyplot as plt
import seaborn as sns
import corner

# Our own imports ---------------------------------------------------
from src.utils import nquad_vec

from src.models.qlf import SchechterQLF, DoublePowerLawQLF, Shen2020QLF
from src.models.qlf import LOG_L_SOLAR


# -----------------------------------------------------------------------------
# GLOBALS
# -----------------------------------------------------------------------------
LOG_L_MIN = 45
LOG_L_MAX = 48
Z_MIN = 0
Z_MAX = 1.5

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# LOCAL UTILITIES
# -----------------------------------------------------------------------------
# plotting options
context = 'talk'
sns.set_context(context)
sns.set_style('ticks')
sns.set_palette('colorblind')
rcparams = {
    # 'font.family': 'DejaVu Sans',
    # 'font.serif': 'Times',
    'lines.markersize': 10,
    # 'text.latex.preamble': r'\\usepackage{amsmath}',
    # 'text.usetex': False,
    'figure.figsize': [12.8, 9.6],
    'xtick.direction': 'in',
    'xtick.top': True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'ytick.right': True,
    'ytick.direction': 'in'
}
plt.rcParams.update(rcparams)

# color and hatch options to cycle through
CS = plt.rcParams['axes.prop_cycle'].by_key()['color']
HS = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
LS = ['-', ':', '--', '-.', (0, (3, 5, 1, 5, 1, 5))]


# -----------------------------------------------------------------------------
# CLASSES
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

# LOADING FUNCTIONS -------------------------------------------
def _load_data(f):
    pop = pd.read_csv(f)
    if 'weight' not in pop.columns:
        pop['weight'] = 1
    return pop


def _load_z_max():
    with open(r"./models/z_complete_crts_fn.pkl", "rb") as f:
        z_max_fn = pickle.load(f)
    return z_max_fn


def _load_log_l_min():
    with open(r"./models/log_l_bol_complete_crts_fn.pkl", "rb") as f:
        log_l_min_fn = pickle.load(f)
    return log_l_min_fn


def _load_completeness():
    with h5py.File('./data/processed/crts_completeness.h5', 'r') as hf:
        S_sky = np.array(hf['S_sky'])
        S_qual = np.array(hf['S_qual'])
        S_f = np.array(hf['S_f'])

    S = S_sky * S_qual * S_f
    return S


# PLOTTING FUNCTIONS ------------------------------------------
def _plot_qlf(qlf, log_l_bw=.5, z_bw=.5, alpha=.25, hatch=HS[0],
              linestyle=LS[0], label=None, fig=None, ax=None):
    # initialize plotting ranges
    log_l_range = np.linspace(LOG_L_MIN, LOG_L_MAX)
    z_range = np.linspace(Z_MIN, Z_MAX)

    # set up bins
    # log_l bins
    log_l_bins = np.arange(LOG_L_MIN, LOG_L_MAX + log_l_bw, log_l_bw)
    log_l_bin_min, log_l_bin_max = log_l_bins[:-1], log_l_bins[1:]

    # z bins
    z_bins = np.arange(Z_MIN, Z_MAX + z_bw, z_bw)
    z_bin_min, z_bin_max = z_bins[:-1], z_bins[1:]

    # calculate the qlf in log_l bins
    qlf_log_l_bins = np.empty((len(qlf), len(log_l_range), 0))
    for z_min, z_max in zip(z_bin_min, z_bin_max):
        qlf_log_l_i = np.transpose([nquad_vec(lambda z: qlf(log_l, z),
                                              [[z_min, z_max]], n_roots=21)
                                    for log_l in log_l_range])
        qlf_log_l_bins = np.append(qlf_log_l_bins,
                                   qlf_log_l_i[..., np.newaxis], axis=-1)

    qlf_log_l_bins_q = np.quantile(qlf_log_l_bins, q=[.16, .5, .84], axis=0)

    # calculate the qlf in z bins
    qlf_z_bins = np.empty((len(qlf), len(z_range), 0))
    for log_l_min, log_l_max in zip(log_l_bin_min, log_l_bin_max):
        qlf_z_i = np.transpose(
            [
                nquad_vec(lambda log_l: qlf(log_l, z),
                          [[log_l_min, log_l_max]],
                          n_roots=21)
                for z in z_range
                ]
            )
        qlf_z_bins = np.append(qlf_z_bins, qlf_z_i[..., np.newaxis], axis=-1)

    qlf_z_bins_q = np.quantile(qlf_z_bins, q=[.16, .5, .84], axis=0)

    # also calculate the total QLF over log_l and z
    qlf_z = np.transpose([nquad_vec(lambda log_l: qlf(log_l, z), [[45, 48]],
                                    n_roots=21) for z in z_range])
    qlf_z_q = np.quantile(qlf_z, q=[.16, .5, .84], axis=0)

    qlf_log_l = np.transpose([nquad_vec(lambda z: qlf(log_l, z), [[0, 1.5]],
                                        n_roots=21) for log_l in log_l_range])
    qlf_log_l_q = np.quantile(qlf_log_l, q=[.16, .5, .84], axis=0)

    # plot
    # adjust figsize for the number of columns we have
    # figsize = [2 * plt.rcParams['figure.figsize'][0],
    #            plt.rcParams['figure.figsize'][1]]
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    # plot total QLF over log_l
    if label is not None:
        lbl = label + ", Total"
    else:
        lbl = "Total"

    ax[0].plot(
        log_l_range,
        qlf_log_l_q[1],
        color=CS[0],
        label=lbl,
        linestyle=linestyle,
        )
    ax[0].fill_between(
        log_l_range,
        qlf_log_l_q[0],
        qlf_log_l_q[2],
        color=CS[0],
        alpha=alpha
        )
    ax[0].fill_between(
        log_l_range,
        qlf_log_l_q[0],
        qlf_log_l_q[2],
        facecolor="none",
        edgecolor=CS[0],
        hatch=hatch,
        alpha=alpha)

    # plot QLF over log_l bins
    for i, (z_min, z_max) in enumerate(
            zip(z_bin_min, z_bin_max)
            ):
        ax[0].plot(
            log_l_range,
            qlf_log_l_bins_q[1, :, i],
            color=CS[i+1],
            linestyle=linestyle,
            label=r"{2}, ${0} \leq z < {1}$".format(z_min, z_max, label)
            )
        ax[0].fill_between(
            log_l_range,
            qlf_log_l_bins_q[0, :, i],
            qlf_log_l_bins_q[2, :, i],
            color=CS[i+1],
            alpha=alpha)
        ax[0].fill_between(
            log_l_range,
            qlf_log_l_bins_q[0, :, i],
            qlf_log_l_bins_q[2, :, i],
            facecolor="none",
            edgecolor=CS[i+1],
            hatch=hatch,
            alpha=alpha)

    ax[0].legend()

    ax[0].set_ylim(1e-11, 1e-4)
    ax[0].set_yscale('log')
    ax[0].set_xlabel(r"$\log\left(L_{\rm bol}~[\rm{erg}~\rm{s}^{-1}]\right)$")
    ax[0].set_ylabel(r"$d\Phi_{\rm QSO} / d\log L_{\rm bol}~[\log L_{\rm bol}^{-1}~\rm{Mpc}^{-3}]$")

    # plot total QLF over z
    ax[1].plot(
        z_range,
        qlf_z_q[1],
        color=CS[0],
        label=lbl,
        linestyle=linestyle,
        )
    ax[1].fill_between(
        z_range,
        qlf_z_q[0],
        qlf_z_q[2],
        color=CS[0],
        alpha=alpha
        )
    ax[1].fill_between(
        z_range,
        qlf_z_q[0],
        qlf_z_q[2],
        facecolor="none",
        edgecolor=CS[0],
        hatch=hatch,
        alpha=alpha
        )

    # plot QLF over z bins
    for i, (log_l_min, log_l_max) in enumerate(
            zip(log_l_bin_min, log_l_bin_max)
            ):
        ax[1].plot(
            z_range,
            qlf_z_bins_q[1, :, i],
            color=CS[i+1],
            linestyle=linestyle,
            label=r"{2}, ${0} \leq \log L_{{\rm bol}} < {1}$".format(
                log_l_min,
                log_l_max,
                label
                )
            )
        ax[1].fill_between(
            z_range,
            qlf_z_bins_q[0, :, i],
            qlf_z_bins_q[2, :, i],
            color=CS[i+1],
            alpha=alpha
            )
        ax[1].fill_between(
            z_range,
            qlf_z_bins_q[0, :, i],
            qlf_z_bins_q[2, :, i],
            facecolor="none",
            edgecolor=CS[i+1],
            hatch=hatch,
            alpha=alpha
            )

    ax[1].legend()

    ax[1].set_ylim(1e-11, 1e-4)
    ax[1].set_yscale('log')
    ax[1].set_xlabel(r"$z$")
    ax[1].set_ylabel(r"$d\Phi_{\rm QSO} / dz~[\rm{Mpc}^{-3}]$")

    plt.tight_layout()


# MCMC FUNCTIONS ----------------------------------------------
def _fit_model(log_l, z, *theta):
    a0, a1, a2, b0, b1, b2, c0, c1, c2, d0, d1 = theta
    return qlf_model(
        log_l, z,
        a0=a0, a1=a1, a2=a2,
        b0=b0, b1=b1, b2=b2,
        c0=c0, c1=c1, c2=c2,
        d0=d0, d1=d1
    )


def _n_integ(log_l, z, *theta):
    a0, a1, a2, b0, b1, b2, c0, c1, c2, d1 = theta
    res = qlf_model(
        log_l, z,
        a0=a0, a1=a1, a2=a2,
        b0=b0, b1=b1, b2=b2,
        c0=c0, c1=c1, c2=c2,
        d0=np.zeros_like(d1), d1=d1
    )
    res = res * cosmo.differential_comoving_volume(z)
    res = res.to(u.Mpc ** 3 / u.sr).value
    res = res * 4 * np.pi
    return res


def _log_normalization(theta, n_bh, completeness_fn, cosmo=cosmo, compl=1,
                       crts_based=False, ext=False):
    if callable(completeness_fn):
        log_l_min = completeness_fn(pop['z'].to_numpy())
        log_l_min = np.maximum(LOG_L_MIN, log_l_min)

        denom = np.transpose(
            [
                nquad_vec(
                    _n_integ,
                    [[llm, LOG_L_MAX], [Z_MIN, Z_MAX]],
                    args=theta
                    )
                for llm in log_l_min
                ]
            )
        res = n_bh / np.sum(denom, axis=-1)
    else:
        log_l_min = completeness_fn
        log_l_min = np.maximum(LOG_L_MIN, log_l_min)
        denom = nquad_vec(
            _n_integ,
            [[log_l_min, LOG_L_MAX], [Z_MIN, Z_MAX]],
            args=theta
            )
        res = n_bh / denom

    res = res / compl
    return np.log10(res)


def log_probability(theta, bounds, completeness_fn, cosmo=cosmo, compl=1,
                    crts_based=False, ext=False):
    # first check basic bounds
    a0, a1, a2, b0, b1, b2, c0, c1, c2, d1 = theta

    (a0_min, a1_min, a2_min,
     b0_min, b1_min, b2_min,
     c0_min, c1_min, c2_min,
     d0_min, d1_min) = bounds[0]
    (a0_max, a1_max, a2_max,
     b0_max, b1_max, b2_max,
     c0_max, c1_max, c2_max,
     d0_max, d1_max) = bounds[1]

    if ~(
        np.all(a0_min <= a0 <= a0_max)
        and np.all(a1_min <= a1 <= a1_max)
        and np.all(a2_min <= a2 <= a2_max)
        and np.all(b0_min <= b0 <= b0_max)
        and np.all(b1_min <= b1 <= b2_max)
        and np.all(b1_min <= b2 <= b2_max)
        and np.all(b1 <= b2)
        and np.all(c0_min <= c0 <= c0_max)
        and np.all(c1_min <= c1 <= c2_max)
        and np.all(c1_min <= c2 <= c2_max)
        and np.all(c1 <= c2)
        and np.all(d1_min <= d1 <= d1_max)
    ):
        return -np.inf, -np.inf

    # calculate number of bhs
    # assume poisson errors on the number. if it's based on CRTS, match CRTS
    if crts_based:
        n_crts = np.sum(crts_pop['weight'].to_numpy())
        n_crts_i = np.random.poisson(n_crts)
        if ext and (n_crts_i > len(crts_pop)):
            n_crts_i = len(crts_pop)
        n_bh = np.sum(pop['weight'].to_numpy())
        if ext:
            n_bh = n_bh + n_crts

        n_bh = n_bh * n_crts_i / n_crts
    else:
        n_bh = np.sum(pop['weight'].to_numpy())
        n_bh = np.random.poisson(n_bh)

    # if n_bh is zero it causes issues later
    if n_bh == 0:
        n_bh = .01

    # also calculate the log normalization
    # uses the blobs interface of emcee
    d0 = _log_normalization(theta, n_bh, completeness_fn, cosmo=cosmo,
                            compl=compl, crts_based=crts_based, ext=ext)

    theta2 = a0, a1, a2, b0, b1, b2, c0, c1, c2, d0, d1

    lp = log_prior(theta2, bounds)
    if not np.isfinite(lp):
        return -np.inf, d0

    if ext:
        crts_sample = crts_pop.sample(n=n_crts_i, replace=False,
                                      weights=crts_pop['weight'])
        sample = pd.concat([pop, crts_sample], ignore_index=True)
    else:
        sample = pop

    ll = log_likelihood(theta2, sample, completeness_fn)

    # print(np.shape(d0))

    return lp + ll, d0


def log_prior(theta, bounds):
    a0, a1, a2, b0, b1, b2, c0, c1, c2, d0, d1 = theta

    (a0_min, a1_min, a2_min,
     b0_min, b1_min, b2_min,
     c0_min, c1_min, c2_min,
     d0_min, d1_min) = bounds[0]
    (a0_max, a1_max, a2_max,
     b0_max, b1_max, b2_max,
     c0_max, c1_max, c2_max,
     d0_max, d1_max) = bounds[1]

    if (
        np.all(a0_min <= a0 <= a0_max)
        and np.all(a1_min <= a1 <= a1_max)
        and np.all(a2_min <= a2 <= a2_max)
        and np.all(b0_min <= b0 <= b0_max)
        and np.all(b1_min <= b1 <= b2_max)
        and np.all(b1_min <= b2 <= b2_max)
        and np.all(b1 <= b2)
        and np.all(c0_min <= c0 <= c0_max)
        and np.all(c1_min <= c1 <= c2_max)
        and np.all(c1_min <= c2 <= c2_max)
        and np.all(c1 <= c2)
        and np.all(d0_min <= d0 <= d0_max)
        and np.all(d1_min <= d1 <= d1_max)
    ):
        z_dummy = 4  # following Shen+ 2020 implementation
        ls = qlf_model.low_slope(z_dummy, np.transpose([a0, a1, a2]))
        hs = qlf_model.high_slope(z_dummy, a0=b0, a1=b1, a2=b2)
        lb = qlf_model.log_break(z_dummy, a0=c0, a1=c1, a2=c2)
        ln = qlf_model.log_norm(z_dummy, coef=np.transpose([d0, d1]))

        if (
            np.all(-5 <= ls <= 5)
            and np.all(-5 <= hs <= 5)
            and np.all(ls <= hs)
            and np.all(5 <= lb <= 20)
            and np.all(-15 <= ln <= 5)
        ):
            return 0.0
    return -np.inf


def log_likelihood(theta, sample, completeness_fn):
    mdl = _fit_model(sample['log_l_bol'].to_numpy(),
                     sample['z'].to_numpy(),
                     *theta)

    m = mdl == 0  # find anywhere the model returns 0 due to numerical limits
    mdl[m] = 1  # temporarily replace it
    ln_num = np.log(mdl)  # calculate log without errors
    ln_num[m] = -np.inf  # correct 0s

    # find the minimum luminosity
    if callable(completeness_fn):
        log_l_min = completeness_fn(sample['z'].to_numpy())
    else:
        log_l_min = completeness_fn
    log_l_min = np.maximum(LOG_L_MIN, log_l_min)

    # integrate over the model to get the denominator
    try:
        denom = np.array([
            nquad_vec(_fit_model,
                      [[llm, LOG_L_MAX], [Z_MIN, Z_MAX]],
                      args=theta)
            for llm in log_l_min
            ])
    except TypeError:
        denom = nquad_vec(_fit_model,
                          [[log_l_min, LOG_L_MAX], [Z_MIN, Z_MAX]],
                          args=theta
                          )

    if np.any(denom == 0):
        print(denom)
        print(theta)
        print(log_l_min)
        raise ValueError("denom is 0")

    ln_denom = np.log(denom)

    # sum the log probabilities, weighted by their maximum precision estimates
    res = np.nansum(sample['weight'].to_numpy() * (ln_num - ln_denom))

    if np.isnan(res):
        print(ln_num)
        print(ln_denom)
        print(res)
        raise ValueError("NaN res")

    return res


def initialize_params(init_bounds, param_bounds, nwalkers=32,
                      completeness_fn=LOG_L_MIN, cosmo=cosmo, compl=1,
                      crts_based=False):
    # grab the bounds on parameters we fit directly
    pos_bounds = param_bounds[:, np.arange(len(param_bounds[0])) != 9]
    param_low, param_high = pos_bounds
    i_bounds = init_bounds[:, np.arange(len(init_bounds[0])) != 9]
    init_low, init_high = i_bounds
    ndim = len(init_low)

    # first make sure we have at least one initialization that's valid
    init_mean = (init_high + init_low) / 2
    init_diff = (init_high - init_low) / 2
    a = (param_low - init_mean) / init_diff
    b = (param_high - init_mean) / init_diff
    pos = truncnorm.rvs(a, b, loc=init_mean, scale=init_diff,
                        size=(nwalkers, ndim))

    # calculate number of bhs
    # assume poisson errors on the number. if it's based on CRTS, match CRTS
    if crts_based:
        n_crts = np.sum(crts_pop['weight'].to_numpy())
        n_crts_i = np.random.poisson(n_crts)
        n_bh = np.sum(pop['weight'].to_numpy()) * n_crts_i / n_crts
    else:
        n_bh = np.sum(pop['weight'].to_numpy())
        n_bh = np.random.poisson(n_bh)

    # if n_bh is zero it causes issues later
    if n_bh == 0:
        n_bh = .01

    d0 = _log_normalization(pos.T, n_bh, completeness_fn=completeness_fn,
                            cosmo=cosmo, compl=compl, crts_based=crts_based)
    pos = np.insert(pos, -1, d0, axis=1)
    lp = [log_prior(p, param_bounds) for p in pos]
    invalid = np.isinf(lp)  # find invalid initializations
    valid = ~invalid
    while np.sum(valid) < nwalkers:  #
        pos_tmp = truncnorm.rvs(a, b, loc=init_mean, scale=init_diff,
                                size=(np.sum(invalid), ndim))

        # calculate number of bhs
        # assume poisson errors on the number. if it's based on CRTS, match CRTS
        if crts_based:
            n_crts_tmp = np.sum(crts_pop['weight'].to_numpy())
            n_crts_i_tmp = np.random.poisson(n_crts_tmp)
            n_bh_tmp = np.sum(pop['weight'].to_numpy()) * n_crts_i_tmp / n_crts_tmp
        else:
            n_bh_tmp = np.sum(pop['weight'].to_numpy())
            n_bh_tmp = np.random.poisson(n_bh_tmp)

        # if n_bh is zero it causes issues later
        if n_bh_tmp == 0:
            n_bh_tmp = .01

        d0_tmp = _log_normalization(pos_tmp.T, n_bh_tmp, completeness_fn,
                                    cosmo=cosmo, compl=compl,
                                    crts_based=crts_based)
        pos_tmp = np.insert(pos_tmp, -1, d0_tmp, axis=1)

        pos[invalid] = pos_tmp
        lp = [log_prior(p, param_bounds) for p in pos]
        invalid = np.isinf(lp)  # find invalid initializations
        valid = ~invalid

    # remove d0 since we don't fit for it explicitly
    pos = pos[:, np.arange(len(pos[0])) != 9]
    return pos


def _converged(acorr, len_acorr=50):
    return 0


def mcmc_sampler(bounds, completeness_fn=LOG_L_MIN, cosmo=cosmo,
                 compl=1, init_bounds=None, nwalkers=32, ncpus=mp.cpu_count(),
                 max_nsamples=10000, len_acorr=50, filename=None,
                 track_acorr=True, cont=True, crts_based=False, ext=False,
                 mpi=False
                 ):
    args = (bounds, completeness_fn, cosmo, compl, crts_based, ext)

    if init_bounds is None:
        init_bounds = bounds

    low, high = init_bounds
    ndim = len(low) - 1

    # initialize the backend
    if filename is not None:
        backend = emcee.backends.HDFBackend("./data/processed/"
                                            + filename
                                            + "_chain.h5")
        if not cont:
            backend.reset(nwalkers, ndim)
    else:
        backend = None

    # initialize autocorrelation tracking
    if track_acorr:
        acorr_idx = np.array([0])
        acorr = np.array([0])

    # start MCMC
    print("Starting MCMC...")
    if ncpus is not None and ncpus > 1:  # multiprocessing
        with schwimmbad.choose_pool(mpi=mpi, processes=ncpus) as pool:
            if mpi and not pool.is_master():
                pool.wait()
                sys.exit(0)

            sampler = emcee.EnsembleSampler(
                nwalkers,
                ndim,
                log_probability,
                args=args,
                backend=backend,
                pool=pool,
                moves=[
                    (emcee.moves.DEMove(), .8),
                    (emcee.moves.DESnookerMove(), .2)
                    ],
                )

            # load/initialize parameters
            if cont:
                try:
                    pos = sampler._previous_state
                except Exception:
                    print("Initializing parameters...")
                    pos = initialize_params(
                        init_bounds,
                        bounds,
                        nwalkers=nwalkers,
                        completeness_fn=completeness_fn,
                        cosmo=cosmo,
                        compl=compl,
                        crts_based=crts_based,
                        )
                    print("Done!")
            else:
                print("Initializing parameters...")
                pos = initialize_params(
                    init_bounds,
                    bounds,
                    nwalkers=nwalkers,
                    completeness_fn=completeness_fn,
                    cosmo=cosmo,
                    compl=compl,
                    crts_based=crts_based,
                    )
                print("Done!")

            for sample in sampler.sample(pos, iterations=max_nsamples,
                                         progress=True):
                # check convergence every 100 steps
                if sampler.iteration % 100:
                    continue

                if track_acorr:
                    # compute current autocorr time
                    tau = sampler.get_autocorr_time(tol=0)

                    # Check convergence
                    converged = np.all(tau * len_acorr < sampler.iteration)
                    converged &= np.all(np.abs(acorr[-1] - tau) / tau < .01)

                    acorr_idx = np.append(acorr_idx, sampler.iteration)
                    acorr = np.append(acorr, np.nanmean(tau))

                    if converged:
                        break
    else:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            args=args,
            backend=backend,
            moves=[
                (emcee.moves.DEMove(), .8),
                (emcee.moves.DESnookerMove(), .2)
                ],
            )

        for sample in sampler.sample(pos, iterations=max_nsamples,
                                     progress=True):
            # check convergence every 100 steps
            if sampler.iteration % 100:
                continue

            if track_acorr:
                # compute current autocorr time
                tau = sampler.get_autocorr_time(tol=0)

                # Check convergence
                converged = np.all(tau * len_acorr < sampler.iteration)
                converged &= np.all(np.abs(acorr[-1] - tau) / tau < .01)

                acorr_idx = np.append(acorr_idx, sampler.iteration)
                acorr = np.append(acorr, np.nanmean(tau))

                if converged:
                    break

    if track_acorr:
        plt.figure()

        plt.plot(acorr_idx, acorr_idx / len_acorr, "--k")
        plt.plot(acorr_idx, acorr)

        plt.title("Autocorrelation")
        plt.xlabel("number of steps")
        plt.ylabel(r"mean $\hat{\tau}$")

        plt.xlim(left=0)
        plt.ylim(bottom=0)

        plt.savefig("./reports/figures/" + filename + "_autocorr.pdf")

        return sampler, acorr, acorr_idx
    else:
        return sampler


# -----------------------------------------------------------------------------
# RUNTIME PROCEDURE
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    '''
    Complete description of the runtime of the script, what it does and how it
    should be used
    :timeComplexityTerm TERM_X: type - term used in the Complexity formula
    :timeComplexityDominantOperation  OP_X: type - operation considered to
        calculate the time complexity of this method
    :timeComplexity: O(OP_X*TERM_X²)
    '''
    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="Fits a double power law to a quasar sample."
        )
    parser.add_argument(
        "--quasars",
        help="CSV of the population to fit.",
        required=True,
        )
    parser.add_argument(
        "--crts_based",
        action='store_true',
        )
    parser.add_argument(
        "--extensions",
        action="store_true",
        )
    parser.add_argument(
        "--flux_incomplete",
        action='store_true',
        )
    parser.add_argument(
        "--cont",
        action="store_true",
        )
    parser.add_argument(
        "--track_acorr",
        action="store_true",
        )
    parser.add_argument(
        "--thin",
        default=1,
        type=float,
        )
    parser.add_argument(
        "--filename",
        )
    parser.add_argument(
        "--config",
        )
    parser.add_argument(
        "--n_samples",
        default=5000,
        type=int
        )
    parser.add_argument(
        "--n_walkers",
        default=12,
        type=int
        )
    parser.add_argument(
        "--n_cpus",
        default=12,
        type=int
        )
    parser.add_argument(
        "--mpi",
        action="store_true",
        )
    args = parser.parse_args()

    if not 0 <= args.thin <= 1:
        raise argparse.ArgumentTypeError("Thin should be between 0 and 1")

    # load the Shen+ 2020 QLF for plotting comparison
    with open(r"./models/qlf_shen20.pkl", "rb") as f:
        qlf_s20 = pickle.load(f)

    # plot the Shen+ 2020 QLF
    figsize = [2 * plt.rcParams['figure.figsize'][0],
               plt.rcParams['figure.figsize'][1]]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    _plot_qlf(qlf_s20, fig=fig, ax=ax)
    plt.savefig('./reports/figures/shen20_qlf.pdf')

    # load the population we're fitting to
    pop = _load_data(args.quasars)

    if args.crts_based:
        crts_pop = _load_data('./data/processed/reduced_crts_complete.csv')
        S = _load_completeness()
    else:
        S = 1

    if args.thin < 1:
        pop = pop.sample(frac=args.thin)
        S = S * args.thin

    # MCMC fitting
    qlf_model = Shen2020QLF()

    labels = [
        r"$a_{0}$", r"$a_{1}$", r"$a_{2}$",
        r"$b_{0}$", r"$b_{1}$", r"$b_{2}$",
        r"$c_{0}$", r"$c_{1}$", r"$c_{2}$",
        r"$d_{0}$", r"$d_{1}$"
    ]

    # load bounds
    with open(args.config) as f:
        bounds_cfg = yaml.safe_load(f)

    bounds_low = [
        bounds_cfg['a0']['min'],
        bounds_cfg['a1']['min'],
        bounds_cfg['a2']['min'],
        bounds_cfg['b0']['min'],
        bounds_cfg['b1']['min'],
        bounds_cfg['b2']['min'],
        bounds_cfg['c0']['min'] - LOG_L_SOLAR,
        bounds_cfg['c1']['min'],
        bounds_cfg['c2']['min'],
        bounds_cfg['d0']['min'],
        bounds_cfg['d1']['min']
    ]
    bounds_high = [
        bounds_cfg['a0']['max'],
        bounds_cfg['a1']['max'],
        bounds_cfg['a2']['max'],
        bounds_cfg['b0']['max'],
        bounds_cfg['b1']['max'],
        bounds_cfg['b2']['max'],
        bounds_cfg['c0']['max'] - LOG_L_SOLAR,
        bounds_cfg['c1']['max'],
        bounds_cfg['c2']['max'],
        bounds_cfg['d0']['max'],
        bounds_cfg['d1']['max']
    ]
    bounds = np.array([bounds_low, bounds_high])

    init_low = [
        bounds_cfg['a0']['init']['min'],
        bounds_cfg['a1']['init']['min'],
        bounds_cfg['a2']['init']['min'],
        bounds_cfg['b0']['init']['min'],
        bounds_cfg['b1']['init']['min'],
        bounds_cfg['b2']['init']['min'],
        bounds_cfg['c0']['init']['min'],
        bounds_cfg['c1']['init']['min'],
        bounds_cfg['c2']['init']['min'],
        bounds_cfg['d0']['init']['min'],
        bounds_cfg['d1']['init']['min']
    ]
    init_high = [
        bounds_cfg['a0']['init']['max'],
        bounds_cfg['a1']['init']['max'],
        bounds_cfg['a2']['init']['max'],
        bounds_cfg['b0']['init']['max'],
        bounds_cfg['b1']['init']['max'],
        bounds_cfg['b2']['init']['max'],
        bounds_cfg['c0']['init']['max'],
        bounds_cfg['c1']['init']['max'],
        bounds_cfg['c2']['init']['max'],
        bounds_cfg['d0']['init']['max'],
        bounds_cfg['d1']['init']['max']
    ]
    init = np.array([init_low, init_high])

    # determine the type of completeness function to use
    if args.flux_incomplete:
        completeness = _load_log_l_min()
    else:
        completeness = LOG_L_MIN

    # sample
    sampler, ac, idx = mcmc_sampler(
        bounds,
        completeness_fn=completeness,
        compl=S,
        init_bounds=init,
        nwalkers=args.n_walkers,
        ncpus=args.n_cpus,
        max_nsamples=args.n_samples,
        filename=args.filename,
        track_acorr=args.track_acorr,
        cont=args.cont,
        crts_based=args.crts_based,
        ext=args.extensions,
        mpi=args.mpi,
        )

    try:
        tau = sampler.get_autocorr_time()
    except Exception:
        print("Run a longer chain!")
        tau = sampler.get_autocorr_time(tol=0)

    # marge samples and blobs and flatten
    samples = sampler.get_chain()
    log_norm = sampler.get_blobs()
    samples = np.concatenate((samples, log_norm[..., np.newaxis]), axis=-1)
    samples[..., [9, 10]] = samples[..., [10, 9]]

    discard = int(3 * tau.max())
    thin = 15
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    flat_log_norm = sampler.get_blobs(discard=discard, thin=thin, flat=True)
    flat_samples = np.concatenate(
        (flat_samples, flat_log_norm[..., np.newaxis]),
        axis=-1
        )
    flat_samples[..., [9, 10]] = flat_samples[..., [10, 9]]

    # plot the chains
    fig, axes = plt.subplots(
        len(labels),
        figsize=(10, 2 * len(labels) - 1),
        sharex=True)

    for i in range(len(labels)):
        ax = axes[i]
        ax.plot(samples[..., i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        if i == 4 or i == 5:
            ax.set_ylim(bounds_cfg['b1']['min'], bounds_cfg['b2']['max'])
        elif i == 7 or i == 8:
            ax.set_ylim(bounds_cfg['c1']['min'], bounds_cfg['c2']['max'])
        else:
            ax.set_ylim(bounds_low[i], bounds_high[i])
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    plt.savefig('./reports/figures/' + args.filename + '_chains.pdf')

    # calculate quantiles
    mcmc = np.quantile(flat_samples, [.16, .5, .84], axis=0)
    q = np.diff(mcmc, axis=0)

    for i in range(len(labels)):
        txt = r"{3} = {0:.3f} (+{2:.3f}/-{1:.3f})"
        txt = txt.format(mcmc[1, i], q[0, i], q[1, i], labels[i])
        print(txt)

    # make a corner plot
    fig = corner.corner(flat_samples, labels=labels,
                        quantiles=[.16, .5, .84], truths=mcmc[1, :])
    plt.savefig('./reports/figures/' + args.filename + '_corner.pdf')

    # save a selection of fit models
    inds = np.random.choice(range(len(flat_samples)), size=1000, replace=False)
    flat_samples = flat_samples[inds]

    a0, a1, a2 = flat_samples[:, 0], flat_samples[:, 1], flat_samples[:, 2]
    b0, b1, b2 = flat_samples[:, 3], flat_samples[:, 4], flat_samples[:, 5]
    c0, c1, c2 = flat_samples[:, 6], flat_samples[:, 7], flat_samples[:, 8]
    d0, d1 = flat_samples[:, 9], flat_samples[:, 10]

    post = Shen2020QLF(
        a0=a0, a1=a1, a2=a2,
        b0=b0, b1=b1, b2=b2,
        c0=c0, c1=c1, c2=c2,
        d0=d0, d1=d1
    )
    with open("./models/" + args.filename + "_posterior.pkl", "wb") as f:
        pickle.dump(post, f)

    # plot and compare to the Shen+ 2020 QLF
    figsize = [2 * plt.rcParams['figure.figsize'][0],
               plt.rcParams['figure.figsize'][1]]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    _plot_qlf(qlf_s20, hatch=HS[0], linestyle=LS[0], label="QLF",
              fig=fig, ax=ax)
    _plot_qlf(post, hatch=HS[1], linestyle=LS[1], label="Fit", fig=fig, ax=ax)
    plt.savefig("./reports/figures/" + args.filename + "_qlf.pdf")
