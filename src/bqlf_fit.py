#!/usr/bin/env python
# coding: utf-8

# # Sample Populations
#
# First we'll import the populations we're using. We'll try fits to the CRTS sample alone, as well as to an extrapolated populaiton of bianries that includes the CRTS sample.

# ## QSOs

# In[1]:

qso_pop = pd.read_csv('../data/processed/mock_qso_pop.csv')
qso_pop['weight'] = 1
qso_pop


# ## CRTS

# In[2]:


crts_pop = pd.read_csv('../data/processed/reduced_crts_complete.csv')
crts_pop


# ## Extrapolated Sample: CRTS + Mock Binaries

# In[3]:


bq_pop = pd.read_csv('../data/processed/sky_freq_bq_pop.csv')
bq_pop


# ## Completeness Functions
#
# Let's also make sure to import the completeness function $z_{\max}(L)$, i.e. the maximum redshift a given luminosity is visible. We lso want this inverse, $L_{\min}(z)$, i.e. the minimum luminosity completely visible at any given redshift.

# In[4]:


import sys
sys.path.insert(0, '../')

import pickle
import h5py


# In[5]:


with open(r"../models/z_complete_crts_fn.pkl", "rb") as f:
    z_max_fn = pickle.load(f)


# In[6]:


with open(r"../models/log_l_bol_complete_crts_fn.pkl", "rb") as f:
    log_l_min_fn = pickle.load(f)


# In[7]:


with h5py.File('../data/processed/crts_completeness.h5', 'r') as hf:
    S_sky = np.array(hf['S_sky'])
    S_qual = np.array(hf['S_qual'])
    S_f = np.array(hf['S_f'])


# ## Comparison

# In[8]:


# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
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
cs = plt.rcParams['axes.prop_cycle'].by_key()['color']


# In[9]:


hs = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']


# In[10]:


# crts_edges = [45, 45.25, 45.5, 45.75,
#               46, 46.25, 46.5, 46.75,
#               47, 47.25, 47.5, 47.75, 48]
crts_edges = np.histogram_bin_edges(crts_pop['log_l_bol'], weights=crts_pop['weight'])


# In[11]:


plt.figure()

plt.hist(qso_pop['log_l_bol'],
         bins=crts_edges,
         weights=np.full_like(qso_pop['log_l_bol'], S_f*S_qual*S_sky),
         log=True,
         label='QLF',
         alpha=.25,
         hatch=hs[0],
        )
plt.hist(crts_pop['log_l_bol'],
         bins=crts_edges,
         weights=crts_pop['weight'],
         log=True,
         label='CRTS',
         alpha=.25,
         hatch=hs[1],
        )
plt.hist(bq_pop['log_l_bol'],
         bins=crts_edges,
         weights=bq_pop['weight'],
         log=True,
         label='CRTS + Extensions',
         alpha=.25,
         hatch=hs[2],
        )

plt.legend()
plt.show()


# # Quasar Model Check
#
# We will be fitting several different versions of binary quasar population models. As a sanity check, we'll want to compare these to the quasar population. Let's import that first.

# In[12]:


with open(r"../models/qlf_shen20.pkl", "rb") as f:
    qlf = pickle.load(f)


# In[13]:


import numpy as np
from src.utils import nquad_vec


# In[14]:


# CRTS includes two binary candidates with log_l < 45, though by only
# a small margin. Nevertheless we will exclude these from the 1/V_max
# approach to constructing the mass function. We will include these
# log_l < 45 candidates for the STY method
LOG_L_MIN = 45
LOG_L_MAX = 48
Z_MIN = 0
# Z_MIN = crts_pop['z'].min()
Z_MAX = 1.5


# In[15]:


# Next let's plot the z and log_l dependent QSO number density
log_l_range = np.linspace(LOG_L_MIN, LOG_L_MAX)

log_l_bw = .5
log_l_bins = np.arange(LOG_L_MIN, LOG_L_MAX + log_l_bw, log_l_bw)
# log_l_bins = np.histogram_bin_edges(crts_pop['log_l_bol'], bins='auto')
# log_l_bw = np.diff(log_l_bins)[0]
log_l_bin_min, log_l_bin_max = log_l_bins[:-1], log_l_bins[1:]
log_l_bin_mid = np.mean([log_l_bin_min, log_l_bin_max], axis=0)

z_bw = .5
z_range = np.linspace(Z_MIN, Z_MAX)
z_bins = np.arange(Z_MIN, Z_MAX + z_bw, z_bw)
z_bin_min, z_bin_max = z_bins[:-1], z_bins[1:]
z_bin_mid = np.mean([z_bin_min, z_bin_max], axis=0)

qlf_log_l_bins = np.empty((len(qlf), len(log_l_range), 0))
for z_min, z_max in zip(z_bin_min, z_bin_max):
    qlf_log_l_i = np.transpose([nquad_vec(lambda z: qlf(log_l, z),
                                          [[z_min, z_max]], n_roots=21)
                                for log_l in log_l_range])
    qlf_log_l_bins = np.append(qlf_log_l_bins, qlf_log_l_i[..., np.newaxis], axis=-1)

qlf_log_l_bins_q = np.quantile(qlf_log_l_bins, q=[.16, .5, .84], axis=0)

qlf_z_bins = np.empty((len(qlf), len(z_range), 0))
for log_l_min, log_l_max in zip(log_l_bin_min, log_l_bin_max):
    qlf_z_i = np.transpose([nquad_vec(lambda log_l: qlf(log_l, z),
                                          [[log_l_min, log_l_max]], n_roots=21)
                                for z in z_range])
    qlf_z_bins = np.append(qlf_z_bins, qlf_z_i[..., np.newaxis], axis=-1)

qlf_z_bins_q = np.quantile(qlf_z_bins, q=[.16, .5, .84], axis=0)

qlf_z = np.transpose([nquad_vec(lambda log_l: qlf(log_l, z), [[45, 48]], n_roots=21) for z in z_range])
qlf_z_q = np.quantile(qlf_z, q=[.16, .5, .84], axis=0)

qlf_log_l = np.transpose([nquad_vec(lambda z: qlf(log_l, z), [[0, 1.5]], n_roots=21) for log_l in log_l_range]) #* np.log(10)
qlf_log_l_q = np.quantile(qlf_log_l, q=[.16, .5, .84], axis=0)


# In[16]:


# adjust figsize for the number of columns we have
figsize = [2 * plt.rcParams['figure.figsize'][0],
           plt.rcParams['figure.figsize'][1]]
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

alpha = .25

ax[0].plot(log_l_range, qlf_log_l_q[1], color=cs[0], label='Total')
ax[0].fill_between(log_l_range, qlf_log_l_q[0], qlf_log_l_q[2],
                   color=cs[0], alpha=alpha)
ax[0].fill_between(log_l_range, qlf_log_l_q[0], qlf_log_l_q[2],
                   facecolor="none", edgecolor=cs[0], hatch='/', alpha=alpha)

for i, (z_min, z_max) in enumerate(zip(z_bin_min, z_bin_max)):
    ax[0].plot(log_l_range, qlf_log_l_bins_q[1, :, i], color=cs[i+1], linestyle='--',
               label=r"${0} \leq z < {1}$".format(z_min, z_max))
    ax[0].fill_between(log_l_range, qlf_log_l_bins_q[0, :, i], qlf_log_l_bins_q[2, :, i],
                       color=cs[i+1], alpha=alpha)
    ax[0].fill_between(log_l_range, qlf_log_l_bins_q[0, :, i], qlf_log_l_bins_q[2, :, i],
                       facecolor="none", edgecolor=cs[i+1], hatch='/', alpha=alpha)

ax[0].legend()

ax[0].set_ylim(1e-11, 1e-4)
ax[0].set_yscale('log')
ax[0].set_xlabel(r"$\log\left(L_{\rm bol}~[\rm{erg}~\rm{s}^{-1}]\right)$")
ax[0].set_ylabel(r"$d\Phi_{\rm QSO} / d\log L_{\rm bol}~[\log L_{\rm bol}^{-1}~\rm{Mpc}^{-3}]$")

ax[1].plot(z_range, qlf_z_q[1], color=cs[0], label='Total')
ax[1].fill_between(z_range, qlf_z_q[0], qlf_z_q[2],
                   color=cs[0], alpha=alpha)
ax[1].fill_between(z_range, qlf_z_q[0], qlf_z_q[2],
                   facecolor="none", edgecolor=cs[0], hatch='/', alpha=alpha)

for i, (log_l_min, log_l_max) in enumerate(zip(log_l_bin_min, log_l_bin_max)):
    ax[1].plot(z_range, qlf_z_bins_q[1, :, i], color=cs[i+1], linestyle='--',
               label=r"${0} \leq \log L_{{\rm bol}} < {1}$".format(log_l_min, log_l_max))
    ax[1].fill_between(z_range, qlf_z_bins_q[0, :, i], qlf_z_bins_q[2, :, i],
                       color=cs[i+1], alpha=alpha)
    ax[1].fill_between(z_range, qlf_z_bins_q[0, :, i], qlf_z_bins_q[2, :, i],
                       facecolor="none", edgecolor=cs[i+1], hatch='/', alpha=alpha)

ax[1].legend()

ax[1].set_ylim(1e-11, 1e-4)
ax[1].set_yscale('log')
ax[1].set_xlabel(r"$z$")
ax[1].set_ylabel(r"$d\Phi_{\rm QSO} / dz~[\rm{Mpc}^{-3}]$")

plt.tight_layout()
plt.show()


# # Binary Quasar Population Model
#
# Next let's fit an analytic model to the binary quasar population. We can do this in the discrete redshift bins first. Afterwards we can also fit a full model similar to the Shen+ 2020 QLF. We'll use the maximum likelihood method of Sandage, Tammann, and Yahil (STY, 1979), which tries to maximize the probability, $p_{i}(L_{i})$, that each binary quasar $i$ in the sample could be generated from the luminosity function, $\phi(L)$, where
# $$
# p_{i}(L_{i}) \propto \frac{\phi(L_{i})}{\int_{L_{\min}(z_{i})}^{L_{\max}} \phi(L) dL},
# $$
# and $L_{\min}(z)$ is the $z$ dependent luminosity completeness function, as before. We want to write down the log-likelihood, $\ln \mathcal{L}$ of observing the binary quasars in our sample (or in our binned subsamples), given $\phi(L_{i}; \vec{\theta})$, where $\vec{\theta}$ are parameters defining $\phi(L)$. We write $\ln \mathcal{L}$ as
# $$
# \ln \mathcal{L}_{j} = \sum_{i = 0}^{N_{\rm{BQ}, j}} \ln \phi(L_{i}) - \ln \left(\int_{L_{\min}(z_{i})}^{L_{\max}} \phi(L) dL\right)
# $$

# ## $1 / V_{\max}$

# In[17]:


from astropy.cosmology import WMAP9 as cosmo
import astropy.units as u

def v_max(log_l, z, z_min, z_max, z_complete=Z_MAX, completeness=1, cosmo=cosmo):
    """Calculate the maximum volume a source could be in and still be part of the sample."""
    if callable(z_complete):
        z_max = np.minimum(z_max, z_complete(log_l))
    else:
        z_max = np.minimum(z_max, z_complete)

    v_high = cosmo.comoving_volume(z_max).to(u.Mpc ** 3).value
    v_low = cosmo.comoving_volume(z_min).to(u.Mpc ** 3).value
    try:
        res = completeness * (v_high - v_low)
    except ValueError as e:
        print(np.shape(z_min))
        print(np.shape(z_max))
        print(np.shape(v_high))
        print(np.shape(v_low))
        print(np.shape(completeness))
        raise e
    # res = completeness * cosmo.differential_comoving_volume(z).to(u.Mpc ** 3 / u.sr).value * 4 * np.pi
    return res


def v_max_luminosity_function(log_l_vals, z_vals, log_l_bins, z_bins,
                              z_complete=Z_MAX, completeness=1, cosmo=cosmo, weights=None):
    """Calcualte the 1/Vmax luminosity function."""
    if weights is None:
        weights = np.ones_like(z_vals)

    # get bin mins and maxes
    z_bins_min = z_bins[:-1]
    z_bins_max = z_bins[1:]
    z_bw = z_bins_max[0] - z_bins_min[0]

    # get the bin indexes for both axes
    # also get min and max bins for each binary quasar candidate
    sample_z_bin_idxs = np.digitize(z_vals, bins=z_bins) - 1
    # sample_z_min = np.zeros_like(sample_z_bin_idxs)
    sample_z_min = z_bins_min[sample_z_bin_idxs]
    sample_z_max = z_bins_max[sample_z_bin_idxs]

    # calculate the max volume for each source
    vm = v_max(log_l_vals, z_vals, sample_z_min, sample_z_max, z_complete, completeness=completeness / weights) #/ z_bw
    # vm = vm * np.log(10) #* 10 ** log_l_vals
    # vm = v_max(log_l_vals, z_vals, z_bins[0], z_bins[-1], z_complete, completeness=completeness / weights)
    vm_tot = v_max(log_l_vals, z_vals, z_bins[0], z_bins[-1], z_complete, completeness=completeness / weights)


    # vm_bin = v_max(LOG_L_MIN, Z_MIN, z_bins_min, z_bins_max, z_bins_max, completeness=completeness)  # volume in each bin
    # vm_tot = np.sum(vm_bin)

    # calculate the 2D luminosity function
    qlf_vmax, _, __ = np.histogram2d(
        log_l_vals,
        z_vals,
        bins=[log_l_bins, z_bins],
        weights=1 / vm,
    )

    try:
        # qlf_vmax_log_l, _ = np.histogram(log_l_vals, bins=log_l_bins, weights=np.ones_like(log_l_vals)/vm_tot)
        # qlf_vmax_log_l, _ = np.histogram(log_l_vals, bins=log_l_bins, weights=1/vm)
        # qlf_vmax_z, _ = np.histogram(z_vals, bins=z_bins, weights=1/vm)
        qlf_vmax_log_l = np.sum(qlf_vmax, axis=1) #/ np.log(10)
        qlf_vmax_z = np.sum(qlf_vmax, axis=0) #* np.log(10)
    except Exception as e:
        print(np.shape(vm_tot))
        raise e

    return qlf_vmax, qlf_vmax_log_l, qlf_vmax_z


# In[18]:


# we'll also need to calculate their errors
# TODO: move to utils
# import warnings

def upper_poisson_errors(n, S=1):
    """Upper poisson errors, valid for small number statistics.

    """
    return np.squeeze(np.array(S * np.sqrt(n + .75) + (((S ** 2) + 3) / 4)))


def lower_poisson_errors(n, S=1):
    """Lower poisson errors, valid for small number statistics.

    """
    # warnings.filterwarnings("error")
    res = np.zeros_like(n)
    m = n != 0
    res[m] = S * np.sqrt(n[m] - .25) - (((S ** 2) - 1) / 4)

    return np.squeeze(res)

def upper_poisson_limit(n, S=1):
    """Upper poisson limit, valid for small number statistics.

    """
    return n + upper_poisson_errors(n, S=S)


def lower_poisson_limit(n, S=1):
    """Lower poisson limit, valid for small number statistics.

    """
    return n - lower_poisson_errors(n, S=S)

def qlf_vmax_errs(log_l_vals, z_vals, log_l_bins, z_bins,
                  z_complete=Z_MAX, completeness=1, cosmo=cosmo, weights=None):
    """Calcualte the 1/Vmax luminosity function."""
    if weights is None:
        weights = np.ones_like(z_vals)

    # get bin mins and maxes
    z_bins_min = z_bins[:-1]
    z_bins_max = z_bins[1:]

    # get the bin indexes for both axes
    # also get min and max bins for each binary quasar candidate
    sample_z_bin_idxs = np.digitize(z_vals, bins=z_bins) - 1
    sample_z_min = z_bins_min[sample_z_bin_idxs]
    sample_z_max = z_bins_max[sample_z_bin_idxs]

    # calculate the max volume for each source
    vm = v_max(log_l_vals, z_vals, sample_z_min, sample_z_max, z_complete, completeness=completeness / weights)
    vm_bin = v_max(LOG_L_MIN, Z_MIN, z_bins_min, z_bins_max,
                   z_complete=z_bins_max, completeness=completeness)  # volume in each bin
    vm_tot = np.sum(vm_bin)  # total volume

    # calculate effective weights for each bin
    sqtot, _, __ = np.histogram2d(log_l_vals, z_vals,
                                  bins=[log_l_bins, z_bins], weights=1/vm**2)
    tot, _, __ = np.histogram2d(log_l_vals, z_vals,
                                bins=[log_l_bins, z_bins], weights=1/vm)

    sqtot = np.squeeze(sqtot)
    tot = np.squeeze(tot)

    sqtot_log_l = np.squeeze(np.sum(sqtot, axis=1))
    tot_log_l = np.squeeze(np.sum(tot, axis=1))

    sqtot_z = np.squeeze(np.sum(sqtot, axis=0))
    tot_z = np.squeeze(np.sum(tot, axis=0))

    # calculate upper limits for bins with no counts
    m = tot != 0
    m_log_l = tot_log_l != 0
    m_z = tot_z != 0

    w_eff = np.ones_like(tot) / vm_bin
    w_eff[m] = sqtot[m] / tot[m]
    # w_eff = np.squeeze(w_eff)

    w_eff_log_l = np.ones_like(tot_log_l) / vm_tot
    w_eff_log_l[m_log_l] = sqtot_log_l[m_log_l] / tot_log_l[m_log_l]

    w_eff_z = np.ones_like(tot_z) / vm_bin
    w_eff_z[m_z] = sqtot_z[m_z] / tot_z[m_z]

    # calculate the effective number of samples in each bin
    n_eff = np.zeros_like(tot)
    n_eff[m] = tot[m] / w_eff[m]
    # n_eff = np.squeeze(n_eff)

    n_eff_log_l = np.zeros_like(tot_log_l)
    n_eff_log_l[m_log_l] = tot_log_l[m_log_l] / w_eff_log_l[m_log_l]

    n_eff_z = np.zeros_like(tot_z)
    n_eff_z[m_z] = tot_z[m_z] / w_eff_z[m_z]

    sigma_u = - tot + w_eff * upper_poisson_limit(n_eff, S=1)
    sigma_l = tot - w_eff * lower_poisson_limit(n_eff, S=1)

    sigma_u_log_l = - tot_log_l + w_eff_log_l * upper_poisson_limit(n_eff_log_l, S=1)
    sigma_l_log_l = tot_log_l - w_eff_log_l * lower_poisson_limit(n_eff_log_l, S=1)

    sigma_u_z = - tot_z + w_eff_z * upper_poisson_limit(n_eff_z, S=1)
    sigma_l_z = tot_z - w_eff_z * lower_poisson_limit(n_eff_z, S=1)

    sigma = np.array([sigma_l, sigma_u])
    sigma_log_l = np.array([sigma_l_log_l, sigma_u_log_l])
    sigma_z = np.array([sigma_l_z, sigma_u_z])

    return sigma, sigma_log_l, sigma_z


# ### Check: Quasar Population
#
# As a sanity check, let's compare this to our mock quasar population. That will tell us if we're getting results that make sense.

# In[19]:


qso_pop_binnable = qso_pop[qso_pop['log_l_bol'] >= LOG_L_MIN]
qso_frac = .05
qso_pop_binnable = qso_pop_binnable.sample(frac=qso_frac, replace=False)

qso_vmax, qso_vmax_log_l, qso_vmax_z = v_max_luminosity_function(
    log_l_vals=qso_pop_binnable['log_l_bol'],
    z_vals=qso_pop_binnable['z'],
    log_l_bins=log_l_bins,
    z_bins=z_bins,
    z_complete=Z_MAX,
    completeness=qso_frac,
    cosmo=cosmo
)

qso_vmax_err, qso_vmax_log_l_err, qso_vmax_z_err = qlf_vmax_errs(
    log_l_vals=qso_pop_binnable['log_l_bol'],
    z_vals=qso_pop_binnable['z'],
    log_l_bins=log_l_bins,
    z_bins=z_bins,
    z_complete=Z_MAX,
    completeness=qso_frac,
    cosmo=cosmo
)


# In[20]:


# replace
ul_mask = qso_vmax == 0
# crts_vmax[ul_mask] = crts_vmax_err[1][ul_mask]

ul_mask_log_l = qso_vmax_log_l == 0
# crts_vmax_log_l[ul_mask_log_l] = crts_vmax_log_l_err[1][ul_mask_log_l]

ul_mask_z = qso_vmax_z == 0
# crts_vmax_z[ul_mask_z] = crts_vmax_z_err[1][ul_mask_z]


# In[21]:


# adjust figsize for the number of columns we have
figsize = [2 * plt.rcParams['figure.figsize'][0],
           plt.rcParams['figure.figsize'][1]]
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

alpha = .25

ax[0].plot(log_l_range, qlf_log_l_q[1] * np.log(10), color=cs[0], label='Total')
ax[0].fill_between(log_l_range, qlf_log_l_q[0] * np.log(10), qlf_log_l_q[2] * np.log(10),
                   color=cs[0], alpha=alpha)
ax[0].fill_between(log_l_range, qlf_log_l_q[0] * np.log(10), qlf_log_l_q[2] * np.log(10),
                   facecolor="none", edgecolor=cs[0], hatch='/', alpha=alpha)

ax[0].errorbar(log_l_bin_mid,
               qso_vmax_log_l, # / np.log(10),
               # xerr=log_l_bw/2,
               yerr=qso_vmax_log_l_err,
               linestyle='none',
               marker='o',
               capsize=5)

for i, (z_min, z_max) in enumerate(zip(z_bin_min, z_bin_max)):
    ax[0].plot(log_l_range, qlf_log_l_bins_q[1, :, i] * np.log(10), color=cs[i+1], linestyle='--',
               label=r"${0} \leq z < {1}$".format(z_min, z_max))
    ax[0].fill_between(log_l_range, qlf_log_l_bins_q[0, :, i] * np.log(10), qlf_log_l_bins_q[2, :, i] * np.log(10),
                       color=cs[i+1], alpha=alpha)
    ax[0].fill_between(log_l_range, qlf_log_l_bins_q[0, :, i] * np.log(10), qlf_log_l_bins_q[2, :, i] * np.log(10),
                       facecolor="none", edgecolor=cs[i+1], hatch='/', alpha=alpha)

    ax[0].errorbar(log_l_bin_mid,
                   qso_vmax[:, i], # / np.log(10),
                   # xerr=log_l_bw/2,
                   yerr=qso_vmax_err[:, :, i],
                   linestyle='none',
                   color=cs[i+1],
                   marker='o',
                   capsize=5)

ax[0].set_ylim(1e-11, 1e-4)
ax[0].legend()
ax[0].set_yscale('log')
ax[0].set_xlabel(r"$\log\left(L_{\rm bol}~[\rm{erg}~\rm{s}^{-1}]\right)$")
ax[0].set_ylabel(r"$d\Phi_{\rm QSO} / d\log L_{\rm bol}~[\log L_{\rm bol}^{-1}~\rm{Mpc}^{-3}]$")

ax[1].plot(z_range, qlf_z_q[1], color=cs[0], label='Total')
ax[1].fill_between(z_range, qlf_z_q[0], qlf_z_q[2],
                   color=cs[0], alpha=alpha)
ax[1].fill_between(z_range, qlf_z_q[0], qlf_z_q[2],
                   facecolor="none", edgecolor=cs[0], hatch='/', alpha=alpha)

ax[1].errorbar(z_bin_mid,
               qso_vmax_z,
               # xerr=z_bw/2,
               yerr=qso_vmax_z_err,
               linestyle='none',
               marker='o',
               capsize=5)

for i, (log_l_min, log_l_max) in enumerate(zip(log_l_bin_min, log_l_bin_max)):
    ax[1].plot(z_range, qlf_z_bins_q[1, :, i], color=cs[i+1], linestyle='--',
               label=r"${0} \leq \log L_{{\rm bol}} < {1}$".format(log_l_min, log_l_max))
    ax[1].fill_between(z_range, qlf_z_bins_q[0, :, i], qlf_z_bins_q[2, :, i],
                       color=cs[i+1], alpha=alpha)
    ax[1].fill_between(z_range, qlf_z_bins_q[0, :, i], qlf_z_bins_q[2, :, i],
                       facecolor="none", edgecolor=cs[i+1], hatch='/', alpha=alpha)

    ax[1].errorbar(z_bin_mid,
                   qso_vmax[i, :],
                   # xerr=z_bw/2,
                   yerr=qso_vmax_err[:, i, :],
                   linestyle='none',
                   color=cs[i+1],
                   marker='o',
                   capsize=5)

ax[1].legend()

ax[1].set_ylim(1e-11, 1e-4)
ax[1].set_yscale('log')
ax[1].set_xlabel(r"$z$")
ax[1].set_ylabel(r"$d\Phi_{\rm QSO} / dz~[\rm{Mpc}^{-3}]$")

plt.tight_layout()
plt.show()


# ### CRTS Only Sample

# In[22]:


crts_pop_binnable = crts_pop[crts_pop['log_l_bol'] >= LOG_L_MIN]

crts_vmax, crts_vmax_log_l, crts_vmax_z = v_max_luminosity_function(
    log_l_vals=crts_pop_binnable['log_l_bol'],
    z_vals=crts_pop_binnable['z'],
    log_l_bins=log_l_bins,
    z_bins=z_bins,
    z_complete=z_max_fn,
    completeness=S_sky*S_qual*S_f,
    weights=crts_pop_binnable['weight'],
    # weights=.05,
    cosmo=cosmo
)

crts_vmax_err, crts_vmax_log_l_err, crts_vmax_z_err = qlf_vmax_errs(
    log_l_vals=crts_pop_binnable['log_l_bol'],
    z_vals=crts_pop_binnable['z'],
    log_l_bins=log_l_bins,
    z_bins=z_bins,
    z_complete=z_max_fn,
    completeness=S_sky*S_qual*S_f,
    weights=crts_pop_binnable['weight'],
    # weights=.05,
    cosmo=cosmo
)


# In[23]:


# replace
ul_mask = crts_vmax == 0
# crts_vmax[ul_mask] = crts_vmax_err[1][ul_mask]

ul_mask_log_l = crts_vmax_log_l == 0
# crts_vmax_log_l[ul_mask_log_l] = crts_vmax_log_l_err[1][ul_mask_log_l]

ul_mask_z = crts_vmax_z == 0
# crts_vmax_z[ul_mask_z] = crts_vmax_z_err[1][ul_mask_z]


# In[24]:


# adjust figsize for the number of columns we have
figsize = [2 * plt.rcParams['figure.figsize'][0],
           plt.rcParams['figure.figsize'][1]]
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

alpha = .25

ax[0].plot(log_l_range, qlf_log_l_q[1], color=cs[0], label='Total')
ax[0].fill_between(log_l_range, qlf_log_l_q[0], qlf_log_l_q[2],
                   color=cs[0], alpha=alpha)
ax[0].fill_between(log_l_range, qlf_log_l_q[0], qlf_log_l_q[2],
                   facecolor="none", edgecolor=cs[0], hatch='/', alpha=alpha)

ax[0].errorbar(log_l_bin_mid, crts_vmax_log_l, crts_vmax_log_l_err, linestyle='none',
               marker='o', markersize=5)

# for i, (z_min, z_max) in enumerate(zip(z_bin_min, z_bin_max)):
#     ax[0].plot(log_l_range, qlf_log_l_bins_q[1, :, i], color=cs[i+1], linestyle='--',
#                label=r"${0} \leq z < {1}$".format(z_min, z_max))
#     ax[0].fill_between(log_l_range, qlf_log_l_bins_q[0, :, i], qlf_log_l_bins_q[2, :, i],
#                        color=cs[i+1], alpha=alpha)
#     ax[0].fill_between(log_l_range, qlf_log_l_bins_q[0, :, i], qlf_log_l_bins_q[2, :, i],
#                        facecolor="none", edgecolor=cs[i+1], hatch='/', alpha=alpha)

#     x_shift = (1 + i) * .02
#     ax[0].errorbar(log_l_bin_mid + x_shift, crts_vmax[:, i], crts_vmax_err[:, :, i], linestyle='none',
#                    color=cs[i+1], marker='o', markersize=5)

ax[0].legend()
ax[0].set_yscale('log')
ax[0].set_xlabel(r"$\log\left(L_{\rm bol}~[\rm{erg}~\rm{s}^{-1}]\right)$")
ax[0].set_ylabel(r"$d\Phi_{\rm QSO} / d\log L_{\rm bol}~[\log L_{\rm bol}^{-1}~\rm{Mpc}^{-3}]$")

ax[1].plot(z_range, qlf_z_q[1], color=cs[0], label='Total')
ax[1].fill_between(z_range, qlf_z_q[0], qlf_z_q[2],
                   color=cs[0], alpha=alpha)
ax[1].fill_between(z_range, qlf_z_q[0], qlf_z_q[2],
                   facecolor="none", edgecolor=cs[0], hatch='/', alpha=alpha)

ax[1].errorbar(z_bin_mid, crts_vmax_z, crts_vmax_z_err, linestyle='none',
               marker='o', markersize=5)

# for i, (log_l_min, log_l_max) in enumerate(zip(log_l_bin_min, log_l_bin_max)):
#     ax[1].plot(z_range, qlf_z_bins_q[1, :, i], color=cs[i+1], linestyle='--',
#                label=r"${0} \leq \log L_{{\rm bol}} < {1}$".format(log_l_min, log_l_max))
#     ax[1].fill_between(z_range, qlf_z_bins_q[0, :, i], qlf_z_bins_q[2, :, i],
#                        color=cs[i+1], alpha=alpha)
#     ax[1].fill_between(z_range, qlf_z_bins_q[0, :, i], qlf_z_bins_q[2, :, i],
#                        facecolor="none", edgecolor=cs[i+1], hatch='/', alpha=alpha)

#     x_shift = (1 + i) * .01
#     ax[1].errorbar(z_bin_mid + x_shift, crts_vmax[i, :], crts_vmax_err[:, i, :], linestyle='none',
#                    color=cs[i+1], marker='o', markersize=5)

ax[1].legend()

ax[1].set_yscale('log')
ax[1].set_xlabel(r"$z$")
ax[1].set_ylabel(r"$d\Phi_{\rm QSO} / dz~[\rm{Mpc}^{-3}]$")

plt.tight_layout()
plt.show()


# ### Extrapolated Sample

# In[25]:


ext_pop_binnable = bq_pop[bq_pop['log_l_bol'] >= LOG_L_MIN]

ext_vmax, ext_vmax_log_l, ext_vmax_z = v_max_luminosity_function(
    log_l_vals=ext_pop_binnable['log_l_bol'],
    z_vals=ext_pop_binnable['z'],
    log_l_bins=log_l_bins,
    z_bins=z_bins,
    z_complete=Z_MAX,
    completeness=S_sky*S_qual*S_f,
    cosmo=cosmo
)

ext_vmax_err, ext_vmax_log_l_err, ext_vmax_z_err = qlf_vmax_errs(
    log_l_vals=ext_pop_binnable['log_l_bol'],
    z_vals=ext_pop_binnable['z'],
    log_l_bins=log_l_bins,
    z_bins=z_bins,
    z_complete=Z_MAX,
    completeness=S_sky*S_qual*S_f,
    cosmo=cosmo
)


# In[26]:


# replace
ul_mask = ext_vmax == 0
# crts_vmax[ul_mask] = crts_vmax_err[1][ul_mask]

ul_mask_log_l = ext_vmax_log_l == 0
# crts_vmax_log_l[ul_mask_log_l] = crts_vmax_log_l_err[1][ul_mask_log_l]

ul_mask_z = ext_vmax_z == 0
# crts_vmax_z[ul_mask_z] = crts_vmax_z_err[1][ul_mask_z]


# In[27]:


# adjust figsize for the number of columns we have
figsize = [2 * plt.rcParams['figure.figsize'][0],
           plt.rcParams['figure.figsize'][1]]
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

alpha = .25

ax[0].plot(log_l_range, qlf_log_l_q[1], color=cs[0], label='Total')
ax[0].fill_between(log_l_range, qlf_log_l_q[0], qlf_log_l_q[2],
                   color=cs[0], alpha=alpha)
ax[0].fill_between(log_l_range, qlf_log_l_q[0], qlf_log_l_q[2],
                   facecolor="none", edgecolor=cs[0], hatch='/', alpha=alpha)

ax[0].errorbar(log_l_bin_mid, ext_vmax_log_l, ext_vmax_log_l_err, linestyle='none',
               marker='o', markersize=5)

for i, (z_min, z_max) in enumerate(zip(z_bin_min, z_bin_max)):
    ax[0].plot(log_l_range, qlf_log_l_bins_q[1, :, i], color=cs[i+1], linestyle='--',
               label=r"${0} \leq z < {1}$".format(z_min, z_max))
    ax[0].fill_between(log_l_range, qlf_log_l_bins_q[0, :, i], qlf_log_l_bins_q[2, :, i],
                       color=cs[i+1], alpha=alpha)
    ax[0].fill_between(log_l_range, qlf_log_l_bins_q[0, :, i], qlf_log_l_bins_q[2, :, i],
                       facecolor="none", edgecolor=cs[i+1], hatch='/', alpha=alpha)

    x_shift = (1 + i) * .02
    ax[0].errorbar(log_l_bin_mid, ext_vmax[:, i], ext_vmax_err[:, :, i], linestyle='none',
                   color=cs[i+1], marker='o', markersize=5)

ax[0].legend()
ax[0].set_yscale('log')
ax[0].set_xlabel(r"$\log\left(L_{\rm bol}~[\rm{erg}~\rm{s}^{-1}]\right)$")
ax[0].set_ylabel(r"$d\Phi_{\rm QSO} / d\log L_{\rm bol}~[\log L_{\rm bol}^{-1}~\rm{Mpc}^{-3}]$")

ax[1].plot(z_range, qlf_z_q[1], color=cs[0], label='Total')
ax[1].fill_between(z_range, qlf_z_q[0], qlf_z_q[2],
                   color=cs[0], alpha=alpha)
ax[1].fill_between(z_range, qlf_z_q[0], qlf_z_q[2],
                   facecolor="none", edgecolor=cs[0], hatch='/', alpha=alpha)

ax[1].errorbar(z_bin_mid, ext_vmax_z, ext_vmax_z_err, linestyle='none',
               marker='o', markersize=5)

for i, (log_l_min, log_l_max) in enumerate(zip(log_l_bin_min, log_l_bin_max)):
    ax[1].plot(z_range, qlf_z_bins_q[1, :, i], color=cs[i+1], linestyle='--',
               label=r"${0} \leq \log L_{{\rm bol}} < {1}$".format(log_l_min, log_l_max))
    ax[1].fill_between(z_range, qlf_z_bins_q[0, :, i], qlf_z_bins_q[2, :, i],
                       color=cs[i+1], alpha=alpha)
    ax[1].fill_between(z_range, qlf_z_bins_q[0, :, i], qlf_z_bins_q[2, :, i],
                       facecolor="none", edgecolor=cs[i+1], hatch='/', alpha=alpha)

    x_shift = (1 + i) * .01
    ax[1].errorbar(z_bin_mid, ext_vmax[i, :], ext_vmax_err[:, i, :], linestyle='none',
                   color=cs[i+1], marker='o', markersize=5)

ax[1].legend()

ax[1].set_yscale('log')
ax[1].set_xlabel(r"$z$")
ax[1].set_ylabel(r"$d\Phi_{\rm QSO} / dz~[\rm{Mpc}^{-3}]$")

plt.tight_layout()
plt.show()


# ## Double Power Law Fit

# In[28]:


import h5py

from scipy.interpolate import interp1d

import astropy.units as u
import astropy.constants as const
from astropy.cosmology import WMAP9 as cosmo

import emcee
import corner
import multiprocess as mp

from IPython.display import display, Math, clear_output


# In[29]:


N_MCMC_SAMPLES = 4_000
CONT = False


# In[30]:


import sys
sys.path.insert(0, '../')

from src.utils import nquad_vec

from src.models.qlf import SchechterQLF, DoublePowerLawQLF
from src.models.qlf import LOG_L_SOLAR


# In[31]:


from src.models.qlf import Shen2020QLF


# In[32]:


dist = np.array([])
crts_based_test = True
ul_test = True
for i in range(100000):
    if crts_based_test:
        n_crts = np.random.poisson(len(crts_pop))
        if ul_test:
            n_crts = np.random.randint(n_crts)
        N_BH = np.sum(bq_pop['weight']) * n_crts / len(crts_pop)
    else:
        N_BH = np.sum(bq_pop['weight'])
        N_BH = np.random.poisson(N_BH)
        if ul_test:
            N_BH = np.random.randint(N_BH)

    if N_BH == 0:
        N_BH = .05

    dist = np.append(dist, N_BH)

display(Math(r'95\% UL = {0:.0f}'.format(np.quantile(dist, q=.95))))
display(Math(r'$\sum_{{i}} w_{{i}} = {0:.0f}$'.format(np.sum(bq_pop['weight']))))

plt.figure()

plt.hist(dist)
plt.axvline(np.sum(bq_pop['weight']), color='k', ls='--')

plt.xlabel(r"$N'_{\rm BQ}$")

plt.tight_layout()

plt.savefig('../reports/figures/upper_limit_theory.pdf')
plt.show()


# In[33]:


def fit_model(log_l, z, *theta):
    a0, a1, a2, b0, b1, b2, c0, c1, c2, d0, d1 = theta
    return MODEL(
        log_l, z,
        a0=a0, a1=a1, a2=a2,
        b0=b0, b1=b1, b2=b2,
        c0=c0, c1=c1, c2=c2,
        d0=d0, d1=d1
    )


def n_integ(log_l, z, *theta):
    a0, a1, a2, b0, b1, b2, c0, c1, c2, d1 = theta
    res = MODEL(
        log_l, z,
        a0=a0, a1=a1, a2=a2,
        b0=b0, b1=b1, b2=b2,
        c0=c0, c1=c1, c2=c2,
        d0=np.zeros_like(d1), d1=d1
    )
    res = res * cosmo.differential_comoving_volume(z).to(u.Mpc ** 3 / u.sr).value
    res = res * 4 * np.pi
    return res


def log_normalization(theta, completeness_fn, cosmo=cosmo, sky_coverage=1, ul=False, crts_based=False):
    # log_break, exp = theta
    # theta = np.hstack((np.zeros(len(theta))[..., np.newaxis], theta))
    # theta = np.transpose(theta)

    # assume poisson errors on the number. if it's based on CRTS, match CRTS
    if crts_based:
        n_crts = np.random.poisson(len(crts_pop))
        if ul:
            n_crts = np.random.randint(n_crts)
        N_BH = np.sum(WEIGHTS) * n_crts / len(crts_pop)
    else:
        N_BH = np.sum(WEIGHTS)
        N_BH = np.random.poisson(N_BH)
        if ul:
            N_BH = np.random.randint(N_BH)

    if N_BH == 0:
        N_BH = .05  # assign a small non-zero value so we don't encounter errors with infinities later

    if callable(completeness_fn):
        log_l_min = completeness_fn(Z_VALS)
        log_l_min = np.maximum(LOG_L_MIN, log_l_min)

        denom = np.transpose([nquad_vec(n_integ, [[llm, LOG_L_MAX], [Z_MIN, Z_MAX]], args=theta) for llm in log_l_min])
        # res = np.sum(WEIGHTS / denom, axis=-1)
        res = N_BH / np.sum(denom, axis=-1)
    else:
        log_l_min = completeness_fn
        log_l_min = np.maximum(LOG_L_MIN, log_l_min)
        denom = nquad_vec(n_integ, [[log_l_min, LOG_L_MAX], [Z_MIN, Z_MAX]], args=theta)
        # res = np.sum(WEIGHTS / denom[..., np.newaxis], axis=-1)
        res = N_BH / denom

    res = res / sky_coverage

    # print("res = {0}".format(res))

    # res = np.where(res == 0, np.random.uniform(-11, -10), np.log10(np.squeeze(res)))
    return np.log10(res)


def log_probability(theta, bounds, completeness_fn, cosmo=cosmo, sky_coverage=1, ul=False, crts_based=False):
    # first check basic bounds
    a0, a1, a2, b0, b1, b2, c0, c1, c2, d1 = theta

    a0_min, a1_min, a2_min, b0_min, b1_min, b2_min, c0_min, c1_min, c2_min, d0_min, d1_min = bounds[0]
    a0_max, a1_max, a2_max, b0_max, b1_max, b2_max, c0_max, c1_max, c2_max, d0_max, d1_max = bounds[1]
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
        # and np.all(d0_min <= d0 <= d0_max)  # this check happens before d0 is calculated
        and np.all(d1_min <= d1 <= d1_max)
    ):
        return -np.inf, -np.inf
    # also calculate the log normalization
    # uses the blobs interface of emcee
    d0 = log_normalization(theta, completeness_fn,
                           cosmo=cosmo, sky_coverage=sky_coverage, ul=ul, crts_based=crts_based)

    theta2 = a0, a1, a2, b0, b1, b2, c0, c1, c2, d0, d1

    lp = log_prior(theta2, bounds)
    if not np.isfinite(lp):
        return -np.inf, d0

    ll = log_likelihood(theta2, completeness_fn)

    # print(np.shape(d0))

    return lp + ll, d0


def log_prior(theta, param_bounds):
    a0, a1, a2, b0, b1, b2, c0, c1, c2, d0, d1 = theta

    a0_min, a1_min, a2_min, b0_min, b1_min, b2_min, c0_min, c1_min, c2_min, d0_min, d1_min = param_bounds[0]
    a0_max, a1_max, a2_max, b0_max, b1_max, b2_max, c0_max, c1_max, c2_max, d0_max, d1_max = param_bounds[1]

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
        ls = MODEL.low_slope(z_dummy, np.transpose([a0, a1, a2]))
        hs = MODEL.high_slope(z_dummy, a0=b0, a1=b1, a2=b2)
        lb = MODEL.log_break(z_dummy, a0=c0, a1=c1, a2=c2)
        ln = MODEL.log_norm(z_dummy, coef=np.transpose([d0, d1]))

        if (
            np.all(-5 <= ls <= 5)
            # and np.all(ls <= 5)
            and np.all(-5 <= hs <= 5)
            # and np.all(hs <= 5)
            and np.all(ls <= hs)
            and np.all(5 <= lb <= 20)
            # and np.all(lb <= 20)
            and np.all(-15 <= ln <= 5)
            # and np.all(ln <= 5)
        ):
            return 0.0
    return -np.inf


def log_likelihood(theta, completeness_fn):
    mdl = fit_model(LOG_L_VALS, Z_VALS, *theta)

    m = mdl == 0  # find anywhere the model returns 0 due to numerical limits
    mdl[m] = 1  # temporarily replace it
    ln_num = np.log(mdl)  # calculate log without errors
    ln_num[m] = -np.inf  # correct 0s

    # find the minimum luminosity
    log_l_min = completeness_fn(Z_VALS) if callable(completeness_fn) else completeness_fn
    log_l_min = np.maximum(LOG_L_MIN, log_l_min)

    # integrate over the model to get the denominator
    try:
        denom = [nquad_vec(fit_model, [[llm, LOG_L_MAX], [Z_MIN, Z_MAX]], n_roots=100, args=theta) for llm in log_l_min]
    except TypeError as e:
        denom = nquad_vec(fit_model, [[log_l_min, LOG_L_MAX], [Z_MIN, Z_MAX]], n_roots=100, args=theta)

    if np.any(denom == 0):
        print(denom)
        print(theta)
        print(log_l_min)
        raise ValueError("denom is 0")

    ln_denom = np.log(denom)

    # sum the log probabilities, weighted by their maximum precision estimates
    res = np.nansum(WEIGHTS * (ln_num - ln_denom))

    if np.isnan(res):
        print(ln_num)
        print(ln_denom)
        print(res)
        raise ValueError("NaN res")

    return res


# In[34]:


from scipy.stats import truncnorm

MODEL = Shen2020QLF()

def initialize_params(init_bounds, param_bounds, nwalkers=32, completeness_fn=LOG_L_MIN,
                      cosmo=cosmo, sky_coverage=1, ul=False, crts_based=False):

    pos_bounds = param_bounds[:, np.arange(len(param_bounds[0])) != 9]
    param_low, param_high = pos_bounds
    i_bounds = init_bounds[:, np.arange(len(init_bounds[0])) != 9]
    init_low, init_high = i_bounds
    ndim = len(init_low)

    # first make sure we have at least one initialization that's valid
    init_mean = (init_high + init_low) / 2
    init_diff = (init_high - init_low) / 2
    a, b = (param_low - init_mean) / init_diff, (param_high - init_mean) / init_diff
    # a = a[np.arange(len(a)) != 9]
    # b = b[np.arange(len(b)) != 9]
    # pos = np.random.uniform(low=init_low, high=init_high, size=(nwalkers, ndim))
    pos = truncnorm.rvs(a, b, loc=init_mean, scale=init_diff, size=(nwalkers, ndim))

    d0 = log_normalization(pos.T, completeness_fn=completeness_fn,
                           cosmo=cosmo, sky_coverage=sky_coverage, ul=ul, crts_based=crts_based)
    pos = np.insert(pos, -1, d0, axis=1)
    lp = [log_prior(p, param_bounds) for p in pos]
    invalid = np.isinf(lp)  # find invalid initializations
    valid = ~invalid
    while np.sum(valid) < nwalkers:  #
        # pos = np.random.uniform(low=low, high=high, size=(nwalkers, ndim))
        pos_tmp = truncnorm.rvs(a, b, loc=init_mean, scale=init_diff,
                                     size=(np.sum(invalid), ndim))
        d0_tmp = log_normalization(pos_tmp.T, completeness_fn,
                           cosmo=cosmo, sky_coverage=sky_coverage, ul=ul, crts_based=crts_based)
        pos_tmp = np.insert(pos_tmp, -1, d0_tmp, axis=1)

        pos[invalid] = pos_tmp
        lp = [log_prior(p, param_bounds) for p in pos]
        invalid = np.isinf(lp)  # find invalid initializations
        valid = ~invalid

    # replace = np.random.randint(np.sum(valid), size=np.sum(invalid))
    # pos[invalid] = pos[valid][replace]
    pos = pos[:, np.arange(len(pos[0])) != 9]
    return pos


def mcmc_sampler(log_prob, bounds, completeness_fn=LOG_L_MIN,
                 cosmo=cosmo, sky_coverage=1,
                 init_bounds=None, nwalkers=32, ncpus=mp.cpu_count(),
                 max_nsamples=10000, len_autocorr=50, filename=None, progress=True,
                 track_autocorr=False, live_plots=False, cont=True, ul=False, crts_based=False,
                 skip_initial_state_check=False
                ):
    args = (bounds, completeness_fn, cosmo, sky_coverage, ul)

    if init_bounds is None:
        init_bounds = bounds

    low, high = init_bounds
    ndim = len(low) - 1

    # n_walkers, n_dim = pos.shape

    if filename is not None:
        backend = emcee.backends.HDFBackend(filename)
        if not cont:
            backend.reset(nwalkers, ndim)
    else:
        backend = None

    if track_autocorr:
        autocorr_idx = np.array([0])
        autocorr = np.array([0])

        autocorr_fig, autocorr_ax = plt.subplots()

        convergence_line, = autocorr_ax.plot(autocorr_idx, autocorr_idx / len_autocorr, "--k")
        autocorr_line, = autocorr_ax.plot(autocorr_idx, autocorr)

        autocorr_ax.set_title("Autocorrelation")
        autocorr_ax.set_xlabel("number of steps")
        autocorr_ax.set_ylabel(r"mean $\hat{\tau}$")

        autocorr_ax.set_xlim(left=0)
        autocorr_ax.set_ylim(bottom=0)

    print("Starting MCMC...")

    if ncpus is not None and ncpus > 1:
        with mp.Pool(ncpus) as pool:
            sampler = emcee.EnsembleSampler(nwalkers,
                                            ndim,
                                            log_prob,
                                            args=args,
                                            backend=backend,
                                            pool=pool,
                                            moves=[
                                                (emcee.moves.DEMove(), .8),
                                                (emcee.moves.DESnookerMove(), .2)
                                            ],
                                           )

            if cont:
                pos = sampler._previous_state
            else:
                print("Initializing parameters...")
                pos = initialize_params(init_bounds, bounds, nwalkers=nwalkers, completeness_fn=completeness_fn,
                                        cosmo=cosmo, sky_coverage=sky_coverage, ul=ul, crts_based=crts_based)
                print("Done!")

            for sample in sampler.sample(pos, iterations=max_nsamples, progress=progress,
                                         skip_initial_state_check=skip_initial_state_check):
                # check convergence every 100 steps
                if sampler.iteration % 100:
                    continue


                if track_autocorr:
                    # check convergence every 100 steps
                    if sampler.iteration % 100:
                        continue

                    # compute current autocorr time
                    tau = sampler.get_autocorr_time(tol=0)

                    # Check convergence
                    converged = np.all(tau * len_autocorr < sampler.iteration)
                    converged &= np.all(np.abs(autocorr[-1] - tau) / tau < 0.02)

                    autocorr_idx = np.append(autocorr_idx, sampler.iteration)
                    autocorr = np.append(autocorr, np.nanmean(tau))

                    if converged:
                        break
    else:
        sampler = emcee.EnsembleSampler(nwalkers,
                                        ndim,
                                        log_prob,
                                        args=args,
                                        backend=backend,
                                        moves=[
                                            (emcee.moves.DEMove(), .8),
                                            (emcee.moves.DESnookerMove(), .2)
                                        ],
                                       )

        for sample in sampler.sample(pos, iterations=max_nsamples, progress=progress,
                                     skip_initial_state_check=skip_initial_state_check):
            # check convergence every 100 steps
            if sampler.iteration % 100:
                continue


            if track_autocorr:
                # check convergence every 100 steps
                if sampler.iteration % 100:
                    continue

                # compute current autocorr time
                tau = sampler.get_autocorr_time(tol=0)

                # Check convergence
                converged = np.all(tau * len_autocorr < sampler.iteration)
                converged &= np.all(np.abs(autocorr[-1] - tau) / tau < 0.02)

                autocorr_idx = np.append(autocorr_idx, sampler.iteration)
                autocorr = np.append(autocorr, np.nanmean(tau))

                if converged:
                    break

    if track_autocorr:
        autocorr_line.set_xdata(autocorr_idx)
        autocorr_line.set_ydata(autocorr)

        convergence_line.set_xdata(autocorr_idx)
        convergence_line.set_ydata(autocorr_idx / len_autocorr)

        autocorr_ax.set_xlim(left=0, right=np.nanmax(autocorr_idx))
        autocorr_ax.set_ylim(bottom=0,
                             top=np.nanmax(autocorr)
                             + .1 * (np.nanmax(autocorr) - np.nanmin(autocorr)))

        # drawing updated values
        # autocorr_fig.canvas.draw()
        # clear_output(wait=True)
        # display(autocorr_fig)
        # plt.close()
        plt.show()

        return sampler, autocorr, autocorr_idx
    else:
        return sampler


# ### Mock Quasar Sample

# In[35]:


# Define the allowed parameter space
DPL_A0_MIN, DPL_A0_MAX = .5, 3
DPL_A1_MIN, DPL_A1_MAX = -1, .5
DPL_A2_MIN, DPL_A2_MAX = -.5, .5

DPL_B0_MIN, DPL_B0_MAX = 2, 4
DPL_B1_MIN, DPL_B1_MAX = -5, 0
DPL_B2_MIN, DPL_B2_MAX = 0, 5

DPL_C0_MIN, DPL_C0_MAX = 44 - LOG_L_SOLAR, 48 - LOG_L_SOLAR
DPL_C1_MIN, DPL_C1_MAX = -1, 0
DPL_C2_MIN, DPL_C2_MAX = 0, 3

DPL_D0_MIN, DPL_D0_MAX = -15, 5  # irrelevant for the STY method
DPL_D1_MIN, DPL_D1_MAX = -1, 1


# In[36]:


# set up lower and upper bounds on each parameter
dpl_low = [
    DPL_A0_MIN, DPL_A1_MIN, DPL_A2_MIN,
    DPL_B0_MIN, DPL_B1_MIN, DPL_B2_MIN,
    DPL_C0_MIN, DPL_C1_MIN, DPL_C2_MIN,
    DPL_D0_MIN, DPL_D1_MIN
]
dpl_high = [
    DPL_A0_MAX, DPL_A1_MAX, DPL_A2_MAX,
    DPL_B0_MAX, DPL_B1_MAX, DPL_B2_MAX,
    DPL_C0_MAX, DPL_C1_MAX, DPL_C2_MAX,
    DPL_D0_MAX, DPL_D1_MAX
]
dpl_bounds = np.array([dpl_low, dpl_high])

dpl_low_init = [
    .8569-.0253, -.2614-.0164, .0200-.0011,
    2.5375-.0187, -1.0425-.0182, 1.1201-.0207,
    13.0088-.0091, -.5759-.0020, .4554-.0027,
    -3.5426-.0209, -.3936-.0073
]
dpl_high_init = [
    .8569+.0247, -.2614+.0162, .0200+.0011,
    2.5375+.0177, -1.0425+.0164, 1.1201+.0199,
    13.0088+.0090, -.5759+.0018, .4554+.0028,
    -3.5426+.0235, -.3936+.0070
]
dpl_init = np.array([dpl_low_init, dpl_high_init])


# In[37]:


labels = [
    r"$a_{0}$", r"$a_{1}$", r"$a_{2}$",
    r"$b_{0}$", r"$b_{1}$", r"$b_{2}$",
    r"$c_{0}$", r"$c_{1}$", r"$c_{2}$",
    r"$d_{0}$", r"$d_{1}$"
]


# In[38]:


# set up arrays to track values
# may be unnecessary for non-binned data
qso_frac = 1e-4
qso_pop_fit = qso_pop.sample(frac=qso_frac)
qso_pop_fit


# In[39]:


LOG_L_VALS = qso_pop_fit['log_l_bol'].values
Z_VALS = qso_pop_fit['z'].values
WEIGHTS = qso_pop_fit['weight'].values

filename = f"../data/processed/qso_dpl_qlf_chain.h5"

dpl_sampler, dpl_ac, dpl_idx = mcmc_sampler(
    log_probability,
    dpl_bounds,
    completeness_fn=LOG_L_MIN,
    cosmo=cosmo,
    sky_coverage=qso_frac,
    init_bounds=dpl_init,
    nwalkers=24,
    ncpus=12,
    max_nsamples=N_MCMC_SAMPLES,
    filename=filename,
    track_autocorr=True,
    live_plots=False,
    cont=CONT
)

try:
    tau = dpl_sampler.get_autocorr_time()
except:
    print("Run a longer chain!")
    tau = dpl_sampler.get_autocorr_time(tol=0)



# In[40]:


samples = dpl_sampler.get_chain()
log_norm = dpl_sampler.get_blobs()
dpl_samples = np.concatenate((samples, log_norm[..., np.newaxis]), axis=-1)
dpl_samples[...,[9, 10]] = dpl_samples[...,[10, 9]]

discard = int(3 * tau.max())
thin = 15
flat_samples = dpl_sampler.get_chain(discard=discard,
                                     thin=thin, flat=True)
flat_log_norm = dpl_sampler.get_blobs(discard=discard,
                                      thin=thin, flat=True)
flat_dpl_samples = np.concatenate((flat_samples, flat_log_norm[..., np.newaxis]), axis=-1)
flat_dpl_samples[...,[9, 10]] = flat_dpl_samples[...,[10, 9]]


# In[41]:


fig, axes = plt.subplots(len(labels), figsize=(10, 2 * len(labels) - 1), sharex=True)

for i in range(len(labels)):
    ax = axes[i]
    ax.plot(dpl_samples[..., i], "k", alpha=0.3)
    ax.set_xlim(0, len(dpl_samples))
    if i == 4 or i == 5:
        ax.set_ylim(DPL_B1_MIN, DPL_B2_MAX)
    elif i == 7 or i == 8:
        ax.set_ylim(DPL_C1_MIN, DPL_C2_MAX)
    else:
        ax.set_ylim(dpl_low[i], dpl_high[i])
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")
plt.show()

mcmc = np.quantile(flat_dpl_samples, [.16, .5, .84], axis=0)
q = np.diff(mcmc, axis=0)

for i in range(len(labels)):
    txt = r"{3} $= {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}$"
    txt = txt.format(mcmc[1, i], q[0, i], q[1, i], labels[i])
    display(Math(txt))

mcmc = np.quantile(flat_dpl_samples, [.16, .5, .84], axis=0)
fig = corner.corner(flat_dpl_samples, labels=labels,
                    quantiles=[.16, .5, .84], truths=mcmc[1,:])
# plt.savefig('../reports/figures/mcmc_schechter_lbol.pdf')
# plt.savefig('../reports/figures/mcmc_schechter_lbol.png')
plt.show()


# In[42]:


inds = np.random.choice(range(len(flat_dpl_samples)), size=1000, replace=False)
flat_dpl_samples = flat_dpl_samples[inds]
# dpl_flat_samples.append(flat_dpl_samples)

a0, a1, a2 = flat_dpl_samples[:, 0], flat_dpl_samples[:, 1], flat_dpl_samples[:, 2]
b0, b1, b2 = flat_dpl_samples[:, 3], flat_dpl_samples[:, 4], flat_dpl_samples[:, 5]
c0, c1, c2 = flat_dpl_samples[:, 6], flat_dpl_samples[:, 7], flat_dpl_samples[:, 8]
d0, d1 = flat_dpl_samples[:, 9], flat_dpl_samples[:, 10]

dpl_post = Shen2020QLF(
    a0=a0, a1=a1, a2=a2,
    b0=b0, b1=b1, b2=b2,
    c0=c0, c1=c1, c2=c2,
    d0=d0, d1=d1
)
with open(f"../models/qso_dpl_posterior.pkl", "wb") as f:
    pickle.dump(dpl_post, f)


# In[43]:


dpl_post_log_l_bins = np.empty((len(dpl_post), len(log_l_range), 0))
for z_min, z_max in zip(z_bin_min, z_bin_max):
    dpl_post_log_l_i = np.transpose([nquad_vec(lambda z: dpl_post(log_l, z),
                                          [[z_min, z_max]], n_roots=21)
                                for log_l in log_l_range])
    dpl_post_log_l_bins = np.append(dpl_post_log_l_bins, dpl_post_log_l_i[..., np.newaxis], axis=-1)

dpl_post_log_l_bins_q = np.quantile(dpl_post_log_l_bins, q=[.16, .5, .84], axis=0)

dpl_post_z_bins = np.empty((len(dpl_post), len(z_range), 0))
for log_l_min, log_l_max in zip(log_l_bin_min, log_l_bin_max):
    dpl_post_z_i = np.transpose([nquad_vec(lambda log_l: dpl_post(log_l, z),
                                          [[log_l_min, log_l_max]], n_roots=21)
                                for z in z_range])
    dpl_post_z_bins = np.append(dpl_post_z_bins, dpl_post_z_i[..., np.newaxis], axis=-1)

dpl_post_z_bins_q = np.quantile(dpl_post_z_bins, q=[.16, .5, .84], axis=0)

dpl_post_z = np.transpose([nquad_vec(lambda log_l: dpl_post(log_l, z), [[45, 48]], n_roots=21) for z in z_range])
dpl_post_z_q = np.quantile(dpl_post_z, q=[.16, .5, .84], axis=0)

dpl_post_log_l = np.transpose([nquad_vec(lambda z: dpl_post(log_l, z), [[0, 1.5]], n_roots=21) for log_l in log_l_range])
dpl_post_log_l_q = np.quantile(dpl_post_log_l, q=[.16, .5, .84], axis=0)


# In[44]:


# adjust figsize for the number of columns we have
figsize = [2 * plt.rcParams['figure.figsize'][0],
           plt.rcParams['figure.figsize'][1]]
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

alpha = .25

ax[0].plot(log_l_range, qlf_log_l_q[1], color=cs[0], label='QLF, Total', linestyle='-')
ax[0].fill_between(log_l_range, qlf_log_l_q[0], qlf_log_l_q[2],
                   color=cs[0], alpha=alpha)
ax[0].fill_between(log_l_range, qlf_log_l_q[0], qlf_log_l_q[2],
                   facecolor="none", edgecolor=cs[0], hatch='/', alpha=alpha)

ax[0].plot(log_l_range, dpl_post_log_l_q[1], color=cs[0], label='Test QLF, Total', linestyle='--')
ax[0].fill_between(log_l_range, dpl_post_log_l_q[0], dpl_post_log_l_q[2],
                   color=cs[0], alpha=alpha)
ax[0].fill_between(log_l_range, dpl_post_log_l_q[0], dpl_post_log_l_q[2],
                   facecolor="none", edgecolor=cs[0], hatch='/', alpha=alpha)

for i, (z_min, z_max) in enumerate(zip(z_bin_min, z_bin_max)):
    ax[0].plot(log_l_range, qlf_log_l_bins_q[1, :, i], color=cs[i+1], linestyle='-',
               label=r"QLF, ${0} \leq z < {1}$".format(z_min, z_max))
    ax[0].fill_between(log_l_range, qlf_log_l_bins_q[0, :, i], qlf_log_l_bins_q[2, :, i],
                       color=cs[i+1], alpha=alpha)
    ax[0].fill_between(log_l_range, qlf_log_l_bins_q[0, :, i], qlf_log_l_bins_q[2, :, i],
                       facecolor="none", edgecolor=cs[i+1], hatch='\\', alpha=alpha)

    ax[0].plot(log_l_range, dpl_post_log_l_bins_q[1, :, i], color=cs[i+1], linestyle='--',
               label=r"Test QLF, ${0} \leq z < {1}$".format(z_min, z_max)
              )
    ax[0].fill_between(log_l_range, dpl_post_log_l_bins_q[0, :, i], dpl_post_log_l_bins_q[2, :, i],
                       color=cs[i+1], alpha=alpha)
    ax[0].fill_between(log_l_range, dpl_post_log_l_bins_q[0, :, i], dpl_post_log_l_bins_q[2, :, i],
                       facecolor="none", edgecolor=cs[i+1], hatch='\\', alpha=alpha)

ax[0].legend()

ax[0].set_ylim(1e-11, 1e-4)
ax[0].set_yscale('log')
ax[0].set_xlabel(r"$\log\left(L_{\rm bol}~[\rm{erg}~\rm{s}^{-1}]\right)$")
ax[0].set_ylabel(r"$d\Phi_{\rm QSO} / d\log L_{\rm bol}~[\log L_{\rm bol}^{-1}~\rm{Mpc}^{-3}]$")

ax[1].plot(z_range, qlf_z_q[1], color=cs[0], label='QLF, Total', linestyle='-')
ax[1].fill_between(z_range, qlf_z_q[0], qlf_z_q[2],
                   color=cs[0], alpha=alpha)
ax[1].fill_between(z_range, qlf_z_q[0], qlf_z_q[2],
                   facecolor="none", edgecolor=cs[0], hatch='/', alpha=alpha)

ax[1].plot(z_range, dpl_post_z_q[1], color=cs[0],
           # label='Test QLF, Total',
           linestyle='--')
ax[1].fill_between(z_range, dpl_post_z_q[0], dpl_post_z_q[2],
                   color=cs[0], alpha=alpha)
ax[1].fill_between(z_range, dpl_post_z_q[0], dpl_post_z_q[2],
                   facecolor="none", edgecolor=cs[0], hatch='\\', alpha=alpha)

for i, (log_l_min, log_l_max) in enumerate(zip(log_l_bin_min, log_l_bin_max)):
    ax[1].plot(z_range, qlf_z_bins_q[1, :, i], color=cs[i+1], linestyle='-',
               label=r"QLF, ${0} \leq \log L_{{\rm bol}} < {1}$".format(log_l_min, log_l_max))
    ax[1].fill_between(z_range, qlf_z_bins_q[0, :, i], qlf_z_bins_q[2, :, i],
                       color=cs[i+1], alpha=alpha)
    ax[1].fill_between(z_range, qlf_z_bins_q[0, :, i], qlf_z_bins_q[2, :, i],
                       facecolor="none", edgecolor=cs[i+1], hatch='/', alpha=alpha)

    ax[1].plot(z_range, dpl_post_z_bins_q[1, :, i], color=cs[i+1], linestyle='--',
               # label=r"Test QLF, ${0} \leq \log L_{{\rm bol}} < {1}$".format(log_l_min, log_l_max)
              )
    ax[1].fill_between(z_range, dpl_post_z_bins_q[0, :, i], dpl_post_z_bins_q[2, :, i],
                       color=cs[i+1], alpha=alpha)
    ax[1].fill_between(z_range, dpl_post_z_bins_q[0, :, i], dpl_post_z_bins_q[2, :, i],
                       facecolor="none", edgecolor=cs[i+1], hatch='\\', alpha=alpha)

ax[1].legend()

ax[1].set_ylim(1e-11, 1e-4)
ax[1].set_yscale('log')
ax[1].set_xlabel(r"$z$")
ax[1].set_ylabel(r"$d\Phi_{\rm QSO} / dz~[\rm{Mpc}^{-3}]$")

plt.tight_layout()

plt.savefig('../reports/figures/qlf_fit_check.pdf')
plt.show()


# ### CRTS Sample

# In[45]:


# Define the allowed parameter space
CRTS_A0_MIN, CRTS_A0_MAX = -1.5, 1.5
CRTS_A1_MIN, CRTS_A1_MAX = -1, 2
CRTS_A2_MIN, CRTS_A2_MAX = -.5, .5

CRTS_B0_MIN, CRTS_B0_MAX = 0, 3
CRTS_B1_MIN, CRTS_B1_MAX = -1.5, 0
CRTS_B2_MIN, CRTS_B2_MAX = 0, 10

CRTS_C0_MIN, CRTS_C0_MAX = 44 - LOG_L_SOLAR, 48 - LOG_L_SOLAR
CRTS_C1_MIN, CRTS_C1_MAX = -1, 0
CRTS_C2_MIN, CRTS_C2_MAX = 0, 2

CRTS_D0_MIN, CRTS_D0_MAX = -15, 5  # irrelevant for the STY method
CRTS_D1_MIN, CRTS_D1_MAX = -1, .5


# In[46]:


# set up lower and upper bounds on each parameter
crts_low = [
    CRTS_A0_MIN, CRTS_A1_MIN, CRTS_A2_MIN,
    CRTS_B0_MIN, CRTS_B1_MIN, CRTS_B2_MIN,
    CRTS_C0_MIN, CRTS_C1_MIN, CRTS_C2_MIN,
    CRTS_D0_MIN, CRTS_D1_MIN
]
crts_high = [
    CRTS_A0_MAX, CRTS_A1_MAX, CRTS_A2_MAX,
    CRTS_B0_MAX, CRTS_B1_MAX, CRTS_B2_MAX,
    CRTS_C0_MAX, CRTS_C1_MAX, CRTS_C2_MAX,
    CRTS_D0_MAX, CRTS_D1_MAX
]
crts_bounds = np.array([crts_low, crts_high])

crts_low_init = [
    .8569-.0253, -.2614-.0164, .0200-.0011,
    2.5375-.0187, -1.0425-.0182, 1.1201-.0207,
    13.0088-.0091, -.5759-.0020, .4554-.0027,
    -3.5426-.0209, -.3936-.0073
]
crts_high_init = [
    .8569+.0247, -.2614+.0162, .0200+.0011,
    2.5375+.0177, -1.0425+.0164, 1.1201+.0199,
    13.0088+.0090, -.5759+.0018, .4554+.0028,
    -3.5426+.0235, -.3936+.0070
]
crts_init = np.array([crts_low_init, crts_high_init])


# In[47]:


labels = [
    r"$a_{0}$", r"$a_{1}$", r"$a_{2}$",
    r"$b_{0}$", r"$b_{1}$", r"$b_{2}$",
    r"$c_{0}$", r"$c_{1}$", r"$c_{2}$",
    r"$d_{0}$", r"$d_{1}$"
]


# In[48]:


# set up arrays to track values
# may be unnecessary for non-binned data
LOG_L_VALS = crts_pop['log_l_bol'].values
Z_VALS = crts_pop['z'].values
WEIGHTS = crts_pop['weight'].values

filename = f"../data/processed/crts_qlf_chain.h5"

crts_sampler, crts_ac, crts_idx = mcmc_sampler(
    log_probability,
    crts_bounds,
    completeness_fn=log_l_min_fn,
    cosmo=cosmo,
    sky_coverage=S_sky*S_qual*S_f,
    init_bounds=crts_init,
    nwalkers=24,
    ncpus=12,
    max_nsamples=N_MCMC_SAMPLES,
    filename=filename,
    track_autocorr=True,
    live_plots=False,
    cont=CONT,
    ul=False,
    crts_based=True
)

try:
    tau = crts_sampler.get_autocorr_time()
except:
    print("Run a longer chain!")
    tau = crts_sampler.get_autocorr_time(tol=0)



# In[49]:


samples = crts_sampler.get_chain()
log_norm = crts_sampler.get_blobs()
crts_samples = np.concatenate((samples, log_norm[..., np.newaxis]), axis=-1)
crts_samples[...,[9, 10]] = crts_samples[...,[10, 9]]

discard = int(3 * tau.max())
thin = 15
flat_samples = crts_sampler.get_chain(discard=discard,
                                     thin=thin, flat=True)
flat_log_norm = crts_sampler.get_blobs(discard=discard,
                                      thin=thin, flat=True)
flat_crts_samples = np.concatenate((flat_samples, flat_log_norm[..., np.newaxis]), axis=-1)
flat_crts_samples[...,[9, 10]] = flat_crts_samples[...,[10, 9]]


# In[50]:


# log_norm_min = flat_crts_samples[:, -2][~np.isinf(flat_crts_samples[:, -2])].min()
# n_inf = len(flat_crts_samples[:, -2][np.isinf(flat_crts_samples[:, -2])])
# flat_crts_samples[:, -2][np.isinf(flat_crts_samples[:, -2])] = np.random.uniform(log_norm_min-1, log_norm_min, size=n_inf)


# In[51]:


fig, axes = plt.subplots(len(labels), figsize=(10, 2 * len(labels) - 1), sharex=True)

for i in range(len(labels)):
    ax = axes[i]
    ax.plot(crts_samples[..., i], "k", alpha=0.3)
    ax.set_xlim(0, len(crts_samples))
    if i == 4 or i == 5:
        ax.set_ylim(CRTS_B1_MIN, CRTS_B2_MAX)
    elif i == 7 or i == 8:
        ax.set_ylim(CRTS_C1_MIN, CRTS_C2_MAX)
    else:
        ax.set_ylim(crts_low[i], crts_high[i])
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")
plt.show()

mcmc = np.quantile(flat_crts_samples, [.16, .5, .84], axis=0)
q = np.diff(mcmc, axis=0)

for i in range(len(labels)):
    txt = r"{3} $= {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}$"
    txt = txt.format(mcmc[1, i], q[0, i], q[1, i], labels[i])
    display(Math(txt))

mcmc = np.quantile(flat_crts_samples, [.16, .5, .84], axis=0)
fig = corner.corner(flat_crts_samples, labels=labels,
                    quantiles=[.16, .5, .84], truths=mcmc[1,:])
plt.savefig('../reports/figures/mcmc_schechter_lbol.pdf')
plt.savefig('../reports/figures/mcmc_schechter_lbol.png')
plt.show()


# In[52]:


inds = np.random.choice(range(len(flat_crts_samples)), size=1000, replace=False)
flat_crts_samples = flat_crts_samples[inds]
# crts_flat_samples.append(flat_crts_samples)

a0, a1, a2 = flat_crts_samples[:, 0], flat_crts_samples[:, 1], flat_crts_samples[:, 2]
b0, b1, b2 = flat_crts_samples[:, 3], flat_crts_samples[:, 4], flat_crts_samples[:, 5]
c0, c1, c2 = flat_crts_samples[:, 6], flat_crts_samples[:, 7], flat_crts_samples[:, 8]
d0, d1 = flat_crts_samples[:, 9], flat_crts_samples[:, 10]

crts_post = Shen2020QLF(
    a0=a0, a1=a1, a2=a2,
    b0=b0, b1=b1, b2=b2,
    c0=c0, c1=c1, c2=c2,
    d0=d0, d1=d1
)


# In[53]:


with open(f"../models/crts_dpl_posterior.pkl", "wb") as f:
    pickle.dump(crts_post, f)


# In[54]:


crts_post_log_l_bins = np.empty((len(crts_post), len(log_l_range), 0))
for z_min, z_max in zip(z_bin_min, z_bin_max):
    crts_post_log_l_i = np.transpose([nquad_vec(lambda z: crts_post(log_l, z),
                                          [[z_min, z_max]], n_roots=21)
                                for log_l in log_l_range])
    crts_post_log_l_bins = np.append(crts_post_log_l_bins, crts_post_log_l_i[..., np.newaxis], axis=-1)

crts_post_log_l_bins_q = np.quantile(crts_post_log_l_bins, q=[.16, .5, .84], axis=0)

crts_post_z_bins = np.empty((len(crts_post), len(z_range), 0))
for log_l_min, log_l_max in zip(log_l_bin_min, log_l_bin_max):
    crts_post_z_i = np.transpose([nquad_vec(lambda log_l: crts_post(log_l, z),
                                          [[log_l_min, log_l_max]], n_roots=21)
                                for z in z_range])
    crts_post_z_bins = np.append(crts_post_z_bins, crts_post_z_i[..., np.newaxis], axis=-1)

crts_post_z_bins_q = np.quantile(crts_post_z_bins, q=[.16, .5, .84], axis=0)

crts_post_z = np.transpose([nquad_vec(lambda log_l: crts_post(log_l, z), [[45, 48]], n_roots=21) for z in z_range])
crts_post_z_q = np.quantile(crts_post_z, q=[.16, .5, .84], axis=0)

crts_post_log_l = np.transpose([nquad_vec(lambda z: crts_post(log_l, z), [[0, 1.5]], n_roots=21) for log_l in log_l_range])
crts_post_log_l_q = np.quantile(crts_post_log_l, q=[.16, .5, .84], axis=0)


# In[55]:


# adjust figsize for the number of columns we have
figsize = [2 * plt.rcParams['figure.figsize'][0],
           plt.rcParams['figure.figsize'][1]]
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

alpha = .25

ax[0].plot(log_l_range, qlf_log_l_q[1], color=cs[0], label='QLF, Total', linestyle='-')
ax[0].fill_between(log_l_range, qlf_log_l_q[0], qlf_log_l_q[2],
                   color=cs[0], alpha=alpha)
ax[0].fill_between(log_l_range, qlf_log_l_q[0], qlf_log_l_q[2],
                   facecolor="none", edgecolor=cs[0], hatch='/', alpha=alpha)

ax[0].plot(log_l_range, crts_post_log_l_q[1], color=cs[0], label='CRTS QLF 95\% upper limit, Total', linestyle='--')
ax[0].fill_between(log_l_range, crts_post_log_l_q[0], crts_post_log_l_q[2],
                   color=cs[0], alpha=alpha,
                   # label='CRTS QLF 95\% upper limit, Total',
                   hatch='/')
ax[0].fill_between(log_l_range, crts_post_log_l_q[0], crts_post_log_l_q[2],
                   facecolor="none", edgecolor=cs[0], hatch='/', alpha=alpha)

for i, (z_min, z_max) in enumerate(zip(z_bin_min, z_bin_max)):
    ax[0].plot(log_l_range, qlf_log_l_bins_q[1, :, i], color=cs[i+1], linestyle='-',
               label=r"QLF, ${0} \leq z < {1}$".format(z_min, z_max))
    ax[0].fill_between(log_l_range, qlf_log_l_bins_q[0, :, i], qlf_log_l_bins_q[2, :, i],
                       color=cs[i+1], alpha=alpha)
    ax[0].fill_between(log_l_range, qlf_log_l_bins_q[0, :, i], qlf_log_l_bins_q[2, :, i],
                       facecolor="none", edgecolor=cs[i+1], hatch='\\', alpha=alpha)

    ax[0].plot(log_l_range, crts_post_log_l_bins_q[1, :, i], color=cs[i+1], linestyle='--',
               label=r"CRTS QLF 95\% upper limit, ${0} \leq z < {1}$".format(z_min, z_max)
              )
    ax[0].fill_between(log_l_range, crts_post_log_l_bins_q[0, :, i], crts_post_log_l_bins_q[2, :, i],
                       color=cs[i+1], alpha=alpha, hatch='\\',
                       # label=r"CRTS QLF 95\% upper limit, ${0} \leq z < {1}$".format(z_min, z_max)
                      )
    ax[0].fill_between(log_l_range, crts_post_log_l_bins_q[0, :, i], crts_post_log_l_bins_q[2, :, i],
                       facecolor="none", edgecolor=cs[i+1], hatch='\\', alpha=alpha)

ax[0].legend()

ax[0].set_ylim(1e-11, 1e-4)
ax[0].set_yscale('log')
ax[0].set_xlabel(r"$\log\left(L_{\rm bol}~[\rm{erg}~\rm{s}^{-1}]\right)$")
ax[0].set_ylabel(r"$d\Phi_{\rm QSO} / d\log L_{\rm bol}~[\log L_{\rm bol}^{-1}~\rm{Mpc}^{-3}]$")

ax[1].plot(z_range, qlf_z_q[1], color=cs[0], label='QLF, Total', linestyle='-')
ax[1].fill_between(z_range, qlf_z_q[0], qlf_z_q[2],
                   color=cs[0], alpha=alpha)
ax[1].fill_between(z_range, qlf_z_q[0], qlf_z_q[2],
                   facecolor="none", edgecolor=cs[0], hatch='/', alpha=alpha)

ax[1].plot(z_range, crts_post_z_q[1], color=cs[0],
           # label='Test QLF, Total',
           linestyle='--')
ax[1].fill_between(z_range, crts_post_z_q[0], crts_post_z_q[2],
                   color=cs[0], alpha=alpha)
ax[1].fill_between(z_range, crts_post_z_q[0], crts_post_z_q[2],
                   facecolor="none", edgecolor=cs[0], hatch='\\', alpha=alpha)

for i, (log_l_min, log_l_max) in enumerate(zip(log_l_bin_min, log_l_bin_max)):
    ax[1].plot(z_range, qlf_z_bins_q[1, :, i], color=cs[i+1], linestyle='-',
               label=r"QLF, ${0} \leq \log L_{{\rm bol}} < {1}$".format(log_l_min, log_l_max))
    ax[1].fill_between(z_range, qlf_z_bins_q[0, :, i], qlf_z_bins_q[2, :, i],
                       color=cs[i+1], alpha=alpha)
    ax[1].fill_between(z_range, qlf_z_bins_q[0, :, i], qlf_z_bins_q[2, :, i],
                       facecolor="none", edgecolor=cs[i+1], hatch='/', alpha=alpha)

    ax[1].plot(z_range, crts_post_z_bins_q[1, :, i], color=cs[i+1], linestyle='--',
               # label=r"Test QLF, ${0} \leq \log L_{{\rm bol}} < {1}$".format(log_l_min, log_l_max)
              )
    ax[1].fill_between(z_range, crts_post_z_bins_q[0, :, i], crts_post_z_bins_q[2, :, i],
                       color=cs[i+1], alpha=alpha)
    ax[1].fill_between(z_range, crts_post_z_bins_q[0, :, i], crts_post_z_bins_q[2, :, i],
                       facecolor="none", edgecolor=cs[i+1], hatch='\\', alpha=alpha)

ax[1].legend()

ax[1].set_ylim(1e-11, 1e-4)
ax[1].set_yscale('log')
ax[1].set_xlabel(r"$z$")
ax[1].set_ylabel(r"$d\Phi_{\rm QSO} / dz~[\rm{Mpc}^{-3}]$")

plt.tight_layout()

plt.savefig('../reports/figures/crts_qlf_fit.pdf')
plt.show()


# ### Extended Sample

# In[56]:


# Define the allowed parameter space
BQ_A0_MIN, BQ_A0_MAX = .5, 3.5
BQ_A1_MIN, BQ_A1_MAX = -2, 1
BQ_A2_MIN, BQ_A2_MAX = -.5, .5

BQ_B0_MIN, BQ_B0_MAX = 0, 3
BQ_B1_MIN, BQ_B1_MAX = -3, 0
BQ_B2_MIN, BQ_B2_MAX = 0, 10

BQ_C0_MIN, BQ_C0_MAX = 44 - LOG_L_SOLAR, 48 - LOG_L_SOLAR
BQ_C1_MIN, BQ_C1_MAX = -1, 0
BQ_C2_MIN, BQ_C2_MAX = 0, 3

BQ_D0_MIN, BQ_D0_MAX = -15, 5  # irrelevant for the STY method
BQ_D1_MIN, BQ_D1_MAX = -2, 2


# In[57]:


# set up lower and upper bounds on each parameter
bq_low = [
    BQ_A0_MIN, BQ_A1_MIN, BQ_A2_MIN,
    BQ_B0_MIN, BQ_B1_MIN, BQ_B2_MIN,
    BQ_C0_MIN, BQ_C1_MIN, BQ_C2_MIN,
    BQ_D0_MIN, BQ_D1_MIN
]
bq_high = [
    BQ_A0_MAX, BQ_A1_MAX, BQ_A2_MAX,
    BQ_B0_MAX, BQ_B1_MAX, BQ_B2_MAX,
    BQ_C0_MAX, BQ_C1_MAX, BQ_C2_MAX,
    BQ_D0_MAX, BQ_D1_MAX
]
bq_bounds = np.array([bq_low, bq_high])

bq_low_init = [
    .8569-.0253, -.2614-.0164, .0200-.0011,
    2.5375-.0187, -1.0425-.0182, 1.1201-.0207,
    13.0088-.0091, -.5759-.0020, .4554-.0027,
    -3.5426-.0209, -.3936-.0073
]
bq_high_init = [
    .8569+.0247, -.2614+.0162, .0200+.0011,
    2.5375+.0177, -1.0425+.0164, 1.1201+.0199,
    13.0088+.0090, -.5759+.0018, .4554+.0028,
    -3.5426+.0235, -.3936+.0070
]
bq_init = np.array([bq_low_init, bq_high_init])


# In[58]:


labels = [
    r"$a_{0}$", r"$a_{1}$", r"$a_{2}$",
    r"$b_{0}$", r"$b_{1}$", r"$b_{2}$",
    r"$c_{0}$", r"$c_{1}$", r"$c_{2}$",
    r"$d_{0}$", r"$d_{1}$"
]


# In[59]:


# set up arrays to track values
# may be unnecessary for non-binned data
bq_frac = .5
bq_pop_fit = bq_pop.sample(frac=bq_frac)
bq_pop_fit


# In[60]:


# set up arrays to track values
# may be unnecessary for non-binned data
LOG_L_VALS = bq_pop['log_l_bol'].values
Z_VALS = bq_pop['z'].values
WEIGHTS = bq_pop['weight'].values

filename = f"../data/processed/bq_qlf_chain.h5"

bq_sampler, bq_ac, bq_idx = mcmc_sampler(
    log_probability,
    bq_bounds,
    completeness_fn=LOG_L_MIN,
    cosmo=cosmo,
    sky_coverage=S_sky*S_qual*S_f*bq_frac,
    init_bounds=bq_init,
    nwalkers=24,
    ncpus=12,
    max_nsamples=N_MCMC_SAMPLES,
    filename=filename,
    track_autocorr=True,
    live_plots=False,
    cont=CONT,
    ul=False,
    crts_based=True,
    skip_initial_state_check=False
)

try:
    tau = bq_sampler.get_autocorr_time()
except:
    print("Run a longer chain!")
    tau = bq_sampler.get_autocorr_time(tol=0)



# In[61]:


samples = bq_sampler.get_chain()
log_norm = bq_sampler.get_blobs()
bq_samples = np.concatenate((samples, log_norm[..., np.newaxis]), axis=-1)
bq_samples[...,[9, 10]] = bq_samples[...,[10, 9]]

discard = int(3 * tau.max())
thin = 15
flat_samples = bq_sampler.get_chain(discard=discard,
                                     thin=thin, flat=True)
flat_log_norm = bq_sampler.get_blobs(discard=discard,
                                      thin=thin, flat=True)
flat_bq_samples = np.concatenate((flat_samples, flat_log_norm[..., np.newaxis]), axis=-1)
flat_bq_samples[...,[9, 10]] = flat_bq_samples[...,[10, 9]]


# In[62]:


# log_norm_min = flat_bq_samples[:, -2][~np.isinf(flat_bq_samples[:, -2])].min()
# n_inf = len(flat_bq_samples[:, -2][np.isinf(flat_bq_samples[:, -2])])
# flat_bq_samples[:, -2][np.isinf(flat_bq_samples[:, -2])] = np.random.uniform(log_norm_min-1, log_norm_min, size=n_inf)


# In[63]:


fig, axes = plt.subplots(len(labels), figsize=(10, 2 * len(labels) - 1), sharex=True)

for i in range(len(labels)):
    ax = axes[i]
    ax.plot(bq_samples[..., i], "k", alpha=0.3)
    ax.set_xlim(0, len(bq_samples))
    if i == 4 or i == 5:
        ax.set_ylim(BQ_B1_MIN, BQ_B2_MAX)
    elif i == 7 or i == 8:
        ax.set_ylim(BQ_C1_MIN, BQ_C2_MAX)
    else:
        ax.set_ylim(bq_low[i], bq_high[i])
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")
plt.show()

mcmc = np.quantile(flat_bq_samples, [.16, .5, .84], axis=0)
q = np.diff(mcmc, axis=0)

for i in range(len(labels)):
    txt = r"{3} $= {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}$"
    txt = txt.format(mcmc[1, i], q[0, i], q[1, i], labels[i])
    display(Math(txt))

mcmc = np.quantile(flat_bq_samples, [.16, .5, .84], axis=0)
bq_corner_bounds = bq_bounds.T
# bq_corner_bounds = np.append(bq_corner_bounds, [[BQ_D0_MIN, BQ_D0_MAX]], axis=0)
# bq_corner_bounds[[-1, -2]] = bq_corner_bounds[[-2, -1]]
fig = corner.corner(flat_bq_samples, labels=labels,
                    quantiles=[.16, .5, .84], truths=mcmc[1,:],
                    # range=bq_corner_bounds
                   )
plt.savefig('../reports/figures/mcmc_schechter_lbol.pdf')
# plt.savefig('../reports/figures/mcmc_schechter_lbol.png')
plt.show()


# In[64]:


inds = np.random.choice(range(len(flat_bq_samples)), size=1000, replace=False)
flat_bq_samples = flat_bq_samples[inds]
# bq_flat_samples.append(flat_bq_samples)

a0, a1, a2 = flat_bq_samples[:, 0], flat_bq_samples[:, 1], flat_bq_samples[:, 2]
b0, b1, b2 = flat_bq_samples[:, 3], flat_bq_samples[:, 4], flat_bq_samples[:, 5]
c0, c1, c2 = flat_bq_samples[:, 6], flat_bq_samples[:, 7], flat_bq_samples[:, 8]
d0, d1 = flat_bq_samples[:, 9], flat_bq_samples[:, 10]

bq_post = Shen2020QLF(
    a0=a0, a1=a1, a2=a2,
    b0=b0, b1=b1, b2=b2,
    c0=c0, c1=c1, c2=c2,
    d0=d0, d1=d1
)


# In[65]:


with open(f"../models/bq_dpl_posterior.pkl", "wb") as f:
    pickle.dump(bq_post, f)


# In[66]:


bq_post_log_l_bins = np.empty((len(bq_post), len(log_l_range), 0))
for z_min, z_max in zip(z_bin_min, z_bin_max):
    bq_post_log_l_i = np.transpose([nquad_vec(lambda z: bq_post(log_l, z),
                                          [[z_min, z_max]], n_roots=21)
                                for log_l in log_l_range])
    bq_post_log_l_bins = np.append(bq_post_log_l_bins, bq_post_log_l_i[..., np.newaxis], axis=-1)

bq_post_log_l_bins_q = np.quantile(bq_post_log_l_bins, q=[.16, .5, .84], axis=0)

bq_post_z_bins = np.empty((len(bq_post), len(z_range), 0))
for log_l_min, log_l_max in zip(log_l_bin_min, log_l_bin_max):
    bq_post_z_i = np.transpose([nquad_vec(lambda log_l: bq_post(log_l, z),
                                          [[log_l_min, log_l_max]], n_roots=21)
                                for z in z_range])
    bq_post_z_bins = np.append(bq_post_z_bins, bq_post_z_i[..., np.newaxis], axis=-1)

bq_post_z_bins_q = np.quantile(bq_post_z_bins, q=[.16, .5, .84], axis=0)

bq_post_z = np.transpose([nquad_vec(lambda log_l: bq_post(log_l, z), [[45, 48]], n_roots=21) for z in z_range])
bq_post_z_q = np.quantile(bq_post_z, q=[.16, .5, .84], axis=0)

bq_post_log_l = np.transpose([nquad_vec(lambda z: bq_post(log_l, z), [[0, 1.5]], n_roots=21) for log_l in log_l_range])
bq_post_log_l_q = np.quantile(bq_post_log_l, q=[.16, .5, .84], axis=0)


# In[67]:


# adjust figsize for the number of columns we have
figsize = [2 * plt.rcParams['figure.figsize'][0],
           plt.rcParams['figure.figsize'][1]]
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

alpha = .25
hatches = ['\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

ax[0].plot(log_l_range, qlf_log_l_q[1], color=cs[0], label='QLF, Total', linestyle='-')
ax[0].fill_between(log_l_range, qlf_log_l_q[0], qlf_log_l_q[2],
                   color=cs[0], alpha=alpha, hatch='/')
ax[0].fill_between(log_l_range, qlf_log_l_q[0], qlf_log_l_q[2],
                   facecolor="none", edgecolor=cs[0], hatch='/', alpha=alpha)

ax[0].plot(log_l_range, bq_post_log_l_q[1], color=cs[0], label='BQLF, Total', linestyle='--')
ax[0].fill_between(log_l_range, bq_post_log_l_q[0], bq_post_log_l_q[2],
                   color=cs[0], alpha=alpha,
                   # label='BQLF, Total',
                   hatch='/')
ax[0].fill_between(log_l_range, bq_post_log_l_q[0], bq_post_log_l_q[2],
                   facecolor="none", edgecolor=cs[0], hatch='/', alpha=alpha)

for i, (z_min, z_max) in enumerate(zip(z_bin_min, z_bin_max)):
    ax[0].plot(log_l_range, qlf_log_l_bins_q[1, :, i], color=cs[i+1], linestyle='-',
               label=r"QLF, ${0} \leq z < {1}$".format(z_min, z_max))
    ax[0].fill_between(log_l_range, qlf_log_l_bins_q[0, :, i], qlf_log_l_bins_q[2, :, i],
                       color=cs[i+1], alpha=alpha, hatch=hatches[i])
    ax[0].fill_between(log_l_range, qlf_log_l_bins_q[0, :, i], qlf_log_l_bins_q[2, :, i],
                       facecolor="none", edgecolor=cs[i+1], hatch=hatches[i], alpha=alpha)

    ax[0].plot(log_l_range, bq_post_log_l_bins_q[1, :, i], color=cs[i+1], linestyle='--',
               label=r"BQLF, ${0} \leq z < {1}$".format(z_min, z_max)
              )
    ax[0].fill_between(log_l_range, bq_post_log_l_bins_q[0, :, i], bq_post_log_l_bins_q[2, :, i],
                       color=cs[i+1], alpha=alpha, hatch=hatches[i],
                       # label=r"BQLF, ${0} \leq z < {1}$".format(z_min, z_max)
                      )
    ax[0].fill_between(log_l_range, bq_post_log_l_bins_q[0, :, i], bq_post_log_l_bins_q[2, :, i],
                       facecolor="none", edgecolor=cs[i+1], hatch=hatches[i], alpha=alpha)

ax[0].legend()

ax[0].set_ylim(1e-11, 1e-4)
ax[0].set_yscale('log')
ax[0].set_xlabel(r"$\log\left(L_{\rm bol}~[\rm{erg}~\rm{s}^{-1}]\right)$")
ax[0].set_ylabel(r"$d\Phi_{\rm QSO} / d\log L_{\rm bol}~[\log L_{\rm bol}^{-1}~\rm{Mpc}^{-3}]$")

ax[1].plot(z_range, qlf_z_q[1], color=cs[0], label='Total', linestyle='-')
ax[1].fill_between(z_range, qlf_z_q[0], qlf_z_q[2],
                   color=cs[0], alpha=alpha)
ax[1].fill_between(z_range, qlf_z_q[0], qlf_z_q[2],
                   facecolor="none", edgecolor=cs[0], hatch='/', alpha=alpha)

ax[1].plot(z_range, bq_post_z_q[1], color=cs[0],
           # label='Test QLF, Total',
           linestyle='--')
ax[1].fill_between(z_range, bq_post_z_q[0], bq_post_z_q[2],
                   color=cs[0], alpha=alpha)
ax[1].fill_between(z_range, bq_post_z_q[0], bq_post_z_q[2],
                   facecolor="none", edgecolor=cs[0], hatch='/', alpha=alpha)

for i, (log_l_min, log_l_max) in enumerate(zip(log_l_bin_min, log_l_bin_max)):
    ax[1].plot(z_range, qlf_z_bins_q[1, :, i], color=cs[i+1], linestyle='-',
               label=r"${0} \leq \log L_{{\rm bol}} < {1}$".format(log_l_min, log_l_max))
    ax[1].fill_between(z_range, qlf_z_bins_q[0, :, i], qlf_z_bins_q[2, :, i],
                       color=cs[i+1], alpha=alpha, hatch=hatches[i])
    ax[1].fill_between(z_range, qlf_z_bins_q[0, :, i], qlf_z_bins_q[2, :, i],
                       facecolor="none", edgecolor=cs[i+1], hatch=hatches[i], alpha=alpha)

    ax[1].plot(z_range, bq_post_z_bins_q[1, :, i], color=cs[i+1], linestyle='--',
               # label=r"Test QLF, ${0} \leq \log L_{{\rm bol}} < {1}$".format(log_l_min, log_l_max)
              )
    ax[1].fill_between(z_range, bq_post_z_bins_q[0, :, i], bq_post_z_bins_q[2, :, i],
                       color=cs[i+1], alpha=alpha, hatch=hatches[i])
    ax[1].fill_between(z_range, bq_post_z_bins_q[0, :, i], bq_post_z_bins_q[2, :, i],
                       facecolor="none", edgecolor=cs[i+1], hatch=hatches[i], alpha=alpha)

ax[1].legend()

ax[1].set_ylim(1e-11, 1e-4)
ax[1].set_yscale('log')
ax[1].set_xlabel(r"$z$")
ax[1].set_ylabel(r"$d\Phi_{\rm QSO} / dz~[\rm{Mpc}^{-3}]$")

plt.tight_layout()

plt.savefig('../reports/figures/bq_qlf_fit.pdf')
plt.show()


# In[ ]:
