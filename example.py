# # -*- coding: utf-8 -*-
# """
# ======================
# Community and Crime
# ======================

# This is a real dataset of per capita violent crime, with demographic
# data comprising 128 attributes from 1994 counties in the US.

# The original dataset can be found here:
# http://archive.ics.uci.edu/ml/datasets/Communities+and+Crime

# The target variables (per capita violent crime) are normalized to lie in
# a [0, 1] range. We preprocessed this dataset to exclude attributes with
# missing values.
# """

# ########################################################

# # Author: Vinicius Marques <vini.type@gmail.com>
# # License: MIT

# ########################################################
# # Imports

# #import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from pyglmnet import GLM, GLMCV, datasets

# ########################################################
# # Download and preprocess data files

# X, y = datasets.fetch_community_crime_data()
# n_samples, n_features = X.shape

# ########################################################
# # Split the data into training and test sets

# X_train, X_test, y_train, y_test = \
#     train_test_split(X, y, test_size=0.33, random_state=0)

# ########################################################
# # Fit a binomial distributed GLM with elastic net regularization

# # use the default value for reg_lambda
# glm = GLMCV(distr='binomial', alpha=0.05, score_metric='pseudo_R2', cv=3,
#             tol=1e-4)

# # fit model
# glm.fit(X_train, y_train)

# # score the test set prediction
# y_test_hat = glm.predict_proba(X_test)
# print("test set pseudo $R^2$ = %f" % glm.score(X_test, y_test))

# ########################################################
# # Now use GridSearchCV to compare

# import numpy as np # noqa
# from sklearn.model_selection import GridSearchCV # noqa
# from sklearn.model_selection import KFold # noqa

# cv = KFold(3)

# reg_lambda = np.logspace(np.log(0.5), np.log(0.01), 10,
#                          base=np.exp(1))
# param_grid = [{'reg_lambda': reg_lambda}]

# glm = GLM(distr='binomial', alpha=0.05, score_metric='pseudo_R2',
#           learning_rate=0.1, tol=1e-4, verbose=True)



# glmcv = GridSearchCV(glm, param_grid, cv=cv)
# glmcv.fit(X_train, y_train)

# print("test set pseudo $R^2$ = %f" % glmcv.score(X_test, y_test))












# -*- coding: utf-8 -*-
"""
================================================================
GLM for Spike Train Prediction in Primate Retinal Ganglion Cells
================================================================

* Original tutorial adapted from Johnathan Pillow, Princeton University
* Dataset provided by E.J. Chichilnisky, Stanford University
* The dataset is granted by the original authors for educational use only
* Please contact ``pillow@princeton.edu`` if using beyond its purposes

The original MATLAB and Python tutorial can be found from
https://github.com/pillowlab/GLMspiketraintutorial.

These data were collected by Valerie Uzzell in the lab of
E.J. Chichilnisky at the Salk Institute. For full information see
Uzzell et al. [1]_, or Pillow et al. [2]_.

In this tutorial, we will demonstrate how to fit linear GLM and
Poisson GLM to predict the spike counts
recorded from primate retinal ganglion cells.
The dataset contains spike responses from 2 ON and 2 OFF parasol
retinal ganglion cells (RGCs) in primate retina, stimulated with
full-field `binary white noise`. Two experiment performed consisted of a
long (20-minute) binary stochastic (non-repeating) stimulus
which can be used for computing the spike-triggered average
(or characterizing some other model of the response).

References
----------
.. [1] Uzzell, V. J., and E. J. Chichilnisky. "Precision of
   spike trains in primate retinal ganglion cells."
   Journal of Neurophysiology 92.2 (2004)
.. [2] Pillow, Jonathan W., et al. "Prediction and decoding of
   retinal ganglion cell responses with a probabilistic
   spiking model." Journal of Neuroscience 25.47 (2005)

"""

# Authors: Jonathan Pillow <pillow@princeton.edu>
#          Titipat Achakulvisut <my.titipat@gmail.com>
# License: MIT

########################################################
#
# Import all the relevance libraries.

import os.path as op
import json

import numpy as np
from scipy.linalg import hankel

from pyglmnet import GLM
from pyglmnet.datasets import fetch_rgc_spike_trains


########################################################
#
# Fetch the dataset. The JSON file contains the
# following keys:  ``stim`` (binary stochastic stimulation),
# ``stim_times`` (time of the stimulation), and
# ``spike_times`` (recorded time of the spikes)

dpath = fetch_rgc_spike_trains()
with open(op.join(dpath, 'data_RGCs.json'), 'r') as f:
    rgcs_dataset = json.loads(f.read())

stim = np.array(rgcs_dataset['stim'])
stim_times = np.array(rgcs_dataset['stim_times'])

n_cells = len(rgcs_dataset['spike_times'])
dt = stim_times[1] - stim_times[0]  # time between the stimulation
n_times = len(stim)  # total number of the stimulation
sfreq = 1. / dt  # frequency of the stimulation

########################################################
#
# You can pick a cell to work with and visualize the spikes for one second.
# In this case, we will pick cell number 2 (ON cell).

cell_idx = 2  # pick cell number 2 (ON cell)
spike_times = rgcs_dataset['spike_times']['cell_%d' % cell_idx]
spike_times = np.array(spike_times)
n_spikes = len(spike_times)  # number of spikes

# bin the spikes to y which is our predictor
t_bins = np.arange(n_times + 1) * dt
y, _ = np.histogram(spike_times, t_bins)

print('Loaded RGC data: cell {}'.format(cell_idx))
print('Number of stim frames: {:d} ({:.1f} minutes)'.
      format(n_times, n_times * dt / 60))
print('Time bin size: {:.1f} ms'.format(dt * 1000))
print('Number of spikes: {} (mean rate = {:.1f} Hz)\n'.
      format(n_spikes, n_spikes / n_times * 60))

########################################################
#
# We can use ``scipy``'s function ``hankel`` to make our design matrix.
# The design matrix :math:`X` can be created using the stimulation and
# its history. Later in the tutorial, we will also incorporate spikes
# history into our design matrix.

n_t_filt = 25  # tweak this to see different results
stim_padded = np.pad(stim, (n_t_filt - 1, 0))
Xdsgn = hankel(stim_padded[0: -n_t_filt + 1], stim[-n_t_filt:])


########################################################
# **Fitting and predicting with a linear-Gaussian GLM**
#
# For a general linear model, the observed spikes can be
# thought of an underlying parameter
# :math:`\beta_0, \beta` that control the spiking.
#
# You can simply use linear Gaussian GLM with no regularization
# to predict the spike counts.

glm_lg = GLM(distr='gaussian',
             reg_lambda=0.0,
             score_metric='pseudo_R2')
glm_lg.fit(Xdsgn, y)

# predict spike counts
ypred_lg = glm_lg.predict(Xdsgn)

########################################################
# **Fitting and predicting with a Poisson GLM**
#
# We can also assume that there is a non-linear function governing
# the underlying the firing patterns.
# In pyglmnet, we use an exponential inverse link function
# for the Poisson distribution.

glm_poisson = GLM(distr='poisson',
                  alpha=0.05,
                  learning_rate=1.0,
                  score_metric='pseudo_R2',
                  reg_lambda=1e-7)
glm_poisson.fit(Xdsgn, y)

# predict spike counts
ypred_poisson = glm_poisson.predict(Xdsgn)

########################################################
# **Adding spikes history for predicting spike counts**
#
# We can even further predict the spikes by concatenating the spikes history
# to the stimulation history in the design matrix.

n_t_filt = 25  # same as before, stimulation history
n_t_hist = 20  # spikes history

# using both stimulation history and spikes history
y_padded = np.pad(y, (n_t_hist, 0))

Xstim = hankel(stim_padded[:-n_t_filt + 1], stim[-n_t_filt:])
Xspikes = hankel(y_padded[:-n_t_hist], stim[-n_t_hist:])
Xdsgn_hist = np.hstack((Xstim, Xspikes))  # design matrix with spikes history

########################################################
#
# .. warning::
#    The spike-history portion of the design
#    matrix had better be shifted so that we aren't allowed to
#    use the spike count on this time bin to predict itself!
#
# Now, we are ready to fit Poisson GLM with spikes history.
glm_poisson_hist = GLM(distr='poisson',
                       alpha=0.05,
                       learning_rate=1.0,
                       score_metric='pseudo_R2',
                       reg_lambda=1e-7)

# fit and predict with spikes history
glm_poisson_hist.fit(Xdsgn_hist, y)
ypred_poisson_hist = glm_poisson_hist.predict(Xdsgn_hist)


# print scores of all the fitted models
print('Training perf (R^2): lin-gauss GLM, w/ offset: {:.2f}'
      .format(glm_lg.score(Xdsgn, y)))
print('Training perf (R^2): Pyglmnet possion GLM {:.2f}'
      .format(glm_poisson.score(Xdsgn, y)))
print('Training perf (R^2): Pyglmnet poisson GLM w/ spikes history {:.2f}'
      .format(glm_poisson_hist.score(Xdsgn_hist, y)))
