# Hidden Markov Models
#
# Author: Ron Weiss <ronweiss@gmail.com>
#         Shiqiao Du <lucidfrontier.45@gmail.com>
# API changes: Jaques Grobler <jaquesgrobler@gmail.com>
# Modifications to create of the HMMLearn module: Gael Varoquaux
# More API changes: Sergei Lebedev <superbobry@gmail.com>

"""
The :mod:`hmmlearn.hmm` module implements hidden Markov models.
"""

import math
import logging
import numpy as np
from scipy.stats import nbinom, norm
from scipy.special import digamma, polygamma
from sklearn import cluster
from sklearn.mixture import (
    GMM, sample_gaussian,
    log_multivariate_normal_density,
    distribute_covar_matrix_to_match_covariance_type, _validate_covars)
from sklearn.utils import check_random_state

import finding_r
from .base import _BaseHMM
from .utils import iter_from_X_lengths, normalize

__all__ = ["GMMHMM", "GaussianHMM", "MultinomialHMM", "NegativeBinomialHMM"]

COVARIANCE_TYPES = frozenset(("spherical", "diag", "full", "tied"))


class GaussianHMM(_BaseHMM):
    """Hidden Markov Model with Gaussian emissions.

    Parameters
    ----------
    n_components : int
        Number of states.

    covariance_type : string
        String describing the type of covariance parameters to
        use.  Must be one of

        * "spherical" --- each state uses a single variance value that
          applies to all features;
        * "diag" --- each state uses a diagonal covariance matrix;
        * "full" --- each state uses a full (i.e. unrestricted)
          covariance matrix;
        * "tied" --- all states use **the same** full covariance matrix.

        Defaults to "diag".

    min_covar : float
        Floor on the diagonal of the covariance matrix to prevent
        overfitting. Defaults to 1e-3.

    startprob_prior : array, shape (n_components, )
        Initial state occupation prior distribution.

    transmat_prior : array, shape (n_components, n_components)
        Matrix of prior transition probabilities between states.

    algorithm : string
        Decoder algorithm. Must be one of "viterbi" or "map".
        Defaults to "viterbi".

    random_state: RandomState or an int seed
        A random number generator instance.

    n_iter : int, optional
        Maximum number of iterations to perform.

    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        is below this value.

    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, 'm' for means and 'c' for covars. Defaults
        to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, 'm' for means and 'c' for covars.
        Defaults to all parameters.

    Attributes
    ----------
    n_features : int
        Dimensionality of the Gaussian emissions.

    monitor\_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    transmat\_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    startprob\_ : array, shape (n_components, )
        Initial state occupation distribution.

    means\_ : array, shape (n_components, n_features)
        Mean parameters for each state.

    covars\_ : array
        Covariance parameters for each state.

        The shape depends on ``covariance_type``::

            (n_components, )                        if 'spherical',
            (n_features, n_features)                if 'tied',
            (n_components, n_features)              if 'diag',
            (n_components, n_features, n_features)  if 'full'

    Examples
    --------
    >>> from hmmlearn.hmm import GaussianHMM
    >>> GaussianHMM(n_components=2)
    ...                             #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    GaussianHMM(algorithm='viterbi',...
    """
    def __init__(self, n_components=1, covariance_type='diag',
                 min_covar=1e-3,
                 startprob_prior=1.0, transmat_prior=1.0,
                 means_prior=0, means_weight=0,
                 covars_prior=1e-2, covars_weight=1,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="stmc", init_params="stmc"):
        _BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          tol=tol, params=params, verbose=verbose,
                          init_params=init_params)

        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.means_prior = means_prior
        self.means_weight = means_weight
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight

    def _get_covars(self):
        """Return covars as a full matrix."""
        if self.covariance_type == 'full':
            return self._covars_
        elif self.covariance_type == 'diag':
            return np.array([np.diag(cov) for cov in self._covars_])
        elif self.covariance_type == 'tied':
            return np.array([self._covars_] * self.n_components)
        elif self.covariance_type == 'spherical':
            return np.array(
                [np.eye(self.n_features) * cov for cov in self._covars_])

    def _set_covars(self, covars):
        self._covars_ = np.asarray(covars).copy()

    covars_ = property(_get_covars, _set_covars)

    def _check(self):
        super(GaussianHMM, self)._check()

        self.means_ = np.asarray(self.means_)
        self.n_features = self.means_.shape[1]

        if self.covariance_type not in COVARIANCE_TYPES:
            raise ValueError('covariance_type must be one of {0}'
                             .format(COVARIANCE_TYPES))

        _validate_covars(self._covars_, self.covariance_type,
                         self.n_components)

    def _init(self, X, lengths=None):
        super(GaussianHMM, self)._init(X, lengths=lengths)

        _, n_features = X.shape
        if hasattr(self, 'n_features') and self.n_features != n_features:
            raise ValueError('Unexpected number of dimensions, got %s but '
                             'expected %s' % (n_features, self.n_features))

        self.n_features = n_features
        if 'm' in self.init_params or not hasattr(self, "means_"):
            kmeans = cluster.KMeans(n_clusters=self.n_components,
                                    random_state=self.random_state)
            kmeans.fit(X)
            #self.means_ = kmeans.cluster_centers_
            means = kmeans.cluster_centers_
            #logging.debug("MEANS:", means)
            means = np.sort(means, axis = 0)
            #logging.debug("MEANS:", means)
            self.means_ = means
 
        if 'c' in self.init_params or not hasattr(self, "covars_"):
            cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
            #logging.debug("CV:", cv)
            if not cv.shape:
                cv.shape = (1, 1)
            self._covars_ = distribute_covar_matrix_to_match_covariance_type(
                cv, self.covariance_type, self.n_components).copy()
            #logging.debug("COVARS:", self._covars_)

    def _compute_log_likelihood(self, X):
        #logging.debug("argumenty dla log_multivariate_normal_density:")
        #logging.debug(self.means_, self._covars_)
        return log_multivariate_normal_density(
            X, self.means_, self._covars_, self.covariance_type)

    # moja wersja, analogiczna do NB, przetestowane ze dziala tak samo:
    #def _compute_log_likelihood(self, X):
    #    n_observations, n_dim = X.shape
    #    means, covars = self.means_, self.covars_
    #    log_likelihood = np.ndarray((n_observations, self.n_components))
    #    for i in xrange(n_observations):
    #        for j in xrange(self.n_components):
    #            log_likelihood[i, j] = np.sum(norm.logpdf(X[i,:],
    #                                          means[j,:], math.sqrt(covars[j,:])))
    #    return log_likelihood



    def _generate_sample_from_state(self, state, random_state=None):
        if self.covariance_type == 'tied':
            cv = self._covars_
        else:
            cv = self._covars_[state]
        return sample_gaussian(self.means_[state], cv, self.covariance_type,
                               random_state=random_state)

    def _initialize_sufficient_statistics(self):
        stats = super(GaussianHMM, self)._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        stats['obs**2'] = np.zeros((self.n_components, self.n_features))
        if self.covariance_type in ('tied', 'full'):
            stats['obs*obs.T'] = np.zeros((self.n_components, self.n_features,
                                           self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):

        #logging.debug("Accumulate_sufficient_statistics, stats:")
        #for key, value in stats.items():
        #    logging.debug(key)
        #    logging.debug((value)

        super(GaussianHMM, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice)

        #logging.debug("Accumulate_sufficient_statistics, stats po raz drugi:")
        #for key, value in stats.items():
        #    logging.debug(key)
        #    logging.debug(value)

        #logging.debug("wymiar posteriors:", posteriors.shape)

        if 'm' in self.params or 'c' in self.params:
            
            stats['post'] += posteriors.sum(axis=0)
            #logging.debug( "w accumulate w hmm stats[post]:")
            #logging.debug( stats['post'])
            #logging.debug( "Czy wszystkie posteriors w kolumnie rowne zero:")
            #logging.debug( np.all(posteriors == 0, axis=0))
            stats['obs'] += np.dot(posteriors.T, obs)
            #logging.debug( "wymiary stats[post] i stats[obs]:", stats['post'].shape, stats['obs'].shape)

        if 'c' in self.params:
            if self.covariance_type in ('spherical', 'diag'):
                stats['obs**2'] += np.dot(posteriors.T, obs ** 2)
            elif self.covariance_type in ('tied', 'full'):
                # posteriors: (nt, nc); obs: (nt, nf); obs: (nt, nf)
                # -> (nc, nf, nf)
                stats['obs*obs.T'] += np.einsum(
                    'ij,ik,il->jkl', posteriors, obs, obs)

        #logging.debug("Accumulate_sufficient_statistics, stats po raz trzeci:")
        #for key, value in stats.items():
        #    logging.debug(key)
        #    logging.debug(value)

        
    def _do_mstep(self, stats):
        super(GaussianHMM, self)._do_mstep(stats)

        means_prior = self.means_prior
        means_weight = self.means_weight
        #logging.debug("Mstep; means_prior, weight:")
        #logging.debug(means_prior)
        #logging.debug(means_weight)


        # TODO: find a proper reference for estimates for different
        #       covariance models.
        # Based on Huang, Acero, Hon, "Spoken Language Processing",
        # p. 443 - 445
        #logging.debug("stats[post]:", stats['post'])
        denom = stats['post'][:, np.newaxis]
        #logging.debug("denom:")
        #logging.debug(denom)
        if 'm' in self.params:
            logging.debug("stare means:")
            logging.debug(self.means_)
            #logging.debug("Uaktualniamy m; denom, stats[obs], nowe means:")
            logging.debug("nowe means:")
            #logging.debug(denom)
            #logging.debug(stats['obs'])
            self.means_ = ((means_weight * means_prior + stats['obs'])
                           / (means_weight + denom))
            logging.debug(self.means_)

        if 'c' in self.params:
            covars_prior = self.covars_prior
            covars_weight = self.covars_weight
            meandiff = self.means_ - means_prior

            if self.covariance_type in ('spherical', 'diag'):
                cv_num = (means_weight * meandiff**2
                          + stats['obs**2']
                          - 2 * self.means_ * stats['obs']
                          + self.means_**2 * denom)
                cv_den = max(covars_weight - 1, 0) + denom
                self._covars_ = \
                    (covars_prior + cv_num) / np.maximum(cv_den, 1e-5)
                if self.covariance_type == 'spherical':
                    self._covars_ = np.tile(
                        self._covars_.mean(1)[:, np.newaxis],
                        (1, self._covars_.shape[1]))
            elif self.covariance_type in ('tied', 'full'):
                cv_num = np.empty((self.n_components, self.n_features,
                                  self.n_features))
                for c in range(self.n_components):
                    obsmean = np.outer(stats['obs'][c], self.means_[c])

                    cv_num[c] = (means_weight * np.outer(meandiff[c],
                                                         meandiff[c])
                                 + stats['obs*obs.T'][c]
                                 - obsmean - obsmean.T
                                 + np.outer(self.means_[c], self.means_[c])
                                 * stats['post'][c])
                cvweight = max(covars_weight - self.n_features, 0)
                if self.covariance_type == 'tied':
                    self._covars_ = ((covars_prior + cv_num.sum(axis=0)) /
                                     (cvweight + stats['post'].sum()))
                elif self.covariance_type == 'full':
                    self._covars_ = ((covars_prior + cv_num) /
                                     (cvweight + stats['post'][:, None, None]))



class MultinomialHMM(_BaseHMM):
    """Hidden Markov Model with multinomial (discrete) emissions

    Parameters
    ----------

    n_components : int
        Number of states.

    startprob_prior : array, shape (n_components, )
        Initial state occupation prior distribution.

    transmat_prior : array, shape (n_components, n_components)
        Matrix of prior transition probabilities between states.

    algorithm : string
        Decoder algorithm. Must be one of "viterbi" or "map".
        Defaults to "viterbi".

    random_state: RandomState or an int seed
        A random number generator instance.

    n_iter : int, optional
        Maximum number of iterations to perform.

    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        is below this value.

    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, 'e' for emissionprob.
        Defaults to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, 'e' for emissionprob.
        Defaults to all parameters.

    Attributes
    ----------
    n_features : int
        Number of possible symbols emitted by the model (in the samples).

    monitor\_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    transmat\_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    startprob\_ : array, shape (n_components, )
        Initial state occupation distribution.

    emissionprob\_ : array, shape (n_components, n_features)
        Probability of emitting a given symbol when in each state.

    Examples
    --------
    >>> from hmmlearn.hmm import MultinomialHMM
    >>> MultinomialHMM(n_components=2)
    ...                             #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    MultinomialHMM(algorithm='viterbi',...
    """

    def __init__(self, n_components=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="ste", init_params="ste"):
        _BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm,
                          random_state=random_state,
                          n_iter=n_iter, tol=tol, verbose=verbose,
                          params=params, init_params=init_params)

    def _init(self, X, lengths=None):
        if not self._check_input_symbols(X):
            raise ValueError("expected a sample from "
                             "a Multinomial distribution.")

        super(MultinomialHMM, self)._init(X, lengths=lengths)
        self.random_state = check_random_state(self.random_state)

        if 'e' in self.init_params:
            if not hasattr(self, "n_features"):
                symbols = set()
                for i, j in iter_from_X_lengths(X, lengths):
                    symbols |= set(X[i:j].flatten())
                self.n_features = len(symbols)
            self.emissionprob_ = self.random_state \
                .rand(self.n_components, self.n_features)
            normalize(self.emissionprob_, axis=1)

    def _check(self):
        super(MultinomialHMM, self)._check()

        self.emissionprob_ = np.atleast_2d(self.emissionprob_)
        n_features = getattr(self, "n_features", self.emissionprob_.shape[1])
        if self.emissionprob_.shape != (self.n_components, n_features):
            raise ValueError(
                "emissionprob_ must have shape (n_components, n_features)")
        else:
            self.n_features = n_features

    def _compute_log_likelihood(self, X):
        return np.log(self.emissionprob_)[:, np.concatenate(X)].T

    def _generate_sample_from_state(self, state, random_state=None):
        cdf = np.cumsum(self.emissionprob_[state, :])
        random_state = check_random_state(random_state)
        return [(cdf > random_state.rand()).argmax()]

    def _initialize_sufficient_statistics(self):
        stats = super(MultinomialHMM, self)._initialize_sufficient_statistics()
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super(MultinomialHMM, self)._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice)
        if 'e' in self.params:
            for t, symbol in enumerate(np.concatenate(X)):
                stats['obs'][:, symbol] += posteriors[t]

    def _do_mstep(self, stats):
        super(MultinomialHMM, self)._do_mstep(stats)
        if 'e' in self.params:
            self.emissionprob_ = (stats['obs']
                                  / stats['obs'].sum(1)[:, np.newaxis])

    def _check_input_symbols(self, X):
        """Check if ``X`` is a sample from a Multinomial distribution.

        That is ``X`` should be an array of non-negative integers from
        range ``[min(X), max(X)]``, such that each integer from the range
        occurs in ``X`` at least once.

        For example ``[0, 0, 2, 1, 3, 1, 1]`` is a valid sample from a
        Multinomial distribution, while ``[0, 0, 3, 5, 10]`` is not.
        """
        symbols = np.concatenate(X)
        if (len(symbols) == 1 or          # not enough data
            symbols.dtype.kind != 'i' or  # not an integer
            (symbols < 0).any()):         # contains negative integers
            return False

        symbols.sort()
        return np.all(np.diff(symbols) <= 1)


class GMMHMM(_BaseHMM):
    """Hidden Markov Model with Gaussian mixture emissions.

    Parameters
    ----------
    n_components : int
        Number of states in the model.

    n_mix : int
        Number of states in the GMM.

    covariance_type : string
        String describing the type of covariance parameters to
        use.  Must be one of

        * "spherical" --- each state uses a single variance value that
          applies to all features;
        * "diag" --- each state uses a diagonal covariance matrix;
        * "full" --- each state uses a full (i.e. unrestricted)
          covariance matrix;
        * "tied" --- all states use **the same** full covariance matrix.

        Defaults to "diag".

    startprob_prior : array, shape (n_components, )
        Initial state occupation prior distribution.

    transmat_prior : array, shape (n_components, n_components)
        Matrix of prior transition probabilities between states.

    algorithm : string
        Decoder algorithm. Must be one of "viterbi" or "map".
        Defaults to "viterbi".

    random_state: RandomState or an int seed
        A random number generator instance.

    n_iter : int, optional
        Maximum number of iterations to perform.

    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        is below this value.

    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.

    init_params : string, optional
        Controls which parameters are initialized prior to training. Can
        contain any combination of 's' for startprob, 't' for transmat, 'm'
        for means, 'c' for covars, and 'w' for GMM mixing weights.
        Defaults to all parameters.

    params : string, optional
        Controls which parameters are updated in the training process.  Can
        contain any combination of 's' for startprob, 't' for transmat, 'm' for
        means, and 'c' for covars, and 'w' for GMM mixing weights.
        Defaults to all parameters.

    Attributes
    ----------
    monitor\_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob\_ : array, shape (n_components, )
        Initial state occupation distribution.

    transmat\_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    gmms\_ : list of GMM objects, length n_components
        GMM emission distributions for each state.

    Examples
    --------
    >>> from hmmlearn.hmm import GMMHMM
    >>> GMMHMM(n_components=2, n_mix=10, covariance_type='diag')
    ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    GMMHMM(algorithm='viterbi', covariance_type='diag',...
    """

    def __init__(self, n_components=1, n_mix=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 covariance_type='diag', covars_prior=1e-2,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="stmcw", init_params="stmcw"):
        _BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm, random_state=random_state,
                          n_iter=n_iter, tol=tol, verbose=verbose,
                          params=params, init_params=init_params)

        if covariance_type != "diag":
            warnings.warn("Fitting a GMMHMM with {0!r} covariance type "
                          "is broken in 0.2.0. Please update to 0.2.1 once "
                          "it's available.".format(covariance_type),
                          UserWarning)

        # XXX: Hotfit for n_mix that is incompatible with the scikit's
        # BaseEstimator API
        self.n_mix = n_mix
        self.covariance_type = covariance_type
        self.covars_prior = covars_prior
        self.gmms_ = []
        for x in range(self.n_components):
            if covariance_type is None:
                gmm = GMM(n_mix, random_state=self.random_state)
            else:
                gmm = GMM(n_mix, covariance_type=covariance_type,
                        random_state=self.random_state)
            self.gmms_.append(gmm)

    def _init(self, X, lengths=None):
        super(GMMHMM, self)._init(X, lengths=lengths)

        for g in self.gmms_:
            g.set_params(init_params=self.init_params, n_iter=0)
            g.fit(X)

    def _compute_log_likelihood(self, X):
        return np.array([g.score(X) for g in self.gmms_]).T

    def _generate_sample_from_state(self, state, random_state=None):
        return self.gmms_[state].sample(1, random_state=random_state).flatten()

    def _initialize_sufficient_statistics(self):
        stats = super(GMMHMM, self)._initialize_sufficient_statistics()
        stats['norm'] = [np.zeros(g.weights_.shape) for g in self.gmms_]
        stats['means'] = [np.zeros(np.shape(g.means_)) for g in self.gmms_]
        stats['covars'] = [np.zeros(np.shape(g.covars_)) for g in self.gmms_]
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super(GMMHMM, self)._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice)

        for state, g in enumerate(self.gmms_):
            lgmm_posteriors = (np.log(g.predict_proba(X))
                               + np.log(posteriors[:, state][:, np.newaxis]
                                        + np.finfo(np.float).eps))
            gmm_posteriors = np.exp(lgmm_posteriors)

            n_features = g.means_.shape[1]
            tmp_gmm = GMM(g.n_components, covariance_type=g.covariance_type)
            tmp_gmm._set_covars(
                distribute_covar_matrix_to_match_covariance_type(
                    np.eye(n_features), g.covariance_type,
                    g.n_components))
            norm = tmp_gmm._do_mstep(X, gmm_posteriors, self.params)

            if np.any(np.isnan(tmp_gmm.covars_)):
                raise ValueError

            stats['norm'][state] += norm
            if 'm' in self.params:
                stats['means'][state] += tmp_gmm.means_ * norm[:, np.newaxis]
            if 'c' in self.params:
                if tmp_gmm.covariance_type == 'tied':
                    stats['covars'][state] += tmp_gmm.covars_ * norm.sum()
                else:
                    cvnorm = np.copy(norm)
                    shape = np.ones(tmp_gmm.covars_.ndim, dtype=np.int)
                    shape[0] = np.shape(tmp_gmm.covars_)[0]
                    cvnorm.shape = shape
                    stats['covars'][state] += (tmp_gmm.covars_
                                               + tmp_gmm.means_**2) * cvnorm

    def _do_mstep(self, stats):
        super(GMMHMM, self)._do_mstep(stats)

        # All that is left to do is to apply covars_prior to the
        # parameters updated in _accumulate_sufficient_statistics.
        for state, g in enumerate(self.gmms_):
            n_features = g.means_.shape[1]
            norm = stats['norm'][state]
            if 'w' in self.params:
                g.weights_ = norm.copy()
                normalize(g.weights_)
            if 'm' in self.params:
                g.means_ = stats['means'][state] / norm[:, np.newaxis]
            if 'c' in self.params:
                if g.covariance_type == 'tied':
                    g.covars_ = ((stats['covars'][state]
                                 + self.covars_prior * np.eye(n_features))
                                 / norm.sum())
                else:
                    cvnorm = np.copy(norm)
                    shape = np.ones(g.covars_.ndim, dtype=np.int)
                    shape[0] = np.shape(g.covars_)[0]
                    cvnorm.shape = shape
                    if g.covariance_type in ['spherical', 'diag']:
                        g.covars_ = (stats['covars'][state] +
                                     self.covars_prior) / cvnorm - g.means_**2
                    elif g.covariance_type == 'full':
                        eye = np.eye(n_features)
                        g.covars_ = ((stats['covars'][state]
                                     + self.covars_prior * eye[np.newaxis])
                                     / cvnorm) - g.means_**2




class NegativeBinomialHMM(_BaseHMM):
    """Hidden Markov Model with negative binomial emissions.

    Parameters
    ----------
    n_components : int
        Number of states.

    startprob_prior : array, shape (n_components, )
        Initial state occupation prior distribution.

    transmat_prior : array, shape (n_components, n_components)
        Matrix of prior transition probabilities between states.

    algorithm : string
        Decoder algorithm. Must be one of "viterbi" or "map".
        Defaults to "viterbi".

    random_state: RandomState or an int seed
        A random number generator instance.

    n_iter : int, optional
        Maximum number of iterations to perform.

    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        is below this value.

    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, 'p' and 'r' for p and r parameters of 
        negative binomial distributions. Defaults to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, 'p' and 'r' for p and r parameters
        of negative binomial distributions. Defaults to all parameters.

    Attributes
    ----------
    n_features : int
        Dimensionality of the Gaussian emissions.

    monitor\_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    transmat\_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    startprob\_ : array, shape (n_components, )
        Initial state occupation distribution.

    """
    def __init__(self, n_components=1, 
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 covars_prior=1e-2, covars_weight=1,
                 min_covar=1e-3,
                 params="stmcpr", init_params="stmcpr",
                 p=1, r=0, s=0, m=0):
        _BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          tol=tol, params=params, verbose=verbose,
                          init_params=init_params)
        self.p = p
        self.r = r
        self.alpha = self.r
        self.beta = (1 - self.p) / self.p
        self.means_ = m  # means?
        self.covars_ = s  # sdev?
        # byc moze wole trzymac srednia i wariancje
        # albo jakas kombinacje tych dwoch wariantow
        self.min_covar = min_covar
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight
        self.covars_le_means = 0

    def _init(self, X, lengths=None):
        super(NegativeBinomialHMM, self)._init(X, lengths=lengths)

        _, n_features = X.shape
        if hasattr(self, 'n_features') and self.n_features != n_features:
            raise ValueError('Unexpected number of dimensions, got %s but '
                             'expected %s' % (n_features, self.n_features))
        self.n_features = n_features
        if 'p' or 'r' in self.init_params:
            kmeans = cluster.KMeans(n_clusters=self.n_components,
                                    random_state=self.random_state)
            kmeans.fit(X)
            means = kmeans.cluster_centers_
            #logging.debug("MEANS:", means)
            means = np.sort(means, axis = 0)
            #logging.debug("MEANS:", means)
            self.means_ = means
            cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
            if not cv.shape:
                cv.shape = (1, 1)
            cv = distribute_covar_matrix_to_match_covariance_type(
                cv, 'diag', self.n_components).copy()
            if cv.min() < means.max():
                cv += means.max()
            #logging.debug("CV:", cv)
            self.covars_ = cv
        if 'p' in self.init_params or not hasattr(self, "p"):
            self.p = 1 - ((cv-means) / cv)
            #logging.debug("P:", self.p)
        if 'r' in self.init_params or not hasattr(self, "r"):
            self.r = means**2 / (cv - means)
        self.alpha = self.r
        self.beta = (1 - self.p) / self.p
            #logging.debug("R:", self.r)
        #logging.debug("transmat:", self.transmat_)
 

        """ w Multinomial:
         if not self._check_input_symbols(X):
                    raise ValueError("expected a sample from "
                                     "a Multinomial distribution.")

                super(MultinomialHMM, self)._init(X, lengths=lengths)
                self.random_state = check_random_state(self.random_state)

                if 'e' in self.init_params:
                    if not hasattr(self, "n_features"):
                        symbols = set()
                        for i, j in iter_from_X_lengths(X, lengths):
                            symbols |= set(X[i:j].flatten())
                        self.n_features = len(symbols)
                    self.emissionprob_ = self.random_state \
                        .rand(self.n_components, self.n_features)
                    normalize(self.emissionprob_, axis=1)
        # w Gaussian jest jeszcze takie cos -
        # szacowanie srednich i kowariancji
        if 'm' in self.init_params or not hasattr(self, "means_"):
            kmeans = cluster.KMeans(n_clusters=self.n_components,
                                    random_state=self.random_state)
            kmeans.fit(X)
            self.means_ = kmeans.cluster_centers_
        if 'c' in self.init_params or not hasattr(self, "covars_"):
            cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
            if not cv.shape:
                cv.shape = (1, 1)
            self._covars_ = distribute_covar_matrix_to_match_covariance_type(
                cv, self.covariance_type, self.n_components).copy()
        """


    def _check(self):
        super(NegativeBinomialHMM, self)._check()
        
        """ w multinomial:
        self.emissionprob_ = np.atleast_2d(self.emissionprob_)
        n_features = getattr(self, "n_features", self.emissionprob_.shape[1])
        if self.emissionprob_.shape != (self.n_components, n_features):
            raise ValueError(
                "emissionprob_ must have shape (n_components, n_features)")
        else:
            self.n_features = n_features
        """

    def _compute_log_likelihood(self, X):
        #logging.debug("compute log likelihood startprob:", self.startprob_)
        n_observations, n_dim = X.shape
        #logging.debug("n_dim:", n_dim)
        #logging.debug("n_features:", self.n_features)
        #n_dim i n_features to to samo
        #logging.debug("n_observations:", n_observations)
        #logging.debug("n_components:", self.n_components)
        r, p = self.r, self.p
        log_likelihood = np.ndarray((n_observations, self.n_components))
        for i in xrange(n_observations):
            for j in xrange(self.n_components):
                #print "log_likelihood, i = %d, j = %d" % (i, j)
                #logging.debug("X:", X[i,:])
                #print "r:", r[j,:]
                #print "p:", p[j,:]
                log_likelihood[i, j] = np.sum(nbinom.logpmf(X[i,:], r[j,:], p[j,:]))
                #print "log_likelihood = %f" % log_likelihood[i, j]
                #if log_likelihood[i,j] == -1 * np.inf:
                #    logging.debug("Uwaga, zle likelihood w iteracji nr:", i, j)
                #    logging.debug("likelihood:", log_likelihood[i,j])
                #    logging.debug("X:", X[i,:])
                #    logging.debug("r:", r[j:])
                #    logging.debug("p:", p[j,:])
                    
        #       czy na pewno suma?
        #       no tak, a co innego
        #        log_likelihood[i,j] = (np.sum(np.log(factorial(X[i,:]+r[:,j])))
        #                         - np.sum(np.log(factorial(X[i,:])))
        #                         - n_dim * np.log(factorial(r[:,j]))
        #                         + np.sum(X[i,:] * np.log(p[:,j]))
        #                         + n_dim * r[:,j] * np.log(1-p[:,j]))
        #                           to jest niewlasciwego wymiaru
        #                           jest wymiaru (n_features,),
        #                           czyli tyle ile jest bedgraphow
        #                           powinno sie to, hm, wymnozyc? dodac?
        #logging.debug("log_likelihood:")
        #logging.debug(log_likelihood)
        #if np.any(log_likelihood == -1 * np.inf):
        #    logging.debug("means:")
        #    logging.debug(self.means_)
        #    logging.debug("covars:")
        #    logging.debug(self.covars_)
        return log_likelihood



    def _generate_sample_from_state(self, state, random_state=None):
        #cv = self.covars_
        #means = self.means_
        p = self.p
        r = self.r
        #if 'p' in self.init_params or not hasattr(self, "p"):
        #    self.p = 1 - ((cv-means) / cv)
        #if 'r' in self.init_params or not hasattr(self, "r"):
        #    self.r = means**2 / (cv - means)
        return [nbinom(self.r[state][feature], self.p[state][feature]).rvs()
                for feature in xrange(self.n_features)]

    def _initialize_sufficient_statistics(self):
        stats = super(NegativeBinomialHMM, self)._initialize_sufficient_statistics()
        #n_features, n_samples = obs.shape
        #stats['t'] = np.zeros((self.n_components,1 ))
            #stats['s'] = np.zeros((self.n_components, 1))#, n_samples)
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        stats['obs**2'] = np.zeros((self.n_components, self.n_features))
        #if self.covariance_type in ('tied', 'full'):
        #    stats['obs*obs.T'] = np.zeros((self.n_components, self.n_features,
        #                                   self.n_features))
        return stats
        # w multinomial:
        stats = super(MultinomialHMM, self)._initialize_sufficient_statistics()
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        #stats['x'] = np.zeros((1, self.n_components))
        return stats


    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super(NegativeBinomialHMM, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice)

        n_samples, n_features = obs.shape
        stats['post'] += posteriors.sum(axis=0)
        stats['obs'] += np.dot(posteriors.T, obs)
        if 'x' not in stats.keys():
            logging.debug("Pierwszy chromosom")
            stats['x'] = obs
            logging.debug("stats[x] shape: %s", str(stats['x'].shape))
        else:
            logging.debug("Kolejny chromosom")
            stats['x'] = np.append(stats['x'], obs, axis=0)
            logging.debug("stats[x] shape: %s", str(stats['x'].shape))
        if 'posteriors' not in stats.keys():
            stats['posteriors'] = posteriors
        else:
            stats['posteriors'] = np.append(stats['posteriors'],
                                            posteriors, axis=0)
        


        #stats['x'] = obs    # achtung - to i to nizej nie zadziala dla wielu chromosomow!
        #stats['posteriors'] = posteriors

        #print(obs.shape)
        #print(self.alpha)
        #stats['t'] += stats['post'] #(obs + self.alpha) #/ (1 + self.beta)
        #stats['t'] = posteriors
        #for line in obs:
        #    t = (line + self.alpha) / (1 + self.beta)
        #    s = digamma(line + self.alpha) - np.log(self.beta + 1)
        #    #print "stats[s]:", stats['s'].shape
        #    #print "s:", s.shape
        #    stats['t'] = np.append(stats['t'], t, axis=1)
        #    stats['s'] = np.append(stats['s'], s, axis=1)
        #stats['t'] = np.delete(stats['t'], 0, axis=1)
        #stats['s'] = np.delete(stats['s'], 0, axis=1)
        #alpha = np.repeat(self.alpha, n_samples, axis = 1).T
        #print(alpha.shape)
        #s = digamma(alpha + obs) - np.log(self.beta + 1).T
        #s_means = np.mean(s, axis=0)
        #print("s shape:")
        #print(s.shape)
        #print("t shape:")
        #print(stats['t'].shape)
        #print("s_means shape:")
        #print(s_means.shape)
        #print("stats[s] shape:")
        #print(stats['s'].shape)
        #stats['s'] += s.T #np.mean(s, axis=0)
        #stats['obs'] += np.dot(posteriors.T, obs)
        stats['obs**2'] += np.dot(posteriors.T, obs ** 2)


        """
        if 'm' in self.params or 'c' in self.params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += np.dot(posteriors.T, obs)

        if 'c' in self.params:
            if self.covariance_type in ('spherical', 'diag'):
                stats['obs**2'] += np.dot(posteriors.T, obs ** 2)
            elif self.covariance_type in ('tied', 'full'):
                # posteriors: (nt, nc); obs: (nt, nf); obs: (nt, nf)
                # -> (nc, nf, nf)
                stats['obs*obs.T'] += np.einsum(
                    'ij,ik,il->jkl', posteriors, obs, obs)
        # w multinomial:
        super(MultinomialHMM, self)._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice)
        if 'e' in self.params:
            for t, symbol in enumerate(np.concatenate(X)):
                stats['obs'][:, symbol] += posteriors[t]
        """

    def _do_mstep(self, stats):
        super(NegativeBinomialHMM, self)._do_mstep(stats)

        #print "update p"
        #print "stats[obs] shape:", stats['obs'].shape
        #print "stats[post] shape:", stats['post'][:, np.newaxis].shape
        #print "r shape:", self.r.shape
        #print "self.r * stats[post] shape:", (self.r * stats['post'][:, np.newaxis]).shape
        
        p = stats['obs'] / (self.r * stats['post'][:, np.newaxis] + stats['obs'])
        logging.debug("new p: %s", str(self.p))

        # p = ( sum_t (P_t * o_t) ) / ( r * sum_t P_t + sum_t (P_t * o_t)  )


        denom = stats['post'][:, np.newaxis]
        #logging.debug("denom: %s", str(denom))
        means = stats['obs'] / denom
        covars_prior = self.covars_prior
        covars_weight = self.covars_weight
        cv_num = (stats['obs**2']
                  - 2 * means * stats['obs']
                  + means**2 * denom)
        cv_den = max(covars_weight - 1, 0) + denom
        covars = (covars_prior + cv_num) / np.maximum(cv_den, 1e-5)

        r_initial = means**2 / (covars - means)
        if np.any(np.isnan(r_initial)):
            print "r_initial jest nan, paczaj:"
            print r_initial
            r_initial = self.r

        r = finding_r.find_r(r_initial, stats['x'],
                             stats['posteriors'], self.p)
        self.r = r      
        logging.debug("new r: %s", str(self.r))
            
        self.means_ = self.p * self.r / (1 - self.p)
        self.covars_ = self.p * self.r / (1 - self.p) ** 2



        """
        mean_t = np.mean(stats['t'], axis=1)[:, np.newaxis]
        beta = self.alpha / mean_t
        #print "beta", beta.shape
        print "Uaktualniamy beta:"
        print "self.alpha:", self.alpha
        print "np.mean(stats[t]):", mean_t
        self.beta = beta
        
        mean_s = np.mean(stats['s'], axis=1)[:, np.newaxis]
        num = digamma(self.alpha) + np.log(self.beta) - mean_s
        print "beta:", self.beta
        print "Uaktualniamy alpha:"
        print "np.log beta:", np.log(self.beta)
        print "np.mean(stats[s]):", mean_s
        print "num:", num
        denom = polygamma(1, self.alpha)
        print "denom:", denom
        alpha = self.alpha - num / denom
        print "alpha:", alpha
        self.alpha = alpha

        self.r = self.alpha
        self.p = 1 / (1 + self.beta)

        self.means_ = self.p * self.r / (1 - self.p)
        self.covars_ = self.p * self.r / (1 - self.p) ** 2
        """
        """
        #logging.debug("stats[post]: %s", str(stats['post']))
        denom = stats['post'][:, np.newaxis]
        #logging.debug("denom: %s", str(denom))
        if 'm' in self.params:
            means = stats['obs'] / denom
            self.means_ = means

        if 'c' in self.params:
            covars_prior = self.covars_prior
            covars_weight = self.covars_weight

            cv_num = (stats['obs**2']
                      - 2 * means * stats['obs']
                      + means**2 * denom)
            #cv_den = denom
            cv_den = max(covars_weight - 1, 0) + denom
            covars = (covars_prior + cv_num) / np.maximum(cv_den, 1e-5)
            self.covars_ = covars

        #logging.debug("mstep, prawie koniec; means:", means)
        #logging.debug("mstep, prawie koniec; covars:", covars)

        #if not np.all(covars > means):
        if np.any(covars <= means):
            logging.debug("covars <= means")
            self.covars_le_means += 1
            covars = np.ones((self.n_components, self.n_features)) * (means.max() + 1)
            # powinnam zwiekszac tylko wariancje tego stanu.
        else:
            logging.debug("covars ok")
      

        #logging.debug("mstep, koniec; means:", means
        #logging.debug("mstep, koniec; covars:", covars

        if 'p' in self.params:
            self.p = 1 - ( (covars-means) / covars)

            #logging.debug("mstep, koniec; p:")
            #logging.debug(self.p)

        if 'r' in self.params:
            self.r = means**2 / (covars - means)
            #logging.debug("mstep, koniec; r:")
            #logging.debug(self.r)
        """


def factorial(x):
    f = np.vectorize(math.factorial)
    return f(x)


