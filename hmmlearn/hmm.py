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

import sys
import string
import logging
import hmmlearn.mynumpy as np
from scipy.special import logsumexp
from scipy.stats import nbinom
from sklearn import cluster
from sklearn.utils import check_random_state

import hmmlearn.finding_r

from . import _utils
from .stats import log_multivariate_normal_density
from .base import _BaseHMM
from .utils import iter_from_X_lengths, normalize, fill_covars, array2str


__all__ = ["GMMHMM", "GaussianHMM", "MultinomialHMM", "NegativeBinomialHMM"]

COVARIANCE_TYPES = frozenset(("spherical", "diag", "full", "tied"))


class GaussianHMM(_BaseHMM):
    r"""Hidden Markov Model with Gaussian emissions.

    Parameters
    ----------
    n_components : int
        Number of states.

    covariance_type : string, optional
        String describing the type of covariance parameters to
        use.  Must be one of

        * "spherical" --- each state uses a single variance value that
          applies to all features.
        * "diag" --- each state uses a diagonal covariance matrix.
        * "full" --- each state uses a full (i.e. unrestricted)
          covariance matrix.
        * "tied" --- all states use **the same** full covariance matrix.

        Defaults to "diag".

    min_covar : float, optional
        Floor on the diagonal of the covariance matrix to prevent
        overfitting. Defaults to 1e-3.

    startprob_prior : array, shape (n_components, ), optional
        Parameters of the Dirichlet prior distribution for
        :attr:`startprob_`.

    transmat_prior : array, shape (n_components, n_components), optional
        Parameters of the Dirichlet prior distribution for each row
        of the transition probabilities :attr:`transmat_`.

    means_prior, means_weight : array, shape (n_components, ), optional
        Mean and precision of the Normal prior distribtion for
        :attr:`means_`.

    covars_prior, covars_weight : array, shape (n_components, ), optional
        Parameters of the prior distribution for the covariance matrix
        :attr:`covars_`.

        If :attr:`covariance_type` is "spherical" or "diag" the prior is
        the inverse gamma distribution, otherwise --- the inverse Wishart
        distribution.

    algorithm : string, optional
        Decoder algorithm. Must be one of "viterbi" or`"map".
        Defaults to "viterbi".

    random_state: RandomState or an int seed, optional
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

        The shape depends on :attr:`covariance_type`::

            (n_components, )                        if "spherical",
            (n_features, n_features)                if "tied",
            (n_components, n_features)              if "diag",
            (n_components, n_features, n_features)  if "full"

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
                 params="stmc", init_params="stmc",
                 debug_prefix=None):
        _BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          tol=tol, params=params, verbose=verbose,
                          init_params=init_params,
                          debug_prefix=debug_prefix)

        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.means_prior = means_prior
        self.means_weight = means_weight
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight

    @property
    def covars_(self):
        """Return covars as a full matrix."""
        return fill_covars(self._covars_, self.covariance_type,
                           self.n_components, self.n_features)

    @covars_.setter
    def covars_(self, covars):
        self._covars_ = np.asarray(covars).copy()

    def _check(self):
        super(GaussianHMM, self)._check()

        self.means_ = np.asarray(self.means_)
        self.n_features = self.means_.shape[1]

        if self.covariance_type not in COVARIANCE_TYPES:
            raise ValueError('covariance_type must be one of {}'
                             .format(COVARIANCE_TYPES))

        _utils._validate_covars(self._covars_, self.covariance_type,
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
            means = kmeans.cluster_centers_
            means = np.sort(means, axis=0)
            self.means_ = means
        logging.debug("Initial means:")
        logging.debug(self.means_)
        if 'c' in self.init_params or not hasattr(self, "covars_"):
            cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
            if not cv.shape:
                cv.shape = (1, 1)
            self._covars_ = \
                _utils.distribute_covar_matrix_to_match_covariance_type(
                    cv, self.covariance_type, self.n_components).copy()
        logging.debug("Initial covars:")
        logging.debug(self.covars_)
        if self.debug_prefix is not None:
             with open("%smeans_%d" %
                       (self.debug_prefix, self.iteration), 'w') as output:
                output.write(array2str(self.means_))
             with open("%scovars_%d" %
                       (self.debug_prefix, self.iteration), 'w') as output:
                output.write(array2str(self.covars_))

    def _compute_log_likelihood(self, X):
        lmnd = log_multivariate_normal_density(
            X, self.means_, self._covars_, self.covariance_type)
        lmnd = lmnd.astype('float128')
        return lmnd

    def _generate_sample_from_state(self, state, random_state=None):
        random_state = check_random_state(random_state)
        return random_state.multivariate_normal(
            self.means_[state], self.covars_[state]
        )

    def _initialize_sufficient_statistics(self):
        stats = super(GaussianHMM, self)._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        stats['obs**2'] = np.zeros((self.n_components, self.n_features))
        if self.covariance_type in ('tied', 'full'):
            stats['obs*obs.T'] = np.zeros((self.n_components, self.n_features,
                                           self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super(GaussianHMM, self)._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice)

        if 'm' in self.params or 'c' in self.params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += np.dot(posteriors.T, X)

        if 'c' in self.params:
            if self.covariance_type in ('spherical', 'diag'):
                stats['obs**2'] += np.dot(posteriors.T, X ** 2)
            elif self.covariance_type in ('tied', 'full'):
                # posteriors: (nt, nc); obs: (nt, nf); obs: (nt, nf)
                # -> (nc, nf, nf)
                stats['obs*obs.T'] += np.einsum(
                    'ij,ik,il->jkl', posteriors, X, X)

    def _do_mstep(self, stats):
        super(GaussianHMM, self)._do_mstep(stats)

        means_prior = self.means_prior
        means_weight = self.means_weight

        # TODO: find a proper reference for estimates for different
        #       covariance models.
        # Based on Huang, Acero, Hon, "Spoken Language Processing",
        # p. 443 - 445
        denom = stats['post'][:, np.newaxis]
        if 'm' in self.params:
            logging.debug("old means:")
            logging.debug(self.means_)
            self.means_ = ((means_weight * means_prior + stats['obs'])
                           / (means_weight + denom))
            logging.debug("new means:")
            logging.debug(self.means_)


        if 'c' in self.params:
            logging.debug("old covars:")
            logging.debug(self.covars_)
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
            logging.debug("new covars:")
            logging.debug(self.covars_)
        if self.debug_prefix is not None:
             with open("%smeans_%d" %
                       (self.debug_prefix, self.iteration), 'w') as output:
                output.write(array2str(self.means_))
             with open("%scovars_%d" %
                           (self.debug_prefix, self.iteration), 'w') as output:
                output.write(array2str(self.covars_))

class MultinomialHMM(_BaseHMM):
    r"""Hidden Markov Model with multinomial (discrete) emissions

    Parameters
    ----------

    n_components : int
        Number of states.

    startprob_prior : array, shape (n_components, ), optional
        Parameters of the Dirichlet prior distribution for
        :attr:`startprob_`.

    transmat_prior : array, shape (n_components, n_components), optional
        Parameters of the Dirichlet prior distribution for each row
        of the transition probabilities :attr:`transmat_`.

    algorithm : string, optional
        Decoder algorithm. Must be one of "viterbi" or "map".
        Defaults to "viterbi".

    random_state: RandomState or an int seed, optional
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
    # TODO: accept the prior on emissionprob_ for consistency.
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
                                  / stats['obs'].sum(axis=1)[:, np.newaxis])

    def _check_input_symbols(self, X):
        """Check if ``X`` is a sample from a Multinomial distribution.

        That is ``X`` should be an array of non-negative integers from
        range ``[min(X), max(X)]``, such that each integer from the range
        occurs in ``X`` at least once.

        For example ``[0, 0, 2, 1, 3, 1, 1]`` is a valid sample from a
        Multinomial distribution, while ``[0, 0, 3, 5, 10]`` is not.
        """
        symbols = np.concatenate(X)
        if (len(symbols) == 1                                    # not enough data
                or not np.issubdtype(symbols.dtype, np.integer)  # not an integer
                or (symbols < 0).any()):                         # not positive
            return False
        u = np.unique(symbols)
        return u[0] == 0 and u[-1] == len(u) - 1


class GMMHMM(_BaseHMM):
    r"""Hidden Markov Model with Gaussian mixture emissions.

    Parameters
    ----------
    n_components : int
        Number of states in the model.

    n_mix : int
        Number of states in the GMM.

    covariance_type : string, optional
        String describing the type of covariance parameters to
        use.  Must be one of

        * "spherical" --- each state uses a single variance value that
          applies to all features.
        * "diag" --- each state uses a diagonal covariance matrix.
        * "full" --- each state uses a full (i.e. unrestricted)
          covariance matrix.
        * "tied" --- all states use **the same** full covariance matrix.

        Defaults to "diag".

    min_covar : float, optional
        Floor on the diagonal of the covariance matrix to prevent
        overfitting. Defaults to 1e-3.

    startprob_prior : array, shape (n_components, ), optional
        Parameters of the Dirichlet prior distribution for
        :attr:`startprob_`.

    transmat_prior : array, shape (n_components, n_components), optional
        Parameters of the Dirichlet prior distribution for each row
        of the transition probabilities :attr:`transmat_`.

    weights_prior : array, shape (n_mix, ), optional
        Parameters of the Dirichlet prior distribution for
        :attr:`weights_`.

    means_prior, means_weight : array, shape (n_mix, ), optional
        Mean and precision of the Normal prior distribtion for
        :attr:`means_`.

    covars_prior, covars_weight : array, shape (n_mix, ), optional
        Parameters of the prior distribution for the covariance matrix
        :attr:`covars_`.

        If :attr:`covariance_type` is "spherical" or "diag" the prior is
        the inverse gamma distribution, otherwise --- the inverse Wishart
        distribution.

    algorithm : string, optional
        Decoder algorithm. Must be one of "viterbi" or "map".
        Defaults to "viterbi".

    random_state: RandomState or an int seed, optional
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

    weights\_ : array, shape (n_components, n_mix)
        Mixture weights for each state.

    means\_ : array, shape (n_components, n_mix)
        Mean parameters for each mixture component in each state.

    covars\_ : array
        Covariance parameters for each mixture components in each state.

        The shape depends on :attr:`covariance_type`::

            (n_components, n_mix)                          if "spherical",
            (n_components, n_features, n_features)         if "tied",
            (n_components, n_mix, n_features)              if "diag",
            (n_components, n_mix, n_features, n_features)  if "full"
    """

    def __init__(self, n_components=1, n_mix=1,
                 min_covar=1e-3, startprob_prior=1.0, transmat_prior=1.0,
                 weights_prior=1.0, means_prior=0.0, means_weight=0.0,
                 covars_prior=None, covars_weight=None,
                 algorithm="viterbi", covariance_type="diag",
                 random_state=None, n_iter=10, tol=1e-2,
                 verbose=False, params="stmcw",
                 init_params="stmcw"):
        _BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm, random_state=random_state,
                          n_iter=n_iter, tol=tol, verbose=verbose,
                          params=params, init_params=init_params)
        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.n_mix = n_mix
        self.weights_prior = weights_prior
        self.means_prior = means_prior
        self.means_weight = means_weight
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight

    def _init(self, X, lengths=None):
        super(GMMHMM, self)._init(X, lengths=lengths)

        _n_samples, self.n_features = X.shape

        # Default values for covariance prior parameters
        self._init_covar_priors()
        self._fix_priors_shape()

        main_kmeans = cluster.KMeans(n_clusters=self.n_components,
                                     random_state=self.random_state)
        labels = main_kmeans.fit_predict(X)
        kmeanses = []
        for label in range(self.n_components):
            kmeans = cluster.KMeans(n_clusters=self.n_mix,
                                    random_state=self.random_state)
            kmeans.fit(X[np.where(labels == label)])
            kmeanses.append(kmeans)

        if 'w' in self.init_params or not hasattr(self, "weights_"):
            self.weights_ = (np.ones((self.n_components, self.n_mix)) /
                             (np.ones((self.n_components, 1)) * self.n_mix))

        if 'm' in self.init_params or not hasattr(self, "means_"):
            self.means_ = np.zeros((self.n_components, self.n_mix,
                                    self.n_features))
            for i, kmeans in enumerate(kmeanses):
                self.means_[i] = kmeans.cluster_centers_

        if 'c' in self.init_params or not hasattr(self, "covars_"):
            cv = np.cov(X.T) + self.min_covar * np.eye(self.n_features)
            if not cv.shape:
                cv.shape = (1, 1)

            if self.covariance_type == 'tied':
                self.covars_ = np.zeros((self.n_components,
                                         self.n_features, self.n_features))
                self.covars_[:] = cv
            elif self.covariance_type == 'full':
                self.covars_ = np.zeros((self.n_components, self.n_mix,
                                         self.n_features, self.n_features))
                self.covars_[:] = cv
            elif self.covariance_type == 'diag':
                self.covars_ = np.zeros((self.n_components, self.n_mix,
                                         self.n_features))
                self.covars_[:] = np.diag(cv)
            elif self.covariance_type == 'spherical':
                self.covars_ = np.zeros((self.n_components, self.n_mix))
                self.covars_[:] = cv.mean()

    def _init_covar_priors(self):
        if self.covariance_type == "full":
            if self.covars_prior is None:
                self.covars_prior = 0.0
            if self.covars_weight is None:
                self.covars_weight = -(1.0 + self.n_features + 1.0)
        elif self.covariance_type == "tied":
            if self.covars_prior is None:
                self.covars_prior = 0.0
            if self.covars_weight is None:
                self.covars_weight = -(self.n_mix + self.n_features + 1.0)
        elif self.covariance_type == "diag":
            if self.covars_prior is None:
                self.covars_prior = -1.5
            if self.covars_weight is None:
                self.covars_weight = 0.0
        elif self.covariance_type == "spherical":
            if self.covars_prior is None:
                self.covars_prior = -(self.n_mix + 2.0) / 2.0
            if self.covars_weight is None:
                self.covars_weight = 0.0

    def _fix_priors_shape(self):
        # If priors are numbers, this function will make them into a
        # matrix of proper shape
        self.weights_prior = np.broadcast_to(
            self.weights_prior, (self.n_components, self.n_mix)).copy()
        self.means_prior = np.broadcast_to(
            self.means_prior,
            (self.n_components, self.n_mix, self.n_features)).copy()
        self.means_weight = np.broadcast_to(
            self.means_weight,
            (self.n_components, self.n_mix)).copy()

        if self.covariance_type == "full":
            self.covars_prior = np.broadcast_to(
                self.covars_prior,
                (self.n_components, self.n_mix,
                 self.n_features, self.n_features)).copy()
            self.covars_weight = np.broadcast_to(
                self.covars_weight, (self.n_components, self.n_mix)).copy()
        elif self.covariance_type == "tied":
            self.covars_prior = np.broadcast_to(
                self.covars_prior,
                (self.n_components, self.n_features, self.n_features)).copy()
            self.covars_weight = np.broadcast_to(
                self.covars_weight, self.n_components).copy()
        elif self.covariance_type == "diag":
            self.covars_prior = np.broadcast_to(
                self.covars_prior,
                (self.n_components, self.n_mix, self.n_features)).copy()
            self.covars_weight = np.broadcast_to(
                self.covars_weight,
                (self.n_components, self.n_mix, self.n_features)).copy()
        elif self.covariance_type == "spherical":
            self.covars_prior = np.broadcast_to(
                self.covars_prior, (self.n_components, self.n_mix)).copy()
            self.covars_weight = np.broadcast_to(
                self.covars_weight, (self.n_components, self.n_mix)).copy()

    def _check(self):
        super(GMMHMM, self)._check()

        if not hasattr(self, "n_features"):
            self.n_features = self.means_.shape[2]

        self._init_covar_priors()
        self._fix_priors_shape()

        # Checking covariance type
        if self.covariance_type not in COVARIANCE_TYPES:
            raise ValueError("covariance_type must be one of {}"
                             .format(COVARIANCE_TYPES))

        self.weights_ = np.array(self.weights_)
        # Checking mixture weights' shape
        if self.weights_.shape != (self.n_components, self.n_mix):
            raise ValueError("mixture weights must have shape "
                             "(n_components, n_mix), actual shape: {}"
                             .format(self.weights_.shape))

        # Checking mixture weights' mathematical correctness
        if not np.allclose(np.sum(self.weights_, axis=1),
                           np.ones(self.n_components)):
            raise ValueError("mixture weights must sum up to 1")

        # Checking means' shape
        self.means_ = np.array(self.means_)
        if self.means_.shape != (self.n_components, self.n_mix,
                                 self.n_features):
            raise ValueError("mixture means must have shape "
                             "(n_components, n_mix, n_features), "
                             "actual shape: {}".format(self.means_.shape))

        # Checking covariances' shape
        self.covars_ = np.array(self.covars_)
        covars_shape = self.covars_.shape
        needed_shapes = {
            "spherical": (self.n_components, self.n_mix),
            "tied": (self.n_components, self.n_features, self.n_features),
            "diag": (self.n_components, self.n_mix, self.n_features),
            "full": (self.n_components, self.n_mix,
                     self.n_features, self.n_features)
        }
        needed_shape = needed_shapes[self.covariance_type]
        if covars_shape != needed_shape:
            raise ValueError("{!r} mixture covars must have shape {}, "
                             "actual shape: {}"
                             .format(self.covariance_type,
                                     needed_shape, covars_shape))

        # Checking covariances' mathematical correctness
        from scipy import linalg

        if (self.covariance_type == "spherical" or
                self.covariance_type == "diag"):
            if np.any(self.covars_ <= 0):
                raise ValueError("{!r} mixture covars must be non-negative"
                                 .format(self.covariance_type))
        elif self.covariance_type == "tied":
            for i, covar in enumerate(self.covars_):
                if (not np.allclose(covar, covar.T) or
                        np.any(linalg.eigvalsh(covar) <= 0)):
                    raise ValueError("'tied' mixture covars must be "
                                     "symmetric, positive-definite")
        elif self.covariance_type == "full":
            for i, mix_covars in enumerate(self.covars_):
                for j, covar in enumerate(mix_covars):
                    if (not np.allclose(covar, covar.T) or
                            np.any(linalg.eigvalsh(covar) <= 0)):
                        raise ValueError(
                            "'full' covariance matrix of mixture {} of "
                            "component {} must be symmetric, positive-definite"
                            .format(j, i))

    def _generate_sample_from_state(self, state, random_state=None):
        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)

        cur_weights = self.weights_[state]
        i_gauss = random_state.choice(self.n_mix, p=cur_weights)
        if self.covariance_type == 'tied':
            # self.covars_.shape == (n_components, n_features, n_features)
            # shouldn't that be (n_mix, ...)?
            covs = self.covars_
        else:
            covs = self.covars_[:, i_gauss]
            covs = fill_covars(covs, self.covariance_type,
                               self.n_components, self.n_features)
        return random_state.multivariate_normal(
            self.means_[state, i_gauss], covs[state]
        )

    def _compute_log_weighted_gaussian_densities(self, X, i_comp):
        cur_means = self.means_[i_comp]
        cur_covs = self.covars_[i_comp]
        if self.covariance_type == 'spherical':
            cur_covs = cur_covs[:, np.newaxis]
        log_cur_weights = np.log(self.weights_[i_comp])

        return log_multivariate_normal_density(
            X, cur_means, cur_covs, self.covariance_type
        ) + log_cur_weights

    def _compute_log_likelihood(self, X):
        n_samples, _ = X.shape
        res = np.zeros((n_samples, self.n_components))

        for i in range(self.n_components):
            log_denses = self._compute_log_weighted_gaussian_densities(X, i)
            with np.errstate(under="ignore"):
                res[:, i] = logsumexp(log_denses, axis=1)

        return res

    def _initialize_sufficient_statistics(self):
        stats = super(GMMHMM, self)._initialize_sufficient_statistics()
        stats['n_samples'] = 0
        stats['post_comp_mix'] = None
        stats['post_mix_sum'] = np.zeros((self.n_components, self.n_mix))
        stats['post_sum'] = np.zeros(self.n_components)
        stats['samples'] = None
        stats['centered'] = None
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          post_comp, fwdlattice, bwdlattice):

        # TODO: support multiple frames

        super(GMMHMM, self)._accumulate_sufficient_statistics(
            stats, X, framelogprob, post_comp, fwdlattice, bwdlattice
        )

        n_samples, _ = X.shape

        stats['n_samples'] = n_samples
        stats['samples'] = X

        prob_mix = np.zeros((n_samples, self.n_components, self.n_mix))
        for p in range(self.n_components):
            log_denses = self._compute_log_weighted_gaussian_densities(X, p)
            with np.errstate(under="ignore"):
                prob_mix[:, p, :] = np.exp(log_denses) + np.finfo(np.float).eps

        prob_mix_sum = np.sum(prob_mix, axis=2)
        post_mix = prob_mix / prob_mix_sum[:, :, np.newaxis]
        post_comp_mix = post_comp[:, :, np.newaxis] * post_mix
        stats['post_comp_mix'] = post_comp_mix

        stats['post_mix_sum'] = np.sum(post_comp_mix, axis=0)
        stats['post_sum'] = np.sum(post_comp, axis=0)

        stats['centered'] = X[:, np.newaxis, np.newaxis, :] - self.means_

    def _do_mstep(self, stats):
        super(GMMHMM, self)._do_mstep(stats)

        n_samples = stats['n_samples']
        n_features = self.n_features

        # Maximizing weights
        alphas_minus_one = self.weights_prior - 1
        new_weights_numer = stats['post_mix_sum'] + alphas_minus_one
        new_weights_denom = (
            stats['post_sum'] + np.sum(alphas_minus_one, axis=1)
        )[:, np.newaxis]
        new_weights = new_weights_numer / new_weights_denom

        # Maximizing means
        lambdas, mus = self.means_weight, self.means_prior
        new_means_numer = np.einsum(
            'ijk,il->jkl',
            stats['post_comp_mix'], stats['samples']
        ) + lambdas[:, :, np.newaxis] * mus
        new_means_denom = (stats['post_mix_sum'] + lambdas)[:, :, np.newaxis]
        new_means = new_means_numer / new_means_denom

        # Maximizing covariances
        centered_means = self.means_ - mus

        if self.covariance_type == 'full':
            centered = stats['centered'].reshape((
                n_samples, self.n_components, self.n_mix, self.n_features, 1
            ))
            centered_t = stats['centered'].reshape((
                n_samples, self.n_components, self.n_mix, 1, self.n_features
            ))
            centered_dots = centered * centered_t

            psis_t = np.transpose(self.covars_prior, axes=(0, 1, 3, 2))
            nus = self.covars_weight

            centr_means_resh = centered_means.reshape((
                self.n_components, self.n_mix, self.n_features, 1
            ))
            centr_means_resh_t = centered_means.reshape((
                self.n_components, self.n_mix, 1, self.n_features
            ))
            centered_means_dots = centr_means_resh * centr_means_resh_t

            new_cov_numer = np.einsum(
                'ijk,ijklm->jklm',
                stats['post_comp_mix'], centered_dots
            ) + psis_t + (lambdas[:, :, np.newaxis, np.newaxis] *
                          centered_means_dots)
            new_cov_denom = (
                stats['post_mix_sum'] + 1 + nus + self.n_features + 1
            )[:, :, np.newaxis, np.newaxis]

            new_cov = new_cov_numer / new_cov_denom
        elif self.covariance_type == 'diag':
            centered2 = stats['centered'] ** 2
            centered_means2 = centered_means ** 2

            alphas = self.covars_prior
            betas = self.covars_weight

            new_cov_numer = np.einsum(
                'ijk,ijkl->jkl',
                stats['post_comp_mix'], centered2
            ) + lambdas[:, :, np.newaxis] * centered_means2 + 2 * betas
            new_cov_denom = (
                stats['post_mix_sum'][:, :, np.newaxis] + 1 + 2 * (alphas + 1)
            )

            new_cov = new_cov_numer / new_cov_denom
        elif self.covariance_type == 'spherical':
            centered_norm2 = np.sum(stats['centered'] ** 2, axis=-1)

            alphas = self.covars_prior
            betas = self.covars_weight

            centered_means_norm2 = np.sum(centered_means ** 2, axis=-1)

            new_cov_numer = np.einsum(
                'ijk,ijk->jk',
                stats['post_comp_mix'], centered_norm2
            ) + lambdas * centered_means_norm2 + 2 * betas
            new_cov_denom = (
                n_features * stats['post_mix_sum'] + n_features +
                2 * (alphas + 1)
            )

            new_cov = new_cov_numer / new_cov_denom
        elif self.covariance_type == 'tied':
            centered = stats['centered'].reshape((
                n_samples, self.n_components, self.n_mix, self.n_features, 1
            ))
            centered_t = stats['centered'].reshape((
                n_samples, self.n_components, self.n_mix, 1, self.n_features
            ))
            centered_dots = centered * centered_t

            psis_t = np.transpose(self.covars_prior, axes=(0, 2, 1))
            nus = self.covars_weight

            centr_means_resh = centered_means.reshape((
                self.n_components, self.n_mix, self.n_features, 1
            ))
            centr_means_resh_t = centered_means.reshape((
                self.n_components, self.n_mix, 1, self.n_features
            ))
            centered_means_dots = centr_means_resh * centr_means_resh_t

            lambdas_cmdots_prod_sum = np.einsum(
                'ij,ijkl->ikl',
                lambdas, centered_means_dots
            )

            new_cov_numer = np.einsum(
                'ijk,ijklm->jlm',
                stats['post_comp_mix'], centered_dots
            ) + lambdas_cmdots_prod_sum + psis_t
            new_cov_denom = (
                stats['post_sum'] + self.n_mix + nus + self.n_features + 1
            )[:, np.newaxis, np.newaxis]

            new_cov = new_cov_numer / new_cov_denom

        # Assigning new values to class members
        self.weights_ = new_weights
        self.means_ = new_means
        self.covars_ = new_cov

class NegativeBinomialHMM(_BaseHMM):

    """
    Two possible notations:

    #####
    (I)

    p - probability of success
    r - number of failures
    X ~ NB(r, p) - number of successes before r failures occures

    mean(X) = rp / (1-p)
    var(X) = rp / (1-p)**2

    p = (var - mean) / var
    r = mean**2 / (var - mean)

    That's notation from wikipedia.

    #####
    (II)

    p - probability of success
    r - number of successes
    X ~ NB(r, p) - number of failures before r successes occures

    mean(X) = r(1-p) / p
    var(X) = r(1-p) / p**2

    p = mean / var
    r = mean**2 / (var - mean)

    That's notation used in scipy.stats.nbinom and R.

    #####

    These notations can be transformed easily into each other by setting p := 1-p.

    Current formulas assume notation (II).

    If you plan on changing that,
    find all methods with #NOTATION tag.
    """

    def __init__(self, n_components,
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters,
                 debug_prefix=None):
        _BaseHMM.__init__(self,
                          n_components=n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm,
                          random_state=random_state,
                          n_iter=n_iter, tol=tol,
                          verbose=verbose,
                          params=params, init_params=init_params,
                          debug_prefix=debug_prefix)
        self._update_r_ = True
        self._update_p_ = False

    def _init(self, X, lengths=None):
        super(NegativeBinomialHMM, self)._init(X, lengths)
        _, n_features = X.shape
        if hasattr(self, 'n_features') and self.n_features != n_features:
            raise ValueError('Unexpected number of dimensions, got %s but '
                             'expected %s' % (n_features, self.n_features))
        self.n_features = n_features
        if any([letter in self.init_params for letter in ['m', 'c', 'p', 'r']]):
            # estimate means, covars; calculate p, r
            means = self._estimate_means(X)
            covars = self._estimate_covars(X)
        if 'm' in self.init_params or not hasattr(self, "means_"):
            self.means_ = means
        if 'c' in self.init_params or not hasattr(self, "covars_"):
            self.covars_ = covars
        p, r = self._calculate_p_r(self.means_, self.covars_)
        if 'p' in self.init_params or not hasattr(self, "p_"):
            self.p_ = p
        if 'r' in self.init_params or not hasattr(self, "r_"):
            self.r_ = r
        if self.debug_prefix is not None:
             with open("%smeans_%d" %
                       (self.debug_prefix, self.iteration), 'w') as output:
                output.write(array2str(self.means_))
             with open("%scovars_%d" %
                       (self.debug_prefix, self.iteration), 'w') as output:
                output.write(array2str(self.covars_))
             with open("%sr_%d" %
                       (self.debug_prefix, self.iteration), 'w') as output:
                output.write(array2str(self.r_))
             with open("%sp_%d" %
                       (self.debug_prefix, self.iteration), 'w') as output:
                output.write(array2str(self.p_))

    def _estimate_means(self, X):
        """
        Estimate means with k-means.
        Based on GaussianHMM.

        I'm not sure if it's the best way,
         maybe simple quantiles would be better?
         But it's ready, so I'll go with it for now.

        Ok, it's actually not used. I leave it just in case.
        """
        kmeans = cluster.KMeans(n_clusters=self.n_components,
                                random_state=self.random_state)
        kmeans.fit(X)
        means = kmeans.cluster_centers_
        means = np.sort(means, axis=0)
        return means

    def _estimate_covars(self, X):
        """
        Estimate covars.
        Based on GaussianHMM.
        I'm not sure what covariance type is appropriate here,
         so I went with diag.

        min_covar is the default value for GaussianHMM self.min_covar.
        """
        min_covar = 1e-3
        cv = np.cov(X.T) + min_covar * np.eye(X.shape[1])
        if not cv.shape:
            cv.shape = (1, 1)
        covars = \
            _utils.distribute_covar_matrix_to_match_covariance_type(
                cv, 'diag', self.n_components).copy()
        return covars

        """
        From GaussianHMM:
        if 'm' in self.init_params or not hasattr(self, "means_"):
            kmeans = cluster.KMeans(n_clusters=self.n_components,
                                    random_state=self.random_state)
            kmeans.fit(X)
            means = kmeans.cluster_centers_
            means = np.sort(means, axis = 0)
            self.means_ = means
        logging.debug("Initial means:")
        logging.debug(self.means_)
        if 'c' in self.init_params or not hasattr(self, "covars_"):
            cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
            if not cv.shape:
                cv.shape = (1, 1)
            self._covars_ = \
                _utils.distribute_covar_matrix_to_match_covariance_type(
                    cv, self.covariance_type, self.n_components).copy()
        """

    def _calculate_p_r(self, means=None, covars=None):
        """
        Calculate p and r parameters from means and covars estimations.

        If covars < means, calculated p and r make no sense.
        We need to correct it, so that 0 < p < 1 and r > 0.
        If indeed covars < means for every state MLE cannot be obtained,
        but it might be that it happened just in the initialisation step,
        and after an iteration it won't happen.
        So no need to raise alarm here, I guess.

        #NOTATION
        To see current notation go to class description.
        """
        if means is None:
            means = self.means_
        if covars is None:
            covars = self.covars_
        #p = (covars - means) / covars
        p = means / covars
        r = means ** 2 / (covars - means)
        if np.any(p > 1):
            p[p > 1] = 0.99
        if np.any(p <= 0):
            p[p <= 0] = 0.000001
        if np.any(r <= 0):
            r[r <= 0] = 0.001
        return p, r

    def _calculate_means_covars(self, p=None, r=None):
        """
        Calculate means and covars from p and r EM estimations.

        #NOTATION
        To see current notation go to class description.
        """
        if p is None:
            p = self.p_
        if r is None:
            r = self.r_
        #means = r * p / (1 - p)
        #covars = r * p / (1 - p)**2
        means = r * (1-p) / p
        covars = r * (1-p) / p**2
        return means, covars

    def _check(self):
        super(NegativeBinomialHMM, self)._check()
        # maybe we could check here whether variance > mean?
        # though I'm not sure if it can be checked a priori

    def _compute_log_likelihood(self, X):
        """
        #NOTATION
        To see current notation go to class description.
        """
        def _logpmf(X, r, p):
            # jesli musze zmieniac tu typ na 64, to czy w ogole jest jakis zysk z uzywania 128?
            return nbinom.logpmf(X.astype('int'), r.astype('float64'),
                                 p.astype('float64')).astype('float128')
        n_observations, n_features = X.shape
        log_likelihood = np.ndarray((n_observations, self.n_components))
        for observation in range(n_observations):
            for state in range(self.n_components):
                log_likelihood[observation, state] = \
                    np.sum(_logpmf(X[observation, :],
                                   self.r_[state, :], self.p_[state, :]))
        return log_likelihood

    def _generate_sample_from_state(self, state, random_state=None):
        """
        #NOTATION
        To see current notation go to class description.
        """
        random_state = check_random_state(random_state)
        return [nbinom(self.r_[state][feature],
                       self.p_[state][feature]).rvs()
                for feature in range(self.n_features)]

    def _initialize_sufficient_statistics(self):
        stats = super(NegativeBinomialHMM, self)._initialize_sufficient_statistics()
        # observations
        stats['X'] = np.ndarray((0, self.n_features))
        # sum_t posteriors * observations
        # I leave name 'obs' for consistency with Gaussian
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        # sum_t posteriors
        # I leave name 'post' for consistency with Gaussian
        stats['post'] = np.zeros(self.n_components)
        # posteriors
        stats['posteriors'] = np.ndarray((0, self.n_components))

        #stats['obs**2'] = np.zeros((self.n_components, self.n_features))
        #if self.covariance_type in ('tied', 'full'):
        #    stats['obs*obs.T'] = np.zeros((self.n_components, self.n_features,
        #                                   self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super(NegativeBinomialHMM, self)._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice)
        stats['X'] = np.append(stats['X'], X, axis=0)
        stats['post'] += posteriors.sum(axis=0)
        stats['obs'] += np.dot(posteriors.T, X)
        stats['posteriors'] = np.append(stats['posteriors'], posteriors, axis=0)
        """
        if 'c' in self.params:
            if self.covariance_type in ('spherical', 'diag'):
                stats['obs**2'] += np.dot(posteriors.T, obs ** 2)
            elif self.covariance_type in ('tied', 'full'):
                # posteriors: (nt, nc); obs: (nt, nf); obs: (nt, nf)
                # -> (nc, nf, nf)
                stats['obs*obs.T'] += np.einsum(
                    'ij,ik,il->jkl', posteriors, obs, obs)
        """

    def _do_mstep(self, stats):
        logging.debug("Start of m step; p, r, means, covars:")
        logging.debug(self.p_)
        logging.debug(self.r_)
        logging.debug(self.means_)
        logging.debug(self.covars_)
        super(NegativeBinomialHMM, self)._do_mstep(stats)
        # update:
        #self.p_, self.r_, self.means_, self.covars_
        if np.any((stats['post']) == 0):
            raise RuntimeError("stats['post'] has zeros."
                               " It means for at least one state the following is true:"
                               " for every window, there is zero posterior probability"
                               " of this window being in this state."
                               " It might be needed to lower number of states."
                               " Maybe different initial means will help."
                               " No reason to continue now,"
                               " from here only errors await."
                               " Here are current values of some parameters,"
                               " for debugging / replicating purposes:\n"
                               " means:\n%s\n covars:\n%s\n p:\n%s\n r:\n%s\n"
                               " stats['obs'] (posteriors.T * X):\n%s\n"
                               " stats['post'] (sum posteriors):\n%s\n"
                               " Aaand we finish here. Bye."
                               % (str(self.means_),
                                  str(self.covars_),
                                  str(self.p_),
                                  str(self.r_),
                                  str(stats['obs']),
                                  str(stats['post'])))
        if np.any((stats['obs']) == 0):
            logging.warning("stats['obs'] has zeros."
                            "Probably sth will go wrong now."
                            " Try different initial means"
                            " or lower number of states.")
        if self._update_p_:
            self.p_ = self._update_p(stats)
            self._update_p_ = False
            self._update_r_ = True
        elif self._update_r_:
            self.r_ = self._update_r(stats)
            self._update_p_ = True
            self._update_r_ = False
        self.means_, self.covars_ = self._calculate_means_covars()
        logging.debug("End of m step; p, r, means, covars:")
        logging.debug(self.p_)
        logging.debug(self.r_)
        logging.debug(self.means_)
        logging.debug(self.covars_)
        if self.debug_prefix is not None:
             with open("%smeans_%d" %
                       (self.debug_prefix, self.iteration), 'w') as output:
                output.write(array2str(self.means_))
             with open("%scovars_%d" %
                       (self.debug_prefix, self.iteration), 'w') as output:
                output.write(array2str(self.covars_))
             with open("%sr_%d" %
                       (self.debug_prefix, self.iteration), 'w') as output:
                output.write(array2str(self.r_))
             with open("%sp_%d" %
                       (self.debug_prefix, self.iteration), 'w') as output:
                output.write(array2str(self.p_))


    def _update_p(self, stats):
        """
        ML estimation of p parameter.

        #NOTATION
        To see current notation go to class description.

        p_j = sum_t (posteriori_j,t * r_j) / sum_t posteriori_j_t(x_t + r_j)
        """
        post_times_r = stats['post'][:, np.newaxis] * self.r_
        #print(stats['post'].shape)
        #print(self.r_.shape)
        #print(stats['obs'].shape)
        #print((stats['post']*self.r_).shape)
        #print(post_times_r.shape)
        p_mle = post_times_r / (stats['obs'] + post_times_r)
        if np.any(np.isnan(p_mle)):
            print("Warning: p MLE is nan")
            if np.any(np.isnan(post_times_r)):
                print("...specifically post_times_r")
            if np.any(np.isnan(stats['obs'])):
                print("...specifically stats['obs']")
            if np.any((stats['obs'] + post_times_r) == 0):
                print("stats obs + post_times_r ma zera:")
                print(stats['obs'] + post_times_r)
                print("stats obs, post times r osobno:")
                print(stats['obs'])
                print(post_times_r)
                print("stats post:")
                print(stats['post'])

        if np.any(p_mle > 1):
            print("Warning: your p MLE is bigger than 1")
            p_mle[p_mle > 1] = 0.99
        if np.any(p_mle < 0):
            print("Warning: your p MLE is smaller than 0")
            p_mle[p_mle < 0] = 0.01
        #X = stats['X']
        #n_samples = X.shape[0]
        #X_sum = np.sum(X, axis=0)
        #p_mle = X_sum / (n_samples * self.r_ + X_sum)
        #p_mle = n_samples * self.r_ / (n_samples * self.r_ + X_sum)
        logging.debug("p MLE:")
        logging.debug(p_mle)
        if p_mle.shape != self.p_.shape:
            raise ValueError('p MLE has different shape than p in previous iteration.'
                             ' Expected %s, got %s.'
                             ' Check your MLE calculations.'
                             % (str(self.p_.shape), str(p_mle.shape)))
        return p_mle

    def _update_r(self, stats):
        """
        ML estimation of r parameter.

        #NOTATION
        To see current notation go to class description.
        """
        r_mle = self.r_
        r_mle = hmmlearn.finding_r.find_r(self.r_, stats['X'], stats['posteriors'], self.p_)
        if np.any(r_mle < 0):
            print("Warning: your r MLE is smaller than 0")
            r_mle[r_mle < 0] = 0.5
        logging.debug("r MLE:")
        logging.debug(r_mle)
        if r_mle.shape != self.r_.shape:
            raise ValueError('r MLE has different shape than r in previous iteration.'
                             ' Expected %s, got %s.'
                             ' Check your MLE calculations.'
                             % (str(self.r_.shape), str(r_mle.shape)))
        return r_mle
