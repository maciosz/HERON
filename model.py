#!/usr/bin/python
import copy
import logging
import warnings
import random
import itertools
import numpy
from hmmlearn import hmm
from data import Data, save_intervals_as_bed

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

class Model(object):

    def __init__(self, number_of_states, distribution, random_seed=None):
        """
        number_of_states - int; how many states the HMM should have
        distribution - str; either "NB" for negative binomial or "Gauss"
        random_seed - int, optional
                      random_seed to be used in random operations;
                      useful for reproducing results
        """
        if not random_seed:
            self.random_seed = random.randint(0, 2**32 - 1)
        else:
            self.random_seed = random_seed
        self.data = Data()
        self.data_for_training = Data()
        self.number_of_states = number_of_states
        self.distribution = distribution
        self.model = self._create_HMM()
        self.probability = None
        self.number_of_samples = 0

    def _create_HMM(self):
        random_state = numpy.random.RandomState(self.random_seed)
        if self.distribution == "Gauss":
            return hmm.GaussianHMM(self.number_of_states,
                                   covariance_type='full',
                                   #covariance_type='diag',
                                   n_iter=1000, tol=0.01,
                                   random_state=random_state,
                                   #means_weight = 0.00001,
                                   #init_params = 'cts',
                                   verbose=True)
        elif self.distribution == "NB":
            return hmm.NegativeBinomialHMM(self.number_of_states,
                                           n_iter=1000,
                                           tol=0.01,
                                           random_state=random_state,
                                           verbose=True)

    def initialise_constant_means(self, means):
        """
        means - list of means to initialise HMM with.
        """
        #means = np.array([[0.], [1.], [2.]])
        if len(means) == self.number_of_states:
            means = numpy.repeat(means, self.number_of_samples)
        elif len(means) != self.number_of_states * self.number_of_samples:
            raise ValueError("Inproper length of initialised means;"
                             " should be either n_states or n_states * n_samples,"
                             " in this case either %d or %d * %d."
                             " Got %d" % (self.number_of_states,
                                          self.number_of_states,
                                          self.number_of_samples,
                                          len(means)))
        means = numpy.array(means).reshape((self.number_of_states, self.number_of_samples))
        self._set_means(means)

    def _set_means(self, means):
        if means.shape != (self.number_of_states, self.number_of_samples):
            raise ValueError("Inproper shape of initialised means;"
                             " should be n_states * n_samples,"
                             " in this case %d * %d."
                             " Got %s" % (self.number_of_states,
                                          self.number_of_samples,
                                          str(means.shape)))
        means = numpy.array(means).astype('float128')
        self.model.init_params = self.model.init_params.replace("m", "")
        self.model.means_ = means
        logging.debug("Means set to:")
        logging.debug(means)

    def _set_covars(self, covars):
        if covars.shape != (self.number_of_states, self.number_of_samples, self.number_of_samples):
            raise ValueError("Inproper shape of initialised covars;"
                             " should be n_states * n_samples * n_samples,"
                             " in this case %d * %d * %d."
                             " Got %s" % (self.number_of_states,
                                          self.number_of_samples,
                                          self.number_of_samples,
                                          str(covars.shape)))
        covars = numpy.array(covars).astype('float128')
        self.model.init_params = self.model.init_params.replace("c", "")
        self.model.covars_ = covars
        logging.debug("Covars set to:")
        logging.debug(covars)

    def initialise_individual_means(self, levels):
        """
        Initialise means for samples based on quantiles of values.

        levels - list of floats in [0, 1] range

        There should be as many levels in the list as states in the HMM.
        Each level corresponds to one state.
        You can't specify different levels for different samples,
        but the final value of initial mean will of course depend on the values in the sample.
        """
        #if len(levels) != (self.number_of_states - 1):
        #    raise ValueError("Number of states and quantile values are incompatible;"
        #                     " #states should be equal to #quantiles + 1,"
        #                     " but I got %d states and %d quantiles." %
        #                     (self.number_of_states, len(levels)))

        if len(levels) != (self.number_of_states):
            raise ValueError("Number of states and quantile values are incompatible;"
                             " #states should be equal to #quantiles,"
                             " but I got %d states and %d quantiles." %
                             (self.number_of_states, len(levels)))
        if any(numpy.array(levels) > 1) or any(numpy.array(levels) < 0):
            raise ValueError("Quantile levels should be between 0 and 1 (inclusive).")
        n_samples = self.number_of_samples
        means = numpy.ones((self.number_of_states, n_samples))
        template = numpy.array(range(len(levels)))
        quantiles = self.data_for_training.calculate_quantiles(levels)
        for state in range(self.number_of_states):
            for sample in range(n_samples):
                mean_class = template[state]
                #if mean_class == 0:
                #    mean = 0
                #else:
                #    mean = quantiles[mean_class - 1, sample]
                mean = quantiles[mean_class, sample]
                means[state, sample] = mean
        self._set_means(means)

    def initialise_grouped_means(self, order, levels):
        """
        Initialise means assuming samples are divided
        into some user-defined groups.

        order - a list of numbers (starting from zero)
                representing a group for each sample.
        levels - levels of quantiles to use as zero-state,
                 background and enrichment, respectively.
        """
        number_of_groups = len(set(order))
        template = self._generate_template_for_grouped_means(number_of_groups)
        n_samples = len(order)
        means = numpy.ones((self.number_of_states, n_samples))
        quantiles = self.data_for_training.calculate_quantiles(levels)
        for state in range(self.number_of_states):
            for sample in range(n_samples):
                group = order[sample]
                mean_class = template[state, group]
                if mean_class == 0:
                    mean = 0
                elif mean_class == 1:
                    mean = quantiles[0, sample]
                elif mean_class == 2:
                    mean = quantiles[1, sample]
                means[state, sample] = mean
        self._set_means(means)

    def initialise_grouped_covars(self, order):
        """
        Initialise covars assuming samples are divided
        into some user-defined groups.
        Samples between groups have zero covariance.

        order - a list of numbers (starting from zero)
                representing a group for each sample.
        """
        covariances = {}
        which = {}
        for group in set(order):
            which[group] = numpy.array(order) == group
            data = self.data.matrix[:, which[group]]
            cov = numpy.cov(data.T)
            if not cov.shape:
                cov.shape = (1, 1)
            covariances[group] = cov
        #print "covariances:"
        #print covariances
        covars = numpy.zeros((self.number_of_states,
                              self.number_of_samples,
                              self.number_of_samples))
        #print "covars:"
        #print covars
        for sample in range(self.number_of_samples):
            group = order[sample]
            #print("sample %d, group %d" % (sample, group))
            covariance = covariances[group][0, :]
            #print("covariance:")
            #print(covariance)
            covariances[group] = numpy.delete(covariances[group], 0, 0)
            #print
            covars[:, sample, which[group]] = covariance
        self._set_covars(covars)
        return covars

    def _generate_template_for_grouped_means(self, number_of_groups):
        template = numpy.array([[0] * number_of_groups])
        next_states = itertools.product(*[[1, 2]] * number_of_groups)
        for state in next_states:
            state = [list(state)]
            template = numpy.append(template, state, axis=0)
        if template.shape[0] != self.number_of_states:
            logging.warning("I'm overwriting your input number of states."
                            " For %d groups I can only deal with %d states."
                            " You wanted %d.",
                            number_of_groups, template.shape[0], self.number_of_states)
            self.number_of_states = template.shape[0]
        return template

    def initialise_transition_matrix(self, n_peaks):
        """
        To trzeba recznie zmieniac zeby dostosowac do aktualnych potrzeb.
        W fazie testowania.
        W sumie bym to wywalila, to nic nie daje.
        """
        self.model.init_params = self.model.init_params.replace("t", "")
        #how_many_peak_states = self.number_of_states - 1
        #background = [1 - n_peaks * how_many_peak_states]
        #             + [n_peaks] * how_many_peak_states
        #peak = [1 - 2 * n_peaks * how_many_peak_states] +
        #       + [2 * n_peaks] * how_many_peak_states
        #transmat = np.array(background,
        #                    peak * how_many_peak_states)
        #how_many_background_states = self.number_of_states - 1
        #background = [(1 - n_peaks) / how_many_background_states] \
        #             * how_many_background_states \
        #             + [n_peaks]
        #peak = [(1 - 5 * n_peaks) / how_many_background_states] \
        #             * how_many_background_states \
        #             + [5 * n_peaks]
        #print background
        #print peak
        #transmat = numpy.array([background] * how_many_background_states + \
        #                    [peak])
        transmat = numpy.array([[0.5, 0.5, 0, 0],
                                [0.5 - n_peaks, 0.5, n_peaks, 0],
                                [0, 0.45, 0.1, 0.45],
                                [0, 0, 0.5, 0.5]])
        self.model.transmat_ = transmat
        #logging.debug(str(transmat))

    def read_in_files(self, files, resolution=0):
        """
        Read in files given as a list of strings.
        The data object actually does all the work.
        Resolution is needed only for reading bams;
        it's ignored when all the data are bedgraphs.
        """
        if self.distribution == "NB":
            mean = False
        elif self.distribution == "Gauss":
            mean = True
        self.number_of_samples = len(files)
        logging.debug("Number of files: %d", self.number_of_samples)
        self.data.add_data_from_files(files, resolution, mean)
        self.data_for_training = copy.deepcopy(self.data)

    def filter_data(self, threshold):
        """
        Filter the data above fixed threshold.
        Currently replaces the values with median value.
        See data.Data.filter_data for details.

        It's not used anymore, right?
        """
        self.data.filter_data(threshold)

    def filter_training_data(self, threshold):
        """
        Filter the data for training,
        removing windows with highest values (outliers),
        thus deconnecting chromosomes.
        See data.Data.split_data for details.

        threshold - how many percentiles of windows we want to remove
        """
        threshold_values = self.data_for_training.find_threshold_value(threshold)
        self.data_for_training.split_data(threshold_values)

    def fit_HMM(self):
        """
        Fit the HMM using Baum-Welch algorithm.
        That is - estimate the parameters of HMM
        basing on the data, using EM approach.
        """
        logging.debug("Data for training shape: %s", str(self.data_for_training.matrix.shape))
        self.prepair_data()
        self.model.fit(self.data_for_training.matrix,
                       lengths=self.data_for_training.numbers_of_windows)
        self._reorder_states()

    def _reorder_states(self):
        order = self._get_order()
        if numpy.any(order != list(range(self.number_of_states))):
            self.model.means_ = self.model.means_[order, :]
            self.model.covars_ = self.model.covars_[order, :]
            self.model.startprob_ = self.model.startprob_[order]
            self.model.transmat_ = self.model.transmat_[order, :][:, order]
            if self.distribution == "NB":
                self.model.p_ = self.model.p_[order, :]
                self.model.r_ = self.model.r_[order, :]
            #print("reordered:")
            #print(self.model.means_)
            #print("***")

    def _get_order(self):
        means = self.model.means_
        order = means.argsort(axis=0)
        order = numpy.sum(order, axis=1).argsort().argsort()
        return order

    def predict_states(self):
        """
        Predict the states in the data.
        First needs to prepare the data
        and fit the model.
        Add predicted states to the data matrix.
        """
        #logging.info("predicting states, stay tuned")
        #logging.info("prepairing data")
        #logging.info("fitting model")
        #logging.debug(self.model)
        #if hasattr(self.model, "transmat_"):
        #    logging.debug(self.model.transmat_)
        #self.fit_HMM()
        #logging.info("predicting states")
        self.probability, states = self.model.decode(self.data.matrix,
                                                     lengths=self.data.numbers_of_windows)
        logging.info("Is convergent: %s", str(self.model.monitor_.converged))
        #self.data.matrix = numpy.c_[self.data.matrix, states]
        self.data.add_column(states)
        logging.info("Number of iterations till convergence: %i", self.model.monitor_.iter)
        #if self.distribution == "NB":
        #    if self.model.covars_le_means > 0:
        #        logging.warning("Covars <= means %i times during fitting. No good.",
        #                        self.model.covars_le_means)
        #return states

    def prepair_data(self):
        """
        Changes data matrix to numpy array and transposes it.
        For NB distribution converts floats to ints.

        TODO: jakos to inaczej zrobic, no co to za kopiowanie kodu.

        Uhm... teraz tylko zmienia floaty na inty, tak?
        """
        if self.distribution == "NB":
            self.data.convert_floats_to_ints()
            self.data_for_training.convert_floats_to_ints()
        #self.data.matrix = numpy.array(self.data.matrix).transpose()
        #self.data_for_training.matrix = numpy.array(self.data.matrix).transpose()
        logging.debug("Wymiary macierzy: %s", str(self.data.matrix.shape))

    def save_states_to_seperate_files(self, output_prefix):
        """
        Changes states to intervals and saves them to seperate files.
        Also writes one bed file with all the states.
        Assumes the states are in the last column of the data matrix.
        """
        intervals = self.data.windows_to_intervals(-1)
        output = output_prefix + "_all_states.bed"
        save_intervals_as_bed(output, intervals, save_value=True)
        for state in range(self.number_of_states):
            output_name = output_prefix + "_state_" + str(state) + ".bed"
            save_intervals_as_bed(output_name, intervals, state)

    def write_stats_to_file(self, output_prefix):
        """
        Write various statistics to a file `output_prefix`_stats.txt
        """
        output = open(output_prefix + "_stats.txt", "w")
        output.write("Score:\t"
                     + str(self.model.score(numpy.delete(self.data.matrix, -1, axis=1),
                                            self.data.numbers_of_windows))
                     + '\n')
        # zapisywanie stanow do data.matrix troche zepsulo ten kawalek,
        # musze usuwac ostatnia kolumne tutaj ^
        self.write_probability_to_file(output)
        self.write_transmat_to_file(output)
        self.write_means_to_file(output)
        self.write_covars_to_file(output)
        if self.distribution == "NB":
            self.write_p_to_file(output)
            self.write_r_to_file(output)
        output.write("Mean length: TODO\n")
        output.write("Number of regions: TODO\n")
        output.close()

    def write_probability_to_file(self, output_file):
        output_file.write("Probability:\t%f\n" % self.probability)

    def _write_some_stat_to_file(self, output_file, name):
        output_file.write("%s:\n" % name)
        for i, stats in enumerate(self.model.__getattribute__(name + "_")):
            if len(stats.shape) == 1:
                output_file.write("%s_of_state_%i:" % (name, i))
                for stat in stats:
                    output_file.write("\t%f" % stat)
                output_file.write("\n")
            elif len(stats.shape) == 2:
                for stat in stats:
                    output_file.write("%s_of_state_%i:\t" % (name, i))
                    output_file.write("\t".join(["%.6f" % value for value in stat]))
                    output_file.write("\n")
            else:
                output_file.write("%s_of_state%i:\n" % (name, i))
                output_file.write(str(stats))
                output_file.write("\n")
                # is this ever happening?
    # Actually, diag covars would be more readable if
    # only diagonal would be printed, without all the zeros.

    def write_means_to_file(self, output_file):
        self._write_some_stat_to_file(output_file,
                                      "means")

    def write_covars_to_file(self, output_file):
        self._write_some_stat_to_file(output_file,
                                      "covars")

    def write_p_to_file(self, output_file):
        self._write_some_stat_to_file(output_file,
                                      "p")

    def write_r_to_file(self, output_file):
        self._write_some_stat_to_file(output_file,
                                      "r")

    def write_transmat_to_file(self, output_file):
        output_file.write("Transition matrix:\n")
        for line in self.model.transmat_:
            output_file.write("\t".join([str(i) for i in line]))
            output_file.write("\n")

    def write_matrix_to_file(self, output_file):
        """
        Write the whole data matrix to file.
        Created mainly for debugging purposes.
        """
        for i in self.data.matrix:
            for j in i:
                if j == 0:
                    j = "0.0" #? why?
                output_file.write(str(j))
                output_file.write("\t")
            output_file.write("\n")
