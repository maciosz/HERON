#!/usr/bin/python

import logging
import warnings
import random
import numpy
from hmmlearn import hmm
from data import Data

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

class Model(object):

    def __init__(self, number_of_states, distribution, random_seed):
        if not random_seed:
            self.random_seed = random.randint(0, 2**32 - 1)
        else:
            self.random_seed = random_seed
        self.data = Data()
        self.data_for_training = Data()
        self.window_size = 0
        self.number_of_states = number_of_states
        self.distribution = distribution
        self.model = self.create_HMM()
        #self.model.means_ = numpy.array([[0], [4], [20]])
        self.probability = None

    def create_HMM(self):
        random_state = numpy.random.RandomState(self.random_seed)
        if self.distribution == "Gauss":
            return hmm.GaussianHMM(self.number_of_states,
                                   covariance_type='diag',
                                   n_iter=1000, tol=0.000005,
                                   random_state = random_state,
                                   #means_weight = 0.00001,
                                   #init_params = 'cts',
                                   verbose=True)
        elif self.distribution == "NB":
            return hmm.NegativeBinomialHMM(self.number_of_states,
                                           n_iter=1000,
                                           tol=0.000005,
                                           random_state = random_state,
                                           verbose=True)

    def initialise_means(self, means, n_samples):
        self.model.init_params = self.model.init_params.replace("m", "")
        #means = np.array([[0.], [1.], [2.]])
        if len(means) == self.number_of_states:
            means = numpy.repeat(means, n_samples)
        elif len(means) != self.number_of_states * n_samples:
            raise ValueError("Inproper length of initialised means;"
                             " should be either n_states or n_states * n_samples,"
                             " in this casa either %d or %d * %d."
                             " Got %d" % (self.number_of_states,
                                          self.number_of_states,
                                          n_samples,
                                          len(means)))
        means = numpy.array(means).astype('float128')
        means = means.reshape((self.number_of_states, n_samples))
        # TODO: jakies sprawdzanie czy to n_samples sie zgadza
        self.model.means_ = means
        

    def initialise_transition_matrix(self, n_peaks):
        """
        To trzeba recznie zmieniac zeby dostosowac do aktualnych potrzeb.
        W fazie testowania.
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

    def read_in_files(self, files, resolution):
        """
        Read in files given as a list of strings.
        The data structure actually does all the work.
        Resolution is needed only for reading bams;
        it's ignored when all the data are bedgraphs.
        """
        if self.distribution == "NB":
            mean = False
        elif self.distribution == "Gauss":
            mean = True
        self.data.add_data_from_files(files, resolution, mean)

    def filter_data(self, threshold):
        """
        Filter the data above threshold.
        """
        self.data.filter_data(threshold)

    def filter_training_data(self, threshold):
        threshold_values = self.data_for_training.find_threshold_value(threshold)
        self.data_for_training.split_data(threshold_values)

    def fit_HMM(self):
        """
        Fit the HMM using Baum-Welch algorithm.
        That is - estimate the parameters of HMM
        basing on the data, using EM approach.
        """
        logging.debug("Data for training shape: %s", str(self.data_for_training.matrix.shape))
        # to sie dzieje i tu i w peakcallerze, zdecyduj sie
        self.prepair_data()
        self.model.fit(self.data_for_training.matrix,
                       lengths=self.data_for_training.numbers_of_windows)

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
        # TODO: metoda add_column dla Data
        self.data.matrix = numpy.c_[self.data.matrix, states]
        logging.info("Number of iterations till convergence: %i", self.model.monitor_.iter)
        if self.distribution == "NB":
            if self.model.covars_le_means > 0:
                logging.warning("Covars <= means %i times during fitting. No good.",
                                self.model.covars_le_means)
        #return states

    def prepair_data(self):
        """
        Changes data matrix to numpy array and transposes it.
        For NB distribution converts floats to ints.

        TODO: jakos to inaczej zrobic, no co to za kopiowanie kodu.
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
        self.data.save_intervals_as_bed(output, intervals, save_value=True)
        for state in xrange(self.number_of_states):
            output_name = output_prefix + "_state_" + str(state) + ".bed"
            self.data.save_intervals_as_bed(output_name, intervals, state)

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
