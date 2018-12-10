#!/usr/bin/python

import logging
import warnings
import numpy
from hmmlearn import hmm
from data import Data

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

class Model(object):

    def __init__(self, number_of_states, distribution):
        self.data = Data()
        self.window_size = 0
        self.number_of_states = number_of_states
        self.distribution = distribution
        self.model = self.create_HMM()
        #self.model.means_ = numpy.array([[0], [4], [20]])
        self.probability = None

    def create_HMM(self):
        if self.distribution == "Gauss":
            return hmm.GaussianHMM(self.number_of_states,
                                   covariance_type='diag',
                                   n_iter=1000, tol=0.000005,
                                   #means_weight = 0.00001,
                                   #init_params = 'cts',
                                   verbose=True)
        elif self.distribution == "NB":
            return hmm.NegativeBinomialHMM(self.number_of_states,
                                           n_iter=1000,
                                           tol=0.000005,
                                           verbose=True)

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
        self.data.add_data_from_files(files, resolution)

    def filter_data(self, threshold):
        self.data.filter_data(threshold)

    def fit_HMM(self):
        self.model.fit(self.data.matrix, lengths=self.data.numbers_of_windows)

    def predict_states(self):
        logging.info("predicting states, stay tuned")
        logging.info("prepairing data")
        self.prepair_data()
        logging.info("fitting model")
        #logging.debug(self.model)
        #if hasattr(self.model, "transmat_"):
        #    logging.debug(self.model.transmat_)
        self.fit_HMM()
        logging.info("predicting states")
        self.probability, states = self.model.decode(self.data.matrix,
                                                     lengths=self.data.numbers_of_windows)
        logging.info("Is convergent: %s", str(self.model.monitor_.converged))
        self.data.matrix = numpy.c_[self.data.matrix, states]
        logging.info("Number of iterations till convergence: %i", self.model.monitor_.iter)
        if self.distribution == "NB":
            if self.model.covars_le_means > 0:
                logging.warning("Covars <= means %i times during fitting. No good.",
                                self.model.covars_le_means)
        #return states

    def prepair_data(self):
        if self.distribution == "NB":
            self.data.convert_floats_to_ints()
        self.data.matrix = numpy.array(self.data.matrix).transpose()
        logging.debug("Wymiary macierzy: %s", str(self.data.matrix.shape))

    def save_states_to_seperate_files(self, output_prefix):
        """
        Assumes states are in the last row of the data matrix.
        """
        intervals = self.data.windows_to_intervals(-1)
        #print self.data.matrix.transpose() #[-1]
        #print intervals
        output = output_prefix + "_all_states.bed"
        self.data.save_intervals_as_bed(output, intervals, save_value=True)
        for state in xrange(self.number_of_states):
            output_name = output_prefix + "_state_" + str(state) + ".bed"
            self.data.save_intervals_as_bed(output_name, intervals, state)

    def write_stats_to_file(self, output_prefix):
        output = open(output_prefix + "_stats.txt", "w")
        output.write("Score:\t"
                     + str(self.model.score(numpy.delete(self.data.matrix, -1, axis=1),
                                            self.data.numbers_of_windows))
                     + '\n')
        # zapisywanie stanow do data.matrix troche zepsulo ten kawalek,
        # musze usuwac ostatnia kolumne tutaj ^
        #output.write("Probability: " + str(self.probability) + '\n')
        self.write_probability_to_file(output)
        #output.write("Transition matrix: \n" + str(self.model.transmat_) + '\n')
        self.write_transmat_to_file(output)
        #output.write("Means: \n" + str(self.model.means_) + '\n')
        self.write_means_to_file(output)
        #output.write("Covars: \n" + str(self.model.covars_) + '\n')
        self.write_covars_to_file(output)
        output.write("Mean length: TODO\n")
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
    # Actually, diag covars would be more readable if
    # only diagonal would be printed, without all the zeros.

    def write_means_to_file(self, output_file):
        self._write_some_stat_to_file(output_file,
                                      "means")

    def write_covars_to_file(self, output_file):
        self._write_some_stat_to_file(output_file,
                                      "covars")

    def write_transmat_to_file(self, output_file):
        output_file.write("Transition matrix:\n")
        for line in self.model.transmat_:
            output_file.write("\t".join([str(i) for i in line]))
            output_file.write("\n")

    def write_matrix_to_file(self, output_file):
        for i in self.data.matrix:
            for j in i:
                if j == 0:
                    j = "0.0"
                output_file.write(str(j))
                output_file.write("\n")
