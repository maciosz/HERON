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
        self.probability = None

    def create_HMM(self):
        if self.distribution == "Gauss":
            return hmm.GaussianHMM(self.number_of_states,
                                   covariance_type='diag',
                                   n_iter=1000, tol=0.000005,
                                   verbose=True)
        elif self.distribution == "NB":
            return hmm.NegativeBinomialHMM(self.number_of_states,
                                           n_iter=1000,
                                           tol=0.00005,
                                           verbose=True)

    def initialise_transition_matrix(self, n_peaks):
        self.model.init_params = self.model.init_params.replace("t", "")
        #how_many_peak_states = self.number_of_states - 1
        #background = [1 - n_peaks * how_many_peak_states]
        #             + [n_peaks] * how_many_peak_states
        #peak = [1 - 2 * n_peaks * how_many_peak_states] +
        #       + [2 * n_peaks] * how_many_peak_states
        #transmat = np.array(background,
        #                    peak * how_many_peak_states)
        how_many_background_states = self.number_of_states - 1
        background = [(1 - n_peaks) / how_many_background_states] \
                     * how_many_background_states \
                     + [n_peaks]
        peak = [(1 - 5 * n_peaks) / how_many_background_states] \
                     * how_many_background_states \
                     + [5 * n_peaks]
        print background
        print peak
        transmat = numpy.array([background] * how_many_background_states + \
                            [peak])
        self.model.transmat_ = transmat
        print transmat

    def read_in_files(self, files):
        self.data.add_data_from_bedgraphs(files)

    def filter_data(self, threshold):
        self.data.filter_data(threshold)

    def fit_HMM(self):
        self.model.fit(self.data.matrix, lengths=self.data.numbers_of_windows)

    def predict_states(self):
        logging.info("predicting states, stay tuned")
        logging.info("prepairing data")
        self.prepair_data()
        logging.info("fitting model")
        print self.model
        print self.model.transmat_
        self.fit_HMM()
        logging.info("predicting states")
        self.probability, states = self.model.decode(self.data.matrix,
                                                     lengths=self.data.numbers_of_windows)
        logging.info("Is convergent: %s", str(self.model.monitor_.converged))
        self.data.matrix = numpy.c_[self.data.matrix, states]
        #return states

    def prepair_data(self):
        if self.distribution == "NB":
            self.data.convert_floats_to_ints()
        self.data.matrix = numpy.array(self.data.matrix).transpose()

    def save_states_to_seperate_files(self, output_prefix):
        """
        Assumes states are in the last row of the data matrix.
        """
        intervals = self.data.windows_to_intervals(-1)
        print self.data.matrix.transpose() #[-1]
        #print intervals
        output = output_prefix + "_all_states.bed"
        self.data.save_intervals_as_bed(output, intervals)
        for state in xrange(self.number_of_states):
            output_name = output_prefix + "_state_" + str(state) + ".bed"
            self.data.save_intervals_as_bed(output_name, intervals, state)

    def write_stats_to_file(self, output_prefix):
        output = open(output_prefix + "_stats.txt", "w")
        output.write("Score: "
                     + str(self.model.score(numpy.delete(self.data.matrix, -1, axis = 1),
                                            self.data.numbers_of_windows))
                     + '\n')
        # zapisywanie stanow do data.matrix troche zepsulo ten kawalek,
        # musze usuwac ostatnia kolumne tutaj ^
        output.write("Probability: " + str(self.probability) + '\n')
        output.write("Transition matrix: \n" + str(self.model.transmat_) + '\n')
        output.write("Means: \n" + str(self.model.means_) + '\n')
        output.write("Covars: \n" + str(self.model.covars_) + '\n')
        output.write("Mean length: TODO\n")
        output.close()
