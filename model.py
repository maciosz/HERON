#!/usr/bin/python

import logging
import warnings
from data import Data

class Model:
    
    def __init__(self, number_of_states, distribution):
        self.data = Data()
        self.window_size = 0 
        self.number_of_states = number_of_states
        self.distribution = distribution
        self.model = self.create_HMM()

        def create_HMM(self):
            if self.distribution == "Gauss":
                self.model = hmm.GaussianHMM(self.number_of_states,
                                             covariance_type='diag',
                                             n_iter=1000, tol=0.000005,
                                             verbose=True)
            elif self.distribution == "NB":
                self.model = hmm.NegativeBinomialHMM(self.number_of_states,
                                                     n_iter=1000,
                                                     tol=0.00005,
                                                     verbose=True)

    def read_in_files(self, files):
        self.data.read_in_files(files)

    def fit_HMM(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        self.model.fit(self.data.matrix, lengths=data.number_of_windows)

    def predict_states(self):
        logging.info("predicting states, stay tuned")
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        logging.info("prepairing data")
        self.prepair_data(self.data)
        logging.info("fitting model")
        self.fit_HMM()
        logging.info("predicting states")
        self.probability, states = self.model.decode(self.matrix, lengths=self.chromosome_lengths)
        logging.info("Is convergent: %s", str(self.model.monitor_.converged))
        return states

    def prepair_data(self):
        if self.distribution == "NB":
            self.data.covert_floats_to_ints()
        self.data.matrix = numpy.array(self.data.matrix).transpose()
