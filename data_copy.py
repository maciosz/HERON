#!/usr/bin/env python

import sys
import subprocess
import warnings
import logging
import numpy
from hmmlearn import hmm

class Data:
    """
    Reading, storing and writing data
    from coverages and intervals.
    """

    def __init__(self, number_of_states, distr):
        # why do I need window_size here?
        # in the arguments, specifically
        self.matrix = []
        self.window_size = 100 #window_size
        self.number_of_states = number_of_states
        if distr == "Gauss":
            self.model = hmm.GaussianHMM(number_of_states,
                                         covariance_type='diag',
                                         n_iter=1000, tol=0.000005,
                                         verbose=True)
        elif distr == "NB":
            self.model = hmm.NegativeBinomialHMM(number_of_states,
                                                 n_iter=1000,
                                                 tol=0.00005,
                                                 verbose=True)
        self.chromosome_lengths = []
        # maybe number_of_windows_in_chromosomes?
        # technically its not a length here
        self.chromosome_names = []
        self.chromosome_ends = []

    def filter_data(self, threshold):
        """
        Set data above the threshold to the median.
        That's just a sketch of what to do with outliers.
        """
        medians = []
        for line in self.matrix:
            median = numpy.median(filter(lambda x: x <= 1000, line))
            medians.append(median)
        #means = map(numpy.mean, filter(lambda x: x <= 1000, self.matrix))
        print "mediany:", medians
        counter = 0
        for which_line, line in enumerate(self.matrix):
            for position, value in enumerate(line):
                if value > threshold:
                    print "podmieniam", value, "na", median[which_line]
                    self.matrix[which_line][position] = median[which_line]
                    counter += 1
        logging.info("I've reduced values in %i windows to median value.", counter)
           


    def add_data_from_bedgraph(self, filename):
        logging.info("reading file %s", filename)
        bedgraph = open(filename)
        chromosome_lengths = []
        chromosome_names = []
        chromosome_ends = []
        self.matrix.append([])
        previous_end = 0
        chromosome, start, end, value = bedgraph.next().strip().split()
        last_chromosome = chromosome
        chromosome_names.append(chromosome)
        no_of_windows_in_current_chromosome = 1
        start, end = int(start), int(end)
        self.window_size = end-start
        possibly_unfixed_resolution = False
        floats = False
        self.matrix[-1].append(int(float(value)))
        for line in bedgraph:
            chromosome, start, end, value = line.strip().split()
            if chromosome == last_chromosome and possibly_unfixed_resolution:
                sys.exit("Unfixed resolution around coordinates " +
                         chromosome + ' ' + str(start) + ' ' + str(end))
            if int(end) - int(start) != self.window_size:
                possibly_unfixed_resolution = True
            if not floats and int(float(value)) != float(value):
                logging.info("Warning, your values contain floats. I'm converting them to ints")
                # TODO: i don't have to do it if distribution = Gauss, should be parametrised
                floats = True
            self.matrix[-1].append(int(float(value)))
            if chromosome != last_chromosome:
                chromosome_ends.append(previous_end)
                possibly_unfixed_resolution = False
                chromosome_lengths.append(no_of_windows_in_current_chromosome)
                chromosome_names.append(chromosome)
                no_of_windows_in_current_chromosome = 0
            no_of_windows_in_current_chromosome += 1
            last_chromosome = chromosome
            previous_end = end
        chromosome_lengths.append(no_of_windows_in_current_chromosome)
        chromosome_ends.append(end)
        if not self.chromosome_lengths:
            self.chromosome_lengths = chromosome_lengths
            self.chromosome_names = chromosome_names
            self.chromosome_ends = chromosome_ends
        elif self.chromosome_lengths != chromosome_lengths:
            sys.exit('chromosome lengths between samples don\'t match')
        elif self.chromosome_names != chromosome_names:
            sys.exit('chromosome names between samples don\'t match')
        if sum(self.chromosome_lengths) != len(self.matrix[0]):
            sys.exit("sth\'s wrong with calculating chromosome lengths:" +
                     str(sum(self.chromosome_lengths)) + ' ' + str(len(self.matrix[0])))
            # that would be a weird bug. Did it ever happen?
            # from the fact that I've written this checking I assume it did
            # maybe it would be a good idea to make a method check()
            # that would check if the data seems correct and consistent
        if self.chromosome_ends != chromosome_ends:
            sys.exit('chromosome ends between samples don\'t match')
    """
    def predict_states(self):
        logging.info("predicting states, stay tuned")
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        self.matrix = numpy.array(self.matrix).transpose()
        logging.info("fitting model")
        self.model.fit(self.matrix, lengths=self.chromosome_lengths)
        logging.info("predicting states")
        self.probability, states = self.model.decode(self.matrix, lengths=self.chromosome_lengths)
        logging.info("Is convergent: %s", str(self.model.monitor_.converged))
        return states
    """

    def save_states_to_file(self, states, prefix=''):
        #output = open(prefix + "_all_states.txt", 'w')
        #for state in states:
        #    output.write(str(state))
        #    output.write('\n')
        #output.close()
        for state_being_saved in xrange(self.number_of_states):
            counter = 0
            last_state = 'last_state'
            chromosome_index = 0
            chromosome_name = self.chromosome_names[chromosome_index]
            chromosome_length = self.chromosome_lengths[chromosome_index]
            output = open(prefix + "_state_" + str(state_being_saved) + ".bed", 'w')
            for current_state in states:
                if counter == chromosome_length:
                    if last_state == state_being_saved:
                        output.write('\t'.join([chromosome_name,
                                                str(start),
                                                str(self.chromosome_ends[chromosome_index])]))
                        output.write('\n')
                    chromosome_index += 1
                    counter = 0
                    chromosome_name = self.chromosome_names[chromosome_index]
                    chromosome_length = self.chromosome_lengths[chromosome_index]
                    last_state = 'last_state'
                if current_state == state_being_saved and last_state != state_being_saved:
                    start = self.window_size * counter
                elif current_state != state_being_saved and last_state == state_being_saved:
                    end = self.window_size * counter
                    output.write('\t'.join([chromosome_name, str(start), str(end)]))
                    output.write('\n')
                counter += 1
                last_state = current_state
            if current_state == state_being_saved:
                output.write('\t'.join([chromosome_name,
                                        str(start),
                                        str(self.chromosome_ends[chromosome_index])]))
                output.write('\n')
            output.close()

    def which_state_is_peaks(self):
        # TODO: check whether mean is the highest among all samples
        return self.model.means_.mean(axis=1).argmax()

    def save_peaks_to_file(self, prefix):
        which_state = self.which_state_is_peaks()
        infile = prefix + "_state_" + str(which_state) + ".bed"
        outfile = prefix + "_peaks.bed"
        subprocess.call(["cp", infile, outfile])

    def write_stats_to_file(self, prefix):
        output = open(prefix + "_stats.txt", "w")
        output.write("Score: " + str(self.model.score(self.matrix, self.chromosome_lengths)) + '\n')
        output.write("Probability: " + str(self.probability) + '\n')
        output.write("Transition matrix: \n" + str(self.model.transmat_) + '\n')
        output.write("Means: \n" + str(self.model.means_) + '\n')
        output.write("Covars: \n" + str(self.model.covars_) + '\n')
        output.write("Mean length: TODO\n")
        output.close()
