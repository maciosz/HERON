#!/usr/bin/env python

# -W ignore::DeprecationWarning

import sys
import numpy
import warnings
from hmmlearn import hmm

class Data:

    def __init__(self, window_size=100, number_of_states=3):
        self.matrix = [] #numpy.array()
        self.window_size = window_size
        self.number_of_states = number_of_states
        self.model = hmm.GaussianHMM(number_of_states,
                                    covariance_type='spherical',
                                    n_iter=1000, tol=0.000005,
                                    verbose=True)
        self.chromosome_lengths = []
        # maybe number_of_windows_in_chromosomes?
        # technically its not a length here
        self.chromosome_names = []

    def add_data_from_bedgraph(self, filename):
        bedgraph = open(filename)
        chromosome_lengths = []
        self.matrix.append([])
        chromosome, start, end, value = bedgraph.next().strip().split()
        last_chromosome = chromosome
        self.chromosome_names.append(chromosome)
        no_of_windows_in_current_chromosome = 1
        start, end = int(start), int(end)
        self.windows_size = end-start
        if start != 0:
            tmp = 0
            while tmp < start:
                self.matrix[-1].append(0)
                tmp += self.window_size
            # to nie zalatwia przesuniecia okien
            # w sensie jak bedgraph zaczyna sie od 50
            # to tu bedzie to interpretowane jako zaczecie od 100
        self.matrix[-1].append(int(value))
        for line in bedgraph:
            chromosome, start, end, value = line.strip().split()
            #if int(end) - int(start) != self.window_size:
            #    sys.exit("niestala rozdzielczosc!")
                # tu powinien byc error
                # tylko ze to sie sypie na koncu chromosomu
            self.matrix[-1].append(int(value))
            if chromosome != last_chromosome:
                chromosome_lengths.append(no_of_windows_in_current_chromosome)
                self.chromosome_names.append(chromosome)
                no_of_windows_in_current_chromosome = 0
            no_of_windows_in_current_chromosome += 1
            last_chromosome = chromosome
        chromosome_lengths.append(no_of_windows_in_current_chromosome)
        if not self.chromosome_lengths:
            self.chromosome_lengths = chromosome_lengths
        elif self.chromosome_lengths != chromosome_lengths:
            sys.exit('chromosome lengths between samples don\'t match')
        if sum(self.chromosome_lengths) != len(self.matrix[0]):
            sys.exit("sth\'s wrong with calculating chromosome lengths:" + str(sum(self.chromosome_lengths)) + ' ' + str(len(self.matrix[0])))

    def predict_states(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        self.matrix = numpy.array(self.matrix).transpose()
        print "fitting model"
        self.model.fit(self.matrix, lengths=self.chromosome_lengths)
        print "predicting states"
        probability, states = self.model.decode(self.matrix, lengths=self.chromosome_lengths)
        print "Transmat matrix:", self.model.transmat_
        print "Score:", self.model.score(self.matrix, self.chromosome_lengths)
        print "Probability:", probability
        print "Is convergent:", self.model.monitor_.converged
        return states

    def save_states_to_file(self, states, prefix=''):
        output = open(prefix + "_all_states.txt", 'w')
        for state in states:
            output.write(str(state))
            output.write('\n')
        output.close()
        for state_being_saved in range(self.number_of_states):
            counter = 0
            last_state = 'last_state'
            chromosome_index = 0
            chromosome_name = self.chromosome_names[chromosome_index]
            chromosome_length = self.chromosome_lengths[chromosome_index]
            output = open(prefix + "_state_" + str(state_being_saved) + ".bed", 'w')
            for current_state in states:
                if counter == chromosome_length:
                    if last_state == state_being_saved:
                        output.write('\t'.join([chromosome_name, str(start), str(self.window_size*chromosome_length)]))
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
                output.write('\t'.join([chromosome_name, str(start), str(self.window_size*chromosome_length)]))
                output.write('\n')
            output.close()
