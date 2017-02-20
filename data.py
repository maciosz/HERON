#!/usr/bin/env python

import sys
import numpy
import warnings
import logging
from hmmlearn import hmm

class Data:

    def __init__(self, window_size=100, number_of_states=3):
        self.matrix = []
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
        logging.info("reading file" + filename)
        bedgraph = open(filename)
        chromosome_lengths = []
        chromosome_names = []
        self.matrix.append([])
        chromosome, start, end, value = bedgraph.next().strip().split()
        last_chromosome = chromosome
        chromosome_names.append(chromosome)
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
                chromosome_names.append(chromosome)
                no_of_windows_in_current_chromosome = 0
            no_of_windows_in_current_chromosome += 1
            last_chromosome = chromosome
        chromosome_lengths.append(no_of_windows_in_current_chromosome)
        if not self.chromosome_lengths:
            self.chromosome_lengths = chromosome_lengths
            self.chromosome_names = chromosome_names
        elif self.chromosome_lengths != chromosome_lengths:
            sys.exit('chromosome lengths between samples don\'t match')
        elif self.chromosome_names != chromosome_names:
            sys.exit('chromosome names between samples don\'t match')
        if sum(self.chromosome_lengths) != len(self.matrix[0]):
            sys.exit("sth\'s wrong with calculating chromosome lengths:" + str(sum(self.chromosome_lengths)) + ' ' + str(len(self.matrix[0])))

    def add_data_from_bed(self, filename, mode='binary', proportionally=True):
        """
        mode:
            binary - 1 if there is any peak in the window, 0 otherwise
            number - count peaks in the window
            length - summaric length of the peaks in the window
            mean_length - mean length of the peaks in the window
                (? is this really needed?)
            sum_score - sum score of the peaks
            mean_score - calculate average score of the peaks
        proportionally:
            (stupid name, think of sth else)
            applies to modes number, sum_score and mean_score;
            if a peak overlaps window partially,
            count it's score/presence multiplied but the proper fraction
                (there is a discussion in the todo file about what "proper" means)
        """
        logging.info("reading file" + filename)
        bed_file = open(filename)
        for line in bed_file:
            line = line.strip().split()
            chromosome, start, end, name, score, strand = line
            # it should check if it's not a simplified bed with less columns

    def predict_states(self):
        logging.info("predicting states, stay tuned")
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        self.matrix = numpy.array(self.matrix).transpose()
        logging.info("fitting model")
        self.model.fit(self.matrix, lengths=self.chromosome_lengths)
        logging.info( "predicting states")
        probability, states = self.model.decode(self.matrix, lengths=self.chromosome_lengths)
        logging.info("Is convergent:" + str(self.model.monitor_.converged))
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

    def write_stats_to_file(self, prefix):
        output = open(prefix + "_stats.txt", "w")
        output.write("Score:" + str(self.model.score(self.matrix, self.chromosome_lengths)) + '\n')
        #output.write("Probability:" + str(probability) + '\n')
        output.write("Transmat matrix:" + str(self.model.transmat_) + '\n')
        output.write("Means:" + str(self.model.means_) + '\n')
        output.write("Covars:" + str(self.model.covars_) + '\n')
        output.close()
 
