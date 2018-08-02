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

    def __init__(self):
        """
        Create an empty Data object
        with resolution 1.
        """
        self.matrix = []
        self.window_size = 1
        self.numbers_of_windows = []
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

    def windows_to_intervals(self, which_line=0):
        """
        Convert data stored in self.matrix as windows
        to intervals, savable in bed.
        That is - merge neighbouring windows
        if they have the same value,
        and set proper coordinates at the end of chromosomes.
        
        Returns list of tuples (chr, start, end, value).

        which_line - integer;
            which line of the matrix should be converted
            (which sample / patient / matrix row);
            indexing 0-relative
        """
        output = []
        previous_value = False
        start, end = 1, False
        previous_chromosome = 0
        window = -1
        for value in self.matrix[which_line]:
            chromosome, window = self.goto_next_window(previous_chromosome, window)
            if chromosome != previous_chromosome:
                end = self.chromosome_ends[previous_chromosome]
            elif value != previous_value and previous_value:
                end = window * self.window_size
            if end:
                output.append((self.chromosome_names[previous_chromosome], start, end, previous_value))
                start = window * self.window_size + 1
                end = False
            previous_value, previous_chromosome = value, chromosome
        output.append((self.chromosome_names[-1], start, self.chromosome_ends[-1], value))
        return output
                
    def goto_next_window(self, chromosome, window):
        window += 1
        if window > self.numbers_of_windows[chromosome] - 1:
            chromosome += 1
            window = 0
        return chromosome, window

    def save_intervals_as_bed(self, output, intervals, condition=None):
        output = open(output, 'w')
        for interval in intervals:
            if check_condition(condition, interval):
                output.write('\t'.join(map(str, interval)))
                output.write('\n')
        output.close()

    def check_condition(condition, interval):
        if not condition:
            return True
        value = interval[-1]
        return value == condition
        
    def add_data_from_bedgraph(self, filename):
        self.matrix.append([float(line.strip().split()[-1]) for line in open(filename)])

    def prepare_metadata_from_bedgraph(self, filename):
        """
        Set chromosome_names, chromosome_lengths and numbers_of_windows
        basing on a single bedgraph file.
        """
        bedgraph = open(filename)
        previous_chromosome = None
        for line in bedgraph:
            chromosome, start, end, value = line.strip().split()
            if chromosome != previous_chromosome:
                self.chromosome_names.append(chromosome)
                self.numbers_of_windows.append(1)
                if previous_chromosome:
                    self.chromosome_ends.append(previous_end)
            else:
                self.numbers_of_windows[-1] += 1
            previous_end, previous_chromosome = end, chromosome
        self.chromosome_ends.append(end)
        

    def add_data_from_bedgraphs(self, files):
        """
        Add data from multiple bedgraphs.
        Uses the first one as the source of metadata.
        """
        self.prepare_metadata_from_bedgraph(files[0])
        for infile in files:
            self.add_data_from_bedgraph(infile)



    def save_states_to_file(self, states, prefix=''):
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

    def convert_floats_to_ints(self):
        #if any(int(self.matrix) != self.matrix):
        for line in self.matrix:
            for value in line:
                if value != int(value):
                    logging.info("Warning: your values contain floats,"
                                 " I'm converting them to integers")
                    break
        self.matrix = [[int(i) for i in line] for line in self.matrix]

