#!/usr/bin/env python

import logging
import numpy

class Data(object):
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
        logging.debug("mediany: %s", str(medians))
        counter = 0
        for which_line, line in enumerate(self.matrix):
            for position, value in enumerate(line):
                if value > threshold:
                    logging.debug("podmieniam %f na %f", (value, medians[which_line]))
                    self.matrix[which_line][position] = medians[which_line]
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

        which_line: integer;
            which line of the matrix should be converted
            (which sample / patient / matrix row);
            indexing 0-relative
        """
        self.matrix = self.matrix.transpose()
        output = []
        previous_value = None
        start, end = 0, None
        previous_chromosome = 0
        window = -1
        for value in self.matrix[which_line]:
            chromosome, window = self.goto_next_window(previous_chromosome, window)
            if chromosome != previous_chromosome:
                end = self.chromosome_ends[previous_chromosome]
            elif value != previous_value and previous_value is not None:
                end = window * self.window_size
            if end is not None:
                output.append((self.chromosome_names[previous_chromosome],
                               start, end, previous_value))
                start = window * self.window_size #+ 1
                # beds are 0-based, half-open, so I think this should work fine.
                end = None
            previous_value, previous_chromosome = value, chromosome
        output.append((self.chromosome_names[-1], start, self.chromosome_ends[-1], value))
        self.matrix = self.matrix.transpose()
        return output

    def goto_next_window(self, chromosome, window):
        window += 1
        if window > self.numbers_of_windows[chromosome] - 1:
            chromosome += 1
            window = 0
        return chromosome, window

    def save_intervals_as_bed(self, output, intervals, condition=None, save_value=False):
        """
        Given set of intervals, saves it to file in bed format.
        Chooses only the intervals with value equal to condition.
        condition = None means all the intervals.
        save_value = False means write only coordinates.
        """
        output = open(output, 'w')
        for interval in intervals:
            if self.check_condition(condition, interval):
                if save_value == False:
                    interval = interval[:-1]
                output.write('\t'.join(map(str, interval)))
                output.write('\n')
        output.close()

    def check_condition(self, condition, interval):
        if condition is None:
            return True
        value = interval[-1]
        return value == condition

    def add_data_from_bedgraph(self, filename):
        logging.info("reading in file %s", filename)
        self.matrix.append([float(line.strip().split()[-1]) for line in open(filename)])

    def prepare_metadata_from_bedgraph(self, filename):
        """
        Set chromosome_names, chromosome_lengths and numbers_of_windows
        basing on a single bedgraph file.
        """
        bedgraph = open(filename)
        previous_chromosome = None
        for line in bedgraph:
            chromosome, start, end, _ = line.strip().split()
            if self.window_size == 1:
                self.window_size = int(end) - int(start) #+ 1
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
        Uses the first one as a source of metadata.

        files: list of filenames (strings)
        """
        self.prepare_metadata_from_bedgraph(files[0])
        for infile in files:
            self.add_data_from_bedgraph(infile)

    #def which_state_is_peaks(self):
    # TODO: check whether mean is the highest among all samples
    #   return self.model.means_.mean(axis=1).argmax()

    def convert_floats_to_ints(self):
        #if any(int(self.matrix) != self.matrix):
        for line in self.matrix:
            for value in line:
                if value != int(value):
                    logging.warning("Warning: your values contain floats,"
                                 " I'm converting them to integers.")
                    # this appears for every file
                    # kind of annoying, would be better if it did only
                    # once *or* for each file but togheter with it's name
                    break
        self.matrix = [[int(i) for i in line] for line in self.matrix]

