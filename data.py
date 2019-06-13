#!/usr/bin/env python

import sys
import math
import logging
import collections
import numpy
import pysam

class Data(object):
    """
    Object for reading, storing and writing data
    from bams, coverages and intervals.
    Okay, read from bam and coverage,
    write to intervals.
    """

    def __init__(self):
        """
        Create an empty Data object
        with resolution 1.
        """
        self.matrix = numpy.array([])
        self.window_size = 1
        self.numbers_of_windows = []
        self.chromosome_names = []
        self.chromosome_ends = []

    def filter_data(self, threshold):
        """
        Set data above the given threshold to the median.
        That's just a sketch of what to do with outliers.

        Actually threshold could be given for every patient,
        not as a single value. But for now it's just one float.
        """
        # TODO: adjust to new transposed version of self.matrix
        medians = []
        for line in self.matrix:
            # co? czemu 1000? nie powinno byc threshold??
            median = numpy.median(filter(lambda x: x <= 1000, line))
            medians.append(median)
        #means = map(numpy.mean, filter(lambda x: x <= 1000, self.matrix))
        logging.debug("mediany: %s", str(medians))
        counter = 0
        for which_line, line in enumerate(self.matrix):
            for position, value in enumerate(line):
                if value > threshold:
                    logging.debug("podmieniam %f na %f", value, medians[which_line])
                    self.matrix[which_line][position] = medians[which_line]
                    counter += 1
        logging.info("I've reduced values in %i windows to median value.", counter)

    def split_data(self, threshold):
        """
        Remove windows with value above given threshold
        (in any sample!),
        # for now, but it shouldn't
        splitting chromosome into parts.
        Update chromosome_ends, chromosome_names and numbers_of_windows.
        """
        # TODO: adjust to new transposed version of self.matrix
        # TODO: ...and to the fact that threshold is a vector of thresholds
        current_chromosome = 0
        to_skip = []
        for row_position, line in enumerate(self.matrix):
            for column_position, value in enumerate(line):
                if row_position in to_skip:
                    #continue
                    break
                if value >= threshold[column_position]:
                    logging.debug("splitting at %d!", row_position)
                    to_skip.append(row_position)
        self.matrix = numpy.delete(self.matrix, to_skip, axis=0)
        new_numbers_of_windows = []
        new_names = []
        new_ends = []
        chromosomes_to_split = collections.defaultdict(list)
        for position in to_skip:
            chromosome = self._find_chromosome(position)
            chromosomes_to_split[chromosome].append(position)
        logging.debug(chromosomes_to_split)
        previous_end = -1
        for chromosome in xrange(len(self.chromosome_names)):
            if chromosome > 0:
                #previous_end = self.chromosome_ends[chromosome - 1]
                previous_end = self.numbers_of_windows[chromosome-1] #?
            end = self.chromosome_ends[chromosome]
            name = self.chromosome_names[chromosome]
            number_of_windows = self.numbers_of_windows[chromosome]
            if chromosome not in chromosomes_to_split.keys():
                new_ends.append(end)
                new_names.append(name)
                new_numbers_of_windows.append(number_of_windows)
            else:
                names, ends, numbers_of_windows = self.split_chromosome(chromosomes_to_split[chromosome],
                                                                        chromosome)
                new_names.extend(names)
                new_ends.extend(ends)
                new_numbers_of_windows.extend(numbers_of_windows)

        logging.debug("numbers of windows, chromosome names, chromosome ends:")
        logging.debug("old:")
        logging.debug(self.numbers_of_windows)
        logging.debug(self.chromosome_names)
        logging.debug(self.chromosome_ends)

        self.numbers_of_windows = new_numbers_of_windows
        self.chromosome_names = new_names
        self.chromosome_ends = new_ends

        logging.debug("new:")
        logging.debug(self.numbers_of_windows)
        logging.debug(self.chromosome_names)
        logging.debug(self.chromosome_ends)

    def split_chromosome(self, positions_of_split, chromosome):
        start = -1
        chromosome_ends = numpy.cumsum(self.numbers_of_windows)
        if chromosome > 0:
            start = chromosome_ends[chromosome - 1]
        name = self.chromosome_names[chromosome]
        previous_end = start
        names = []
        ends = []
        numbers_of_windows = []
        positions_of_split.append(chromosome_ends[chromosome])
        for nr, position in enumerate(positions_of_split):
            number_of_windows = position - previous_end - 1
            if number_of_windows != 0:
                names.append(name+ "_" + str(nr))
                numbers_of_windows.append(number_of_windows)
                # to nie uwzglednia koncow chromosomow, one maja inny end
                # ale tez jesli to sie dzieje tylko dla fitowania
                # to nie jest to tak naprawde potrzebne
                ends.append(number_of_windows * self.window_size)
            previous_end = position
        return names, ends, numbers_of_windows

    def _find_chromosome(self, position):
        logging.debug("Looking for %d", position)
        chromosome_ends = numpy.cumsum(self.numbers_of_windows)# * self.window_size
        chromosome_ends = numpy.append([0], chromosome_ends)
        logging.debug("Chromosome ends:")
        logging.debug(chromosome_ends)
        for number, (start, end) in enumerate(zip(chromosome_ends[:-1], chromosome_ends[1:])):
            logging.debug("number %i, start %i end %i", number, start, end)
            if start <= position < end:
                return number

    def find_threshold_value(self, threshold, factor=0.001):
        """
        Say we want to remove threshold * factor (threshold promils by default)
        windows with the highest values.
        It's easier to remove windows with value above some x.
        So this method finds the x for the desired threshold.
        """
        #sorted_values = numpy.sort(self.matrix.flatten())
        #threshold_index = int(len(sorted_values) * threshold * factor)
        #threshold_value = sorted_values[threshold_index]
        #return threshold_value
        #It might be better to return a list of values,
        #one for each sample:
        sorted_matrix = - numpy.sort(- self.matrix, axis=0)
        threshold_index = int(sorted_matrix.shape[0] * threshold * factor)
        threshold_values = sorted_matrix[threshold_index, :]
        logging.debug("Na poziomie %f thresholdy wynosza:", threshold * factor)
        logging.debug(threshold_values)
        return threshold_values
        # jezeli bedzie duzo okien o takich wartosciach to usune znacznie wiecej niz threshold.
        # ale nie wiem czy to nam przeszkadza.

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
            (which sample / patient / matrix column);
            indexing 0-based
        """
        #self.matrix = self.matrix.transpose()
        output = []
        previous_value = None
        start, end = 0, None
        previous_chromosome = 0
        window = -1
        for value in self.matrix[:, which_line]:
            try:
                # chyba moge wywalic tego try'a?
                chromosome, window = self._goto_next_window(previous_chromosome, window)
            except IndexError:
                logging.error("Index error w goto_next_window")
                logging.error("Zapisuje %d linie", which_line)
                logging.error("Jestem na chromosomie %d, w oknie %d", previous_chromosome, window)
                logging.error("Atrybuty chromosomowe:")
                logging.error(str(self.numbers_of_windows))
                logging.error(str(self.chromosome_names))
                logging.error(str(self.chromosome_ends))
                break
            if chromosome != previous_chromosome:
                end = self.chromosome_ends[previous_chromosome]
            elif value != previous_value and previous_value is not None:
                end = window * self.window_size
            if end is not None:
                output.append([self.chromosome_names[previous_chromosome],
                               start, end, previous_value])
                start = window * self.window_size #+ 1
                # beds are 0-based, half-open, so I think this should work fine.
                end = None
            previous_value, previous_chromosome = value, chromosome
        output.append([self.chromosome_names[-1], start, self.chromosome_ends[-1], value])
        #self.matrix = self.matrix.transpose()
        return output

    def _goto_next_window(self, chromosome, window):
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
            if _check_condition(condition, interval):
                if save_value is False:
                    interval = interval[:-1]
                else:
                    interval[-1] = int(interval[-1])
                output.write('\t'.join(map(str, interval)))
                output.write('\n')
        output.close()

    def add_data_from_bedgraph(self, filename):
        """
        Add coverage data from single bedgraph file.
        """
        logging.info("reading in file %s", filename)
        #self.matrix.append([float(line.strip().split()[-1]) for line in open(filename)])
        # aa czy tu nie moglabym zamiast float dac jakies self.type?
        # i zmieniac go w zaleznosci od distr
        new_line = [[float(line.strip().split()[-1])] for line in open(filename)]
        if self.matrix.shape == (0,):
            self.matrix = numpy.array(new_line)
        else:
            self.matrix = numpy.append(self.matrix, new_line, axis=1)

    def prepare_metadata_from_bedgraph(self, filename):
        """
        Set chromosome_names, chromosome_ends and numbers_of_windows
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

    #def which_state_is_peaks(self):
    # TODO: check whether mean is the highest among all samples
    #   return self.model.means_.mean(axis=1).argmax()

    def prepare_metadata_from_bam(self, filename, resolution):
        """
        Set chromosome_names, chromosome_ends and numbers_of_windows
        basing on a single bam file.
        Set window_size to given resolution.
        """
        self.window_size = resolution
        bam = pysam.AlignmentFile(filename)
        self.chromosome_names = list(bam.references)
        #self.chromosome_ends = [bam.get_reference_length(chromosome)
        #                           for chromosome in self.chromosome_names]
        # zaleznie od wersji pysama
        # to nizej jest juz deprecated w najnowszej ale wciaz dziala
        self.chromosome_ends = list(bam.lengths)
        self.numbers_of_windows = [int(math.ceil(float(length) / resolution))
                                   for length in self.chromosome_ends]

    def add_data_from_bam(self, filename, mean=True):
        """
        Add coverage data from bam file.
        Assumes some metadata is already added.
        """
        #TODO: zmienic na zapisywanie "transponowane", w bedgraphach juz jest
        logging.info("reading in file %s", filename)
        resolution = self.window_size
        bam = pysam.AlignmentFile(filename)
        windows = []
        counter = 0
        for chr_id, chromosome in enumerate(self.chromosome_names):
            pileup = bam.pileup(reference=chromosome)
            try:
                first_read = pileup.next().pos
            except StopIteration:
                windows.extend([0] * self.numbers_of_windows[chr_id])
                continue
                # no reads mapped to this chromosom
                # should I remove these chromosomes
                # or (like now) add zeros?
                # Adding zeros it's easier, I don't have to check anything
                # between the samples.
            first_window = first_read / resolution
            for window in xrange(self.numbers_of_windows[chr_id]):
                counter += 1
                if window < first_window:
                    windows.append(0)
                    continue
                if counter % 1000 == 0:
                    logging.debug("%d windows processed", counter)
                start = window * resolution
                end = start + resolution
                #pileup = bam.pileup(reference=chromosome,
                #                    start=start, end=end)
                value = sum(position.n for position in pileup if start <= position.pos < end)
                # pileup bierze ready zazebiajace sie z tym regionem
                # ale w szczegolnosci tez wychodzace z niego
                # also jesli jest pusty to to bedzie zero, wiec ok
                # ...a skoro i tak musze tak filtrowac to niepotrzebne jest robienie nowych pileupow
                # to bardzo mocno wydluza
                # a wziecie jednego na wszystkie chromosomy to przesada w druga strone,
                # tez wydluza
                if mean:
                    value = float(value) / resolution
                windows.append(value)
        logging.debug("Dlugosc tego pliku: %d", len(windows))
        self.matrix.append(windows)

    def prepare_metadata_from_file(self, filename, resolution):
        """
        Set chromosome_names, chromosome_ends and numbers_of_windows
        basing on a single bedgraph/bam file.
        Guess the type basing on suffix.
        """
        if filename.endswith("bedgraph"):
            self.prepare_metadata_from_bedgraph(filename)
        elif filename.endswith("bam"):
            self.prepare_metadata_from_bam(filename, resolution)
        else:
            logging.error("Unknown file type: %s",
                          filename.split(".")[-1])
            sys.exit()

    def add_data_from_file(self, filename, mean):
        """
        Add data from a single file.
        Guess the type basing on suffix.
        """
        if filename.endswith("bedgraph"):
            self.add_data_from_bedgraph(filename)
        elif filename.endswith("bam"):
            self.add_data_from_bam(filename, mean)
        else:
            logging.error("Unknown file type: %s",
                          filename.split(".")[1])
            sys.exit()

    def add_data_from_files(self, filenames, resolution, mean):
        """
        Add data from multiple files.
        Use the first one as a source of metadata.

        files: list of filenames (strings)
        resoluition: desired window size (int)
            (used only for reading bams)
        """
        self.prepare_metadata_from_file(filenames[0], resolution)
        for filename in filenames:
            self.add_data_from_file(filename, mean)
        #logging.debug("Wymiary macierzy: %d", len(self.matrix))
        #logging.debug("Liczba kolumn:  %d", len(self.matrix[0]))
        logging.debug("Matrix dimensions: %s", str(self.matrix.shape))

    def convert_floats_to_ints(self):
        #if any(int(self.matrix) != self.matrix):
        #for line in self.matrix:
        #    for value in line:
        #        if value != int(value):
        #            logging.warning("Warning: your values contain floats,"
        #                            " I'm converting them to integers.")
        #            # this appears for every file
        #            # kind of annoying, would be better if it did only once
        #            # *or* for each file but togheter with it's name
        #            break
        #self.matrix = [[int(i) for i in line] for line in self.matrix]
        if numpy.any(self.matrix != self.matrix.astype(int)):
            logging.warning("Warning: your values contain floats,"
                            " I'm converting them to integers.")
        self.matrix = self.matrix.astype(int)

    def calculate_quantiles(self, levels):
        """
        Calculate desired quantiles for every sample,
        excluding zero values.
        """
        #quantiles = numpy.quantile(self.matrix, levels, axis=0)
        n_samples = self.matrix.shape[1]
        quantiles = numpy.zeros((len(levels), n_samples))
        for sample in xrange(n_samples):
            values = self.matrix[:, sample]
            values = values[values != 0]
            sample_quantiles = numpy.quantile(values, levels)
            quantiles[:, sample] = sample_quantiles
        return quantiles

def _check_condition(condition, interval):
    if condition is None:
        return True
    value = interval[-1]
    return value == condition
