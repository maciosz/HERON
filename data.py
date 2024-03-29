#!/usr/bin/env python3.6

import math
import logging
import collections
import numpy
import pysam

class Data():
    """
    Object for reading, storing and writing data
    from bams, coverages and intervals.
    Okay, read from bam and coverage,
    write to intervals.
    """

    def __init__(self):
        """
        Create an empty Data object with resolution 1.
        """
        self.matrix = numpy.array([])
        self.window_size = 1
        self.numbers_of_windows = []
        self.chromosome_names = []
        self.chromosome_ends = []
        self.states = None
        self.intervals = None
        self.scores = []
        self.posteriors = []

    def merge_data(self):
        self.matrix = self.matrix.sum(axis=1)[:, numpy.newaxis]

    def filter_data(self, threshold):
        """
        Set data above the given threshold to the median.
        That's just a sketch of what to do with outliers.

        Actually threshold could be given for every patient,
        not as a single value. But for now it's just one float.
        """
        medians = []
        for line in self.matrix.T:
            median = numpy.median([x for x in line if x <= threshold])
            medians.append(median)
        logging.debug("Medians: %s", str(medians))
        counter = 0
        for which_line, line in enumerate(self.matrix.T):
            for position, value in enumerate(line):
                if value > threshold:
                    #logging.debug("Changing %f to %f", value, medians[which_line])
                    try:
                        self.matrix.T[which_line][position] = medians[which_line]
                    except IndexError:
                        print(which_line, line)
                        print(medians)
                        print(self.matrix.shape)
                    counter += 1
        logging.info("I've reduced values in %i windows to median value.", counter)

    def split_data(self, threshold):
        """
        Remove windows with value above given threshold
        splitting chromosome into parts.
        Update chromosome_ends, chromosome_names and numbers_of_windows.
        """
        to_skip = []
        for row_position, line in enumerate(self.matrix):
            for column_position, value in enumerate(line):
                if row_position in to_skip:
                    break
                if value >= threshold[column_position]:
                    #logging.debug("splitting at %d!", row_position)
                    to_skip.append(row_position)
        self._split_data(to_skip)

    def _split_data(self, to_skip):
        """
        Split chromosomes into parts, removing positions listed in to_skip.
        Update chromosome_ends, chromosome_names and numbers_of_windows.
        """
        self.matrix = numpy.delete(self.matrix, to_skip, axis=0)
        new_numbers_of_windows = []
        new_names = []
        new_ends = []
        chromosomes_to_split = collections.defaultdict(list)
        for position in to_skip:
            chromosome = self._find_chromosome(position)
            chromosomes_to_split[chromosome].append(position)
        logging.debug(chromosomes_to_split)
        for chromosome in range(len(self.chromosome_names)):
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
        """
        Given list of positions to make a split and index of chromosome,
        return list of new names, ends and numbers_of_windows
        that this chromosome was splitted to.
        """
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
            if number_of_windows > 0:
                names.append(name+ "_" + str(nr))
                numbers_of_windows.append(number_of_windows)
                # to nie uwzglednia koncow chromosomow, one maja inny end
                # ale tez jesli to sie dzieje tylko dla fitowania
                # to nie jest to tak naprawde potrzebne
                ends.append(number_of_windows * self.window_size)
            previous_end = position
        return names, ends, numbers_of_windows

    def _find_chromosome(self, position):
        """
        In which chromosome given position (window) occurs.
        """
        #logging.debug("Looking for %d", position)
        chromosome_ends = numpy.cumsum(self.numbers_of_windows)# * self.window_size
        chromosome_ends = numpy.append([0], chromosome_ends)
        #logging.debug("Chromosome ends:")
        #logging.debug(chromosome_ends)
        for number, (start, end) in enumerate(zip(chromosome_ends[:-1], chromosome_ends[1:])):
            #logging.debug("number %i, start %i end %i", number, start, end)
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

    def states_to_intervals(self):
        """
        Convert data stored in self.states as windows
        to intervals, savable in bed.
        That is - merge neighbouring windows
        if they have the same value,
        and set proper coordinates at the end of chromosomes.

        Returns list of tuples (chr, start, end, value).
        (does it have to return anything if it sets an attribute?..)
        """
        output = []
        previous_value = None
        start, end = 0, None
        previous_chromosome = 0
        window = -1
        if self.states is None:
            raise ValueError("States not predicted yet.")
        for value in self.states:
            chromosome, window = self._goto_next_window(previous_chromosome, window)
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
        self.intervals = output
        return output

    def _goto_next_window(self, chromosome, window):
        window += 1
        if window > self.numbers_of_windows[chromosome] - 1:
            chromosome += 1
            window = 0
        return chromosome, window

    def add_data_from_bedgraph(self, filename):
        """
        Add coverage data from single bedgraph file.
        """
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
                    self.chromosome_ends.append(int(previous_end))
            else:
                self.numbers_of_windows[-1] += 1
            previous_end, previous_chromosome = end, chromosome
        self.chromosome_ends.append(int(end))

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

    def add_data_from_bam(self, filename, mean):
        """
        Add coverage data from bam file.
        Assumes some metadata is already added.
        """
        resolution = self.window_size
        bam = pysam.AlignmentFile(filename)
        windows = []
        #length, prev_length = 0, 0
        for chr_id, chromosome in enumerate(self.chromosome_names):
            pileup = bam.pileup(reference=chromosome)
            try:
                first_read = next(pileup).pos
            except StopIteration:
                windows.extend([0] * self.numbers_of_windows[chr_id])
                continue
            current_window = int(first_read / resolution)
            # adding zeros if first read is not in the first window
            windows.extend([0] * (current_window))
            previous_window = current_window - 1
            start = current_window * resolution
            end = start + resolution
            values = []
            # robie jeszcze raz bo tamto next wyzej mi zuzylo jedna pozycje
            # wiem ze mozna zresetowac ale nie pamietam jak
            pileup = bam.pileup(reference=chromosome)
            for position in pileup:
                if start <= position.pos < end:
                    values.append(position.n)
                elif position.pos >= end:
                    value = sum(values)
                    if mean:
                        value = float(value) / resolution
                    windows.append(value)
                    current_window = int(position.pos / resolution)
                    if current_window != previous_window + 1:
                        windows.extend([0] * (current_window - previous_window - 1))
                    start = current_window * resolution
                    end = start + resolution
                    values = [position.n]
                previous_window = current_window
            if current_window != (self.numbers_of_windows[chr_id] - 1):
                final_window_length = resolution
            else:
                final_window_length = self.chromosome_ends[chr_id] % resolution
                if final_window_length == 0:
                    final_window_length = resolution
            value = sum(values)
            if mean:
                value = float(value) / final_window_length
            windows.append(value)
            windows.extend([0] * (self.numbers_of_windows[chr_id] - current_window - 1))
            #length = len(windows)
            #prev_length = length
        windows = numpy.array(windows)[:, numpy.newaxis]
        if self.matrix.shape == (0,):
            self.matrix = numpy.array(windows)
        else:
            self.matrix = numpy.append(self.matrix, windows, axis=1)

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
            raise ValueError("Unknown file type: %s" %
                             filename.split(".")[-1])

    def add_data_from_file(self, filename, mean):
        """
        Add data from a single file.
        Guess the type basing on suffix.
        """
        logging.info("reading in file %s", filename)
        if filename.endswith("bedgraph"):
            self.add_data_from_bedgraph(filename)
        elif filename.endswith("bam"):
            self.add_data_from_bam(filename, mean)
        else:
            raise ValueError("Unknown file type: %s" %
                             filename.split(".")[1])

    def add_data_from_files(self, filenames, resolution=100,
                            mean=True, prepare_metadata=True):
        """
        Add data from multiple files.
        Use the first one as a source of metadata.

        files: list of filenames (strings)
        resoluition: desired window size (int)
            (used only for reading bams)
        mean: bool; whether to calculate mean coverage in windows
            or not (summaric coverage)
            (used only for reading bams)
        """
        if prepare_metadata:
            self.prepare_metadata_from_file(filenames[0], resolution)
        for filename in filenames:
            self.add_data_from_file(filename, mean)
        if len(self.matrix.shape) == 1:
            self.matrix = self.matrix.reshape((self.matrix.shape[0], 1))
        logging.debug("Matrix dimensions: %s", str(self.matrix.shape))

    def convert_floats_to_ints(self):
        """
        Converts floats to integers in self.matrix.
        Issues a warning if that actually was necessary,
        i.e. if any entry changed its value
        because of this conversion.
        """
        if numpy.any(self.matrix != self.matrix.astype(int)):
            logging.warning("Warning: your values contain floats,"
                            " I'm converting them to integers.")
        self.matrix = self.matrix.astype(int)

    def calculate_quantiles(self, levels):
        """
        Calculate desired quantiles for every sample,
        excluding zero values.

        (Why excluding zeros?)
        """
        #quantiles = numpy.quantile(self.matrix, levels, axis=0)
        n_samples = self.matrix.shape[1]
        quantiles = numpy.zeros((len(levels), n_samples))
        for sample in range(n_samples):
            values = self.matrix[:, sample]
            values = values[values != 0]
            sample_quantiles = numpy.quantile(values, levels)
            quantiles[:, sample] = sample_quantiles
        return quantiles

    def add_column(self, column):
        """
        Add given column to self.matrix.
        """
        self.matrix = numpy.c_[self.matrix, column]

    def remove_column(self, column_index):
        """
        Remove given column from self.matrix.
        """
        self.matrix = numpy.delete(self.matrix, column_index, 1)

    def normalise_column(self, signal_column, control_column,
                         how="log_diff"):
        """
        signal_column - index of column with signal (the one we want to normalise)
        control_column - index of column with control (used to normalise)
        how - how signal should be normalised? Supported:
            "log_diff" - log(signal / control) = log(signal) - log(control)
            "diff" - signal - control

        Modifies self.matrix in place. Doesn't return anything.
        """
        signal = self.matrix[:, signal_column]
        control = self.matrix[:, control_column]
        if how == "log_diff":
            #normalized = numpy.log(signal / control)
            normalized = numpy.log((signal+1) / (control+1))
        if how == "diff":
            normalized = signal - control
        self.matrix[:, signal_column] = normalized

    def normalise_signals(self, signal_columns, control_columns,
                          how="log_diff"):
        """
        Normalise columns with signal samples using columns with control samples.
        Afterwards remove the control columns.
        Modifies self.matrix in place. Doesn't return anything.

        signal_columns - list of indexes of columns with signal
        control_columns - list of indexes of columns with control
        how - how signals should be normalised. See normalise_column()

        Uwaga: w ogolnosci moglby uzytkownik chciec podac kontrole do kilku sygnalow
        a do kilku nie. Tylko nie jestem pewna jak by to zaimplementowac
        po stronie uzytkownika, w sensie jak wygodnie podawac takie opcje.
        """
        if len(control_columns) == 1 and len(signal_columns) > 1:
            control_columns = numpy.repeat(control_columns, len(signal_columns))
        elif len(control_columns) != len(signal_columns):
            raise ValueError("Number of samples and controls don't match."
                             " I can either take one control and many samples"
                             " or exactly as many controls as samples."
                             " I've got %d signals and %d controls" %
                             (len(signal_columns), len(control_columns)))
        for signal, control in zip(signal_columns, control_columns):
            self.normalise_column(signal, control, how)
        #for column in control_columns:
        #    self.remove_column(column)
        self.matrix = numpy.delete(self.matrix, control_columns, 1)
        is_valid = numpy.all(numpy.isfinite(self.matrix), axis=1)
        invalid_rows = numpy.where(~is_valid)[0]
        invalid_rows = [int(x) for x in invalid_rows]
        logging.debug("invalid rows:")
        logging.debug(invalid_rows)
        logging.debug(type(invalid_rows))
        self._split_data(invalid_rows)
        # to splitowanie nie jest juz potrzebne
        # dodalam bo wychodzily nan przy logarytmoawniu zera i dzieleniu przez zero
        # natomiast nawet jak cos pojdzie nie tak i sie to splitowanie wydarzy,
        #  to predykowanie stanow tego nie uwzglednia chyba?
        # Wydaje mi sie ze cos dalej bylo nie tak z rozspojnionymi koordynatami.

    def save_intervals(self, output, condition=None,
                       save_value=False, save_score=False,
                       which_score=3):
        if self.intervals is None:
            self.states_to_intervals()
        output = open(output, 'w')
        for nr, interval in enumerate(self.intervals):
            if _check_condition(condition, interval):
                if save_value is False:
                    interval = interval[:-1]
                else:
                    # ? po co to?
                    interval[-1] = int(interval[-1])
                if save_score:
                    if type(which_score) is int:
                        if which_score >= len(self.scores[nr]):
                            raise ValueError("I have %s types of scores to choose from,"
                                             " and you want the score with index %s."
                                             " With zero-relative indexing. Good luck with that."
                                             % (len(self.scores[nr]), which_score))
                        score = [self.scores[nr][which_score]]
                    elif which_score == "all":
                        score = self.scores[nr]
                    else:
                        raise ValueError("which_score should be either an index"
                                         " or string 'all'. You wanted %s."
                                         " I don't understand it, sorry." % which_score)
                    interval.extend(score)
                output.write('\t'.join(map(str, interval)))
                output.write('\n')
        output.close()

    def score_peaks(self, which):
        """
        which - int; which state represents peaks
        to nie jest uzywane...
        moze moglabym robic continue jak state != which
        tylko wtedy trzeba tez uwzglednic przy zapisywaniu do pliku
            ze score'y beda tylko dla pikow
        """
        posteriors = self.posteriors
        if self.intervals is None:
            self.states_to_intervals()
        for interval in self.intervals:
            chrom, start, end, state = interval
            scores = self.score_peak(start, end, state)
            self.scores.append(scores)

    def score_peak(self, start, end, state):
        length = end - start
        start = int(numpy.floor(start / self.window_size))
        end = int(numpy.floor(end / self.window_size))
        # end is in this window, but to take the right slice we need to add 1
        end += 1
        posteriors = self.posteriors[start:end, state]
        coverages = self.matrix[start:end, :]
        try:
            posteriors_scores = _get_scores_from_values(posteriors)
        except ValueError:
            logging.warning("ValueError in scoring posteriors")
            logging.warning("start %d, end %d, posteriors:" % (start, end))
            logging.warning(str(posteriors))
            posteriors_scores = _get_scores_from_values([0])
        coverage_scores = _get_scores_from_values(coverages)
        scores = [#posteriors_scores['mean'],
            posteriors_scores['product'],
            posteriors_scores['median'],
            posteriors_scores['max'],
            coverage_scores['mean'],
            coverage_scores['median'],
            coverage_scores['max'],
            #coverage_scores['sum'],
            length]
        return scores

# w tej chwili to jest nieuzywane, i chyba juz nie bedzie
def _get_score_from_posteriors(posteriors):
    # na razie po prostu max
    chosen_posteriors = max(posteriors)
    # i robie log(1-x) * -10 zeby bylo bardziej human readable
    return _transform_posteriors_for_readability(chosen_posteriors)
    # ...tylko czy to posteriors z hmmlearn to juz nie jest log?
    # Po wartosciach patrzac nie. Po dokumentacji - tak.

def _check_condition(condition, interval):
    if condition is None:
        return True
    value = interval[-1]
    return value == condition

def _get_scores_from_values(values):
    # a co kiedy pokrycia sa wielowymiarowe?
    # imho niewiele to zmienia, musze je tylko splaszczyc.
    # (a nawet nie, bo funkcje numpy'owe traktuja je po prostu jak splaszczone.)
    # Posteriors wciaz bedzie jednowymiarowe,
    #  a co do pokryc to moze mnie interesowac max / mean etc
    #  po wszystkich wartosciach, tj. po oknach i po probkach.
    # Ew. moge najpierw usrednic po oknach i z tego brac max / mean.
    # Moge jedna i druga wersje zapisywac.
    functions = {'mean':numpy.mean, 'median':numpy.median, 'max':numpy.max,
                 'product':numpy.product, 'sum':numpy.sum}
    scores = {}
    for name, function in functions.items():
        score = function(values)
        scores[name] = score
    return scores

def _transform_posteriors_for_readability(posteriors):
    # robie log(1-x) * -10 zeby bylo bardziej human readable
    # wydaje mi sie ze dziesietny algorytm jest bardziej intuicyjny tutaj
    return numpy.log10(1 - posteriors) * -10
