import sys
import numpy
from hmmlearn import hmm

class Data:

    def __init__(self, window_size=100):
        self.matrix = [] #numpy.array()
        self.window_size = window_size

    def add_data_from_bedgraph(self, filename):
        bedgraph = open(filename)
        self.matrix.append([])
        chromosome, start, end, value = bedgraph.next().strip().split()
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
            if int(end) - int(start) != self.window_size:
                sys.exit("niestala rozdzielczosc!")
                # tu powinien byc error
            self.matrix[-1].append(int(value))

    def find_peaks(self):
        self.matrix = numpy.array(self.matrix).transpose()
        model = hmm.GaussianHMM(2, covariance_type='full')
        model.fit(self.matrix)
        states = model.predict(self.matrix)
        peaks = self.states_to_peaks(states)
        return peaks

    def states_to_peaks(self, states):
        peaks = []
        counter = 0
        last_state = 0
        for state in states:
            if state and not last_state:
                peaks.append([self.window_size * counter])
            elif not state and last_state:
                peaks[-1].append(self.window_size * counter)
            counter += 1
            last_state = state
        if len(peaks[-1]) == 1:
            peaks[-1].append(self.window_size * counter)    # obczaic czy nie +/- 1
        return peaks
