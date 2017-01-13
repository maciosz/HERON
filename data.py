import numpy
from hmmlearn import hmm

class Data:

    def __init__(self, window_size=100):
        self.matrix = [] #numpy.array()
        self.windows_size = window_size

    def add_data_from_bedgraph(self, filename):
        bedgraph = open(filename)
        self.matrix.append([])
        for line in bedgraph:
            chromosome, start, end, value = line.strip().split()
            self.matrix[-1].append(int(value))

    def find_peaks(self):
        self.matrix = numpy.array(self.matrix)
        self.matrix = self.matrix.transpose()
        model = hmm.GaussianHMM(2, covariance_type='full')
        model.fit(self.matrix)
        states = model.predict(self.matrix)
        return states
