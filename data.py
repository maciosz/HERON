import sys
import numpy
from hmmlearn import hmm

class Data:

    def __init__(self, window_size=100, number_of_states=3):
        self.matrix = [] #numpy.array()
        self.window_size = window_size
        self.number_of_states = number_of_states
        #self.model = hmm.GaussainHMM(number_of_states)

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
            #if int(end) - int(start) != self.window_size:
            #    sys.exit("niestala rozdzielczosc!")
                # tu powinien byc error
                # tylko ze to sie sypie na koncu chromosomu
            self.matrix[-1].append(int(value))

    def find_peaks(self):
        self.matrix = numpy.array(self.matrix).transpose()
        model = hmm.GaussianHMM(self.number_of_states)#, covariance_type='full')
        model.fit(self.matrix)
        #print self.matrix
        states = model.predict(self.matrix)
        self.save_states_to_file(states)
        peaks = self.states_to_peaks(states)
        print "Transmat matrix:", model.transmat_
        return peaks

    def states_to_peaks(self, states):
        """
        Assumes that peaks consist of non-zero state.
        Which is stupid.
        """
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
        if peaks and len(peaks[-1]) == 1:
            peaks[-1].append(self.window_size * counter)    # obczaic czy nie +/- 1
        return peaks

    def save_states_to_file(self, states):
        output = open("states", 'w')
        for state in states:
            output.write(str(state))
            output.write('\n')

        #ten kawalek z jakiegos powodu nie dziala
        # moze teraz zadziala jak zmienilam poczatkowa wartosc last_state
        # bo dla False porownanie False == 0 daje True
        # i moze za wczesnie wejsc w elif
        for state_being_saved in range(self.number_of_states):
            counter = 0
            last_state = 'last_state'
            output = open("state_" + str(state_being_saved) + ".bed", 'w')
            for current_state in states:
                if current_state == state_being_saved and last_state != state_being_saved:
                    region = [self.window_size * counter]
                elif current_state != state_being_saved and last_state == state_being_saved:
                    try:
                        region.append(self.window_size * counter)
                    except:
                        print counter
                        print states[counter-20:counter+20]
                        sys.exit()
                    output.write('\t'.join(['chr6', str(region[0]), str(region[1])]))
                    output.write('\n')
                counter += 1
                last_state = current_state
            output.close()
                    
