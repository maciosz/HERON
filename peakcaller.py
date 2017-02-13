#!/usr/bin/python
import argparse
from hmmlearn import hmm
import coverage
from data import Data

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='infiles', action='store', type=str, nargs='+',
                        help='input files')
    parser.add_argument('-s', dest='number_of_states', action='store', type=int, default=3,
                        help='number of states (default: 3)')
    parser.add_argument('-o', dest='output', action='store', type=str,
                        help='output (currently not used)')
    return parser.parse_args()

def read_bedgraph_file(filename):
    return coverage.Coverage(filename, 'bedgraph')
    # do i even need that?

def main():
    arguments = parse_arguments()
    #infiles = [read_bedgraph_file(i) for i in arguments.infiles]
    data = Data(number_of_states=arguments.number_of_states)
    for infile in arguments.infiles:
        data.add_data_from_bedgraph(infile)
    peaks = data.find_peaks()
    #print data.matrix
    for start, end in peaks:
        print '\t'.join((str(start), str(end)))

if __name__=='__main__':
    main()

