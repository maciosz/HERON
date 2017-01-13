#!/usr/bin/python
import argparse
from hmmlearn import hmm
import coverage
from data import Data

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='infiles', action='store', type=str, nargs='+',
                        help='input files')
    return parser.parse_args()

def read_bedgraph_file(filename):
    return coverage.Coverage(filename, 'bedgraph')

def main():
    arguments = parse_arguments()
    infiles = [read_bedgraph_file(i) for i in arguments.infiles]
    data = Data()
    for infile in arguments.infiles:
        data.add_data_from_bedgraph(infile)
    peaks = data.find_peaks()
    print data.matrix
    print peaks

if __name__=='__main__':
    main()

