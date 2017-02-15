#!/usr/bin/env python 

import argparse
from hmmlearn import hmm
from data import Data

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='infiles', action='store', type=str, nargs='+',
                        help='input files (currently only bedgraph format is allowed')
    parser.add_argument('-s', dest='number_of_states', action='store', type=int, default=3,
                        help='number of states (default: 3)')
    parser.add_argument('-o', dest='output_prefix', action='store', type=str,
                        help='prefix to output files')
    parser.add_argument('-b', dest='bed_file', action='store', type=str,
                        help='optional bed file (currently not used)')
    parser.add_argument('-m', dest='bed_mode', action='store', type=str,
                        help='mode for reading in bed file, currently not used')
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    print "Creating data structure..."
    data = Data(number_of_states=arguments.number_of_states)
    print "Reading in data..."
    for infile in arguments.infiles:
        data.add_data_from_bedgraph(infile)
    print "Chromosome names:", data.chromosome_names
    print "Chromosome lengths:", data.chromosome_lengths
    print "Data ready to analyse. Finding peaks"
    states = data.predict_states()
    data.save_states_to_file(states, arguments.output_prefix)
    "...done."

if __name__=='__main__':
    main()

