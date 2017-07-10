#!/usr/bin/env python 

import logging
import argparse
from data import Data

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='infiles', action='store', type=str, nargs='+',
                        help='input files (currently only bedgraph format is allowed)')
    parser.add_argument('-s', dest='number_of_states', action='store', type=int, default=3,
                        help='number of states (default: 3)')
    parser.add_argument('-o', dest='output_prefix', action='store', type=str, default='',
                        help='prefix to output files')
    parser.add_argument('-b', dest='bed_file', action='store', type=str,
                        help='optional bed file (currently not used)')
    parser.add_argument('-m', dest='bed_mode', action='store', type=str, default='binary',
                        help='mode for reading in bed file, currently not used')
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    logging.basicConfig(filename=arguments.output_prefix + ".log",
                        filemode='w',
                        level=logging.INFO,
                        format='%(levelname)s\t%(asctime)s\t%(message)s',
                        datefmt="%d.%m.%Y %H:%M:%S")
    logging.info("Creating data structure...")
    data = Data(number_of_states=arguments.number_of_states)
    logging.info("Reading in data...")
    for infile in arguments.infiles:
        data.add_data_from_bedgraph(infile)
    if arguments.bed_file:
        data.add_data_from_bed(arguments.bed_file, arguments.bed_mode)
    logging.debug("Chromosome names: " + str(data.chromosome_names))
    logging.debug("Chromosome lengths: " + str(data.chromosome_lengths))
    logging.info("Data ready to analyse. Finding peaks")
    states = data.predict_states()
    data.save_states_to_file(states, arguments.output_prefix)
    data.write_stats_to_file(arguments.output_prefix)
    data.save_peaks_to_file(arguments.output_prefix)
    logging.info("...done.")

if __name__=='__main__':
    main()

