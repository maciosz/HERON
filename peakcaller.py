#!/usr/bin/env python

import sys
import logging
import argparse
from data import Data

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    source:
    https://www.electricmonk.nl/log/2011/08/14/redirect-stdout-and-stderr-to-a-logger-in-python/
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='infiles',
                        action='store', type=str, nargs='+',
                        help='input files (currently only bedgraph format is allowed)')
    parser.add_argument('-s', dest='number_of_states',
                        action='store', type=int, default=3,
                        help='number of states (default: 3)')
    parser.add_argument('-o', dest='output_prefix',
                        action='store', type=str, default='',
                        help='prefix to output files')
    parser.add_argument('-d', dest='distribution',
                        action='store', type=str, default='NB',
                        help='distribution of emissions; "Gauss" or "NB" (default)')
    parser.add_argument('-b', dest='bed_file',
                        action='store', type=str,
                        help='optional bed file (currently not used)')
    parser.add_argument('-m', dest='bed_mode',
                        action='store', type=str, default='binary',
                        help='mode for reading in bed file, currently not used')
    parser.add_argument('-t', dest='threshold',
                        action='store', type=int, default=0,
                        help='windows above this value will be reduced to the mean value')
    parser.add_argument('-v', dest='verbosity',
                        action='store', type=str, default='i',
                        help=
                        'level of logging: c (critical), e (error), '
                        'w (warning), i (info), d (debug). '
                        'Defaults to i.')
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    verbosity_dict = {'c': logging.CRITICAL,
                      'e': logging.ERROR,
                      'w': logging.WARNING,
                      'i': logging.INFO,
                      'd': logging.DEBUG}
    logging_level = verbosity_dict[arguments.verbosity]
    log_name = arguments.output_prefix + ".log"
    if log_name == '.log':
        log_name = "log"
    logging.basicConfig(filename=log_name,
                        filemode='w',
                        level=logging_level,
                        format='%(levelname)s\t%(asctime)s\t%(message)s',
                        datefmt="%d.%m.%Y %H:%M:%S")
    stdout_logger = logging.getLogger('STDOUT')
    stream_to_logger = StreamToLogger(stdout_logger, logging.DEBUG)
    sys.stdout = stream_to_logger
    stderr_logger = logging.getLogger('STDERR')
    stream_to_logger = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = stream_to_logger
    logging.info("Command used: %s", ' '.join(sys.argv))
    logging.info("Creating data structure...")
    data = Data(number_of_states=arguments.number_of_states, distr=arguments.distribution)
    logging.info("Reading in data...")
    for infile in arguments.infiles:
        data.add_data_from_bedgraph(infile)
    if arguments.bed_file:
        data.add_data_from_bed(arguments.bed_file, arguments.bed_mode)
    if arguments.threshold != 0:
        logging.info("Filtering data (removing outliers)")
        data.filter_data(arguments.threshold)
    logging.debug("Chromosome names: %s", str(data.chromosome_names))
    logging.debug("Chromosome lengths: %s", str(data.chromosome_lengths))
    logging.info("Data ready to analyse. Finding peaks")
    states = data.predict_states()
    data.save_states_to_file(states, arguments.output_prefix)
    data.write_stats_to_file(arguments.output_prefix)
    data.save_peaks_to_file(arguments.output_prefix)
    logging.info("...done.")

if __name__ == '__main__':
    main()
