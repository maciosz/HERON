#!/usr/bin/env python

import sys
import logging
import argparse
from model import Model

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
                        help='windows above this value will be considered outliers'
                        ' and reduced to the median value;'
                        ' 0 (default) means no threshold')
    parser.add_argument('-l', dest='verbosity',
                        action='store', type=str, default='i',
                        help=
                        'level of logging: c (critical), e (error), '
                        'w (warning), i (info), d (debug). '
                        'Defaults to i.')
    parser.add_argument('--dont-save', dest='save_peaks',
                        action='store_false',
                        help=
                        'Should the state with highest mean be saved as peaks?'
                        ' By default it will.'
                        ' If you specify this, it won\'t.')
    parser.add_argument('--n-peaks', dest = 'n_peaks',
                        action='store', type=float, default=0,
                        help=
                        'How many peaks do you expect,'
                        ' as the fraction of the whole genome? (E.g. 0.01)'
                        ' It will be used to initialise a transition matrix.'
                        ' By default model doesn\'t assume anything on that matter.')
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
    model = Model(number_of_states=arguments.number_of_states,
                  distribution=arguments.distribution)
    if arguments.n_peaks != 0:
        logging.info("Initialising transition matrix...")
        model.initialise_transition_matrix(arguments.n_peaks)
    logging.info("Reading in data...")
    model.read_in_files(arguments.infiles)
    if arguments.threshold != 0:
        logging.info("Filtering data (removing outliers)")
        model.filter_data(arguments.threshold)
    logging.debug("Window size: %i", model.data.window_size)
    logging.debug("Chromosome names: %s", str(model.data.chromosome_names))
    logging.debug("Chromosome ends: %s", str(model.data.chromosome_ends))
    logging.info("Data ready to analyse. Finding peaks")
    model.predict_states()
    model.save_states_to_seperate_files(arguments.output_prefix)
    model.write_stats_to_file(arguments.output_prefix)
    #if arguments.save_peaks:
    #    pass
    logging.info("...done.")

if __name__ == '__main__':
    main()
