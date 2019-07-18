#!/usr/bin/env python

import sys
import copy
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
                        help='input files (currently bam and bedgraph formats are allowed)')
    parser.add_argument('-s', dest='number_of_states',
                        action='store', type=int, default=3,
                        help='number of states (default: 3)')
    parser.add_argument('-o', dest='output_prefix',
                        action='store', type=str, default='',
                        help='prefix to output files')
    parser.add_argument('-d', dest='distribution',
                        action='store', type=str, default='NB',
                        help='distribution of emissions; "Gauss" or "NB" (default)')
    parser.add_argument('-t', dest='threshold',
                        action='store', type=float, default=0,
                        help='t promils of windows with highest value'
                        ' will not be used to train the model.'
                        ' 0 means no threshold.')
    #parser.add_argument('-t', dest='threshold',
    #                    action='store', type=int, default=0,
    #                    help='windows above this value will be considered outliers'
    #                    ' and reduced to the median value;'
    #                    ' 0 (default) means no threshold')
    parser.add_argument('-l', dest='logging',
                        action='store', type=str, default='i',
                        help=
                        'level of logging: c (critical), e (error), '
                        'w (warning), i (info), d (debug). '
                        'Defaults to i.')
    parser.add_argument('-r', dest='resolution',
                        action='store', type=int, default=200,
                        help=
                        'Resolution to use. Ignored when infiles are bedgraphs.')
    parser.add_argument('--dont-save', dest='save_peaks',
                        action='store_false',
                        help=
                        'Should the state with highest mean be saved as peaks?'
                        ' By default it will.'
                        ' If you specify this, it won\'t.')
    parser.add_argument('-m', '--means', nargs='+', default=None, type=float,
                        help=
                        'Initial means. By default I will estimate them.'
                        ' When you have p samples and k states'
                        ' you should provide either k means'
                        ' (then I will use these means for all samples)'
                        ' or p * k means'
                        ' (first all the means for the first state,'
                        ' then for the second etc.).')
    parser.add_argument('--random-seed', '--rs', default=None, type=int,
                        help=
                        'random seed for initialising means.'
                        ' Can be used to reproduce exact results.')
    parser.add_argument('-g', '--groups', nargs='+', type=int, default=None,
                        help=
                        'Are your samples divided into groups?'
                        ' If so, specify here the order of samples using 0s and 1s.'
                        ' For example, -g 0 1 1 0 0 means that 1., 4. and 5. sample'
                        ' are from one group and 2. and 3. are from another.'
                        ' Currently only 2 groups are allowed.')
    parser.add_argument('-q', '--quantiles', nargs='+', type=float, default=[0, 0.5, 0.99],
                        help=
                        'What quantiles should I use as background and enrichment?'
                        ' Or as any other states, if you want more than 3.')
                        #' I will always start from value zero anyway.')
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    verbosity_dict = {'c': logging.CRITICAL,
                      'e': logging.ERROR,
                      'w': logging.WARNING,
                      'i': logging.INFO,
                      'd': logging.DEBUG}
    logging_level = verbosity_dict[arguments.logging]
    log_name = arguments.output_prefix + ".log"
    if log_name == '.log':
        log_name = "log"
    logging.basicConfig(filename=log_name,
                        filemode='w',
                        level=logging_level,
                        format='%(levelname)s\t%(asctime)s\t%(message)s',
                        datefmt="%d.%m.%Y %H:%M:%S")
    stdout_logger = logging.getLogger('STDOUT')
    stream_to_logger = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = stream_to_logger
    stderr_logger = logging.getLogger('STDERR')
    stream_to_logger = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = stream_to_logger
    logging.info("Command used: %s", " ".join(sys.argv))
    logging.info("Creating data structure...")
    model = Model(number_of_states=arguments.number_of_states,
                  distribution=arguments.distribution,
                  random_seed=arguments.random_seed)
    logging.debug("Random seed: %d", model.random_seed)
    logging.info("Reading in data...")
    model.read_in_files(arguments.infiles, resolution=arguments.resolution)
    #if arguments.threshold != 0:
    #    logging.info("Filtering data (removing outliers)")
    #    model.filter_data(arguments.threshold)
    logging.debug("Window size: %i", model.data.window_size)
    logging.debug("Chromosome names: %s", str(model.data.chromosome_names))
    logging.debug("Chromosome ends: %s", str(model.data.chromosome_ends))
    #model.write_matrix_to_file(open(arguments.output_prefix + "matrix", "w"))
    #logging.info("Data ready to analyse. Finding peaks")
    logging.info("All files read in.")
    if arguments.groups:
        logging.debug("I will initialise grouped means")
        model.initialise_grouped_means(arguments.groups, arguments.quantiles)
    elif arguments.means:
        logging.info("Initialising means...")
        model.initialise_constant_means(arguments.means)
    else:
        model.initialise_individual_means(arguments.quantiles)
    model.data_for_training = copy.deepcopy(model.data)
    if arguments.threshold != 0:
        logging.info("Preparing data for fitting.")
        logging.info("Filtering data...")
        model.filter_training_data(arguments.threshold)
    # to prepair jest ni z gruszki ni z wiatraka
    # trzeba by pomyslec nad innym flowem tego wszystkiego
    model.prepair_data()
    logging.info("Fitting model...")
    model.fit_HMM()
    logging.info("Predicting states...")
    model.predict_states()
    model.save_states_to_seperate_files(arguments.output_prefix)
    model.write_stats_to_file(arguments.output_prefix)
    #if arguments.save_peaks:
    #    pass
    logging.info("...done.")

if __name__ == '__main__':
    main()
