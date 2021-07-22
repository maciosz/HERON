#!/usr/bin/env python3.6

import os
import sys
import logging
import argparse
import numpy as np
from model import Model

DISTRIBUTIONS = {'NB': ['n', 'nb', 'negativebinomial', 'negative_binomial', 'negative-binomial'],
                 'Gauss': ['g', 'gauss', 'normal', 'gaussian']}
DISTRIBUTIONS_REVERSE = {}
for key, values in DISTRIBUTIONS.items():
    for value in values:
        DISTRIBUTIONS_REVERSE[value] = key
COVARIANCE_TYPES = ['full', 'diag', 'spherical', 'tied']
POSSIBLE_SCORES = ['prob', 'median_prob', 'max_prob', 'mean_cov', 'max_cov', 'length']

class StreamToLogger():
    """
    Fake file-like stream object that redirects writes to a logger instance.
    source:
    https://www.electricmonk.nl/log/2011/08/14/redirect-stdout-and-stderr-to-a-logger-in-python/
    I'm not using it currently, right?
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
    parser.add_argument('-i', '--infiles',
                        action='store', type=str, nargs='+',
                        help='input files (currently bam and bedgraph formats are allowed)')
    parser.add_argument('-s', '--number-of-states',
                        action='store', type=int, default=3,
                        help='number of states (default: 3)')
    parser.add_argument('-o', '--output-prefix',
                        action='store', type=str, default='',
                        help='prefix to output files')
    parser.add_argument('-d', '--distribution',
                        action='store', type=str, default='Gauss',
                        help='distribution of emissions; "Gauss" (default) or "NB";'
                             ' you can also use "g", "G", "gaussian"'
                             ' or "n", "N", "negativebinomial".')
    parser.add_argument('--control', default=None, nargs='+',
                        help=
                        'control file(s), usually called input.'
                        ' Either provide one and I will reuse it for all samples,'
                        ' or one for every sample.')
    parser.add_argument('--merge', action='store_true', help=
                        'Ignored when you provide only one input file'
                        ' (or one input file and one control for it).'
                        ' Should all the input files be merged togheter'
                        ' (if so, choose this option)'
                        ' or treated seperatedly (that\'s the default)?'
                        ' See paper for details.')
    parser.add_argument('-t', '--threshold',
                        action='store', type=float, default=0,
                        help='t promils of windows with highest value'
                        ' will not be used to train the model.'
                        ' 0 means no threshold.')
    parser.add_argument('-l', '--logging',
                        action='store', type=str, default='i',
                        help=
                        'level of logging: c (critical), e (error), '
                        'w (warning), i (info), d (debug). '
                        'Defaults to i.')
    parser.add_argument('-r', '--resolution',
                        action='store', type=int, default=600,
                        help=
                        'Resolution to use. Ignored when input files are bedgraphs.'
                        'Defaults to 600.')
    parser.add_argument('--scores', default='mean_cov',
                        help=
                        'What should I save as peak score to the bed file?'
                        ' Possible options: prob, median_prob, max_prob, mean_cov, max_cov, length;'
                        ' prob stands for posterior probability, cov for coverage.'
                        ' See README for details. Defaults to mean_cov.')
    parser.add_argument('--dont-save-peaks', dest='save_peaks',
                        action='store_false',
                        help=
                        'Should the state with highest mean be saved as peaks?'
                        ' By default it will.'
                        ' If you specify this, it won\'t.')
    parser.add_argument('--save-all-states', action='store_true',
                        help=
                        'Should I save all the states'
                        ' (one file for all at once, and one seperate for each)?'
                        ' By default I won\'t.')
    parser.add_argument('-m', '--means', nargs='+', default=None, type=float,
                        help=
                        'Initial means. By default I will estimate them.'
                        ' When you have p samples and k states'
                        ' you should provide either k means'
                        ' (then I will use these means for all samples)'
                        ' or p * k means'
                        ' (first all the means for the first state,'
                        ' then for the second etc.).')
    #parser.add_argument('--random-seed', '--rs', default=None, type=int,
    #                    help=
    #                    'random seed for initialising means.'
    #                    ' Can be used to reproduce exact results.')
    parser.add_argument('-g', '--groups', nargs='+', type=int, default=None,
                        help=
                        'Are your samples divided into groups?'
                        ' If so, specify here the order of samples using 0s, 1s and so on.'
                        ' For example, -g 0 1 1 0 0 means that 1., 4. and 5. sample'
                        ' are from one group and 2. and 3. are from another.'
                        ' You can specify more than two groups.')
    parser.add_argument('-q', '--quantiles', nargs='+', type=float, default=None,
                        help=
                        'What quantiles should I use as no-signal, background and enrichment?'
                        ' Or as any other states, if you don\'t want 3 states.'
                        ' Defaults to 0 0.5 0.99 for 3 states'
                        ' or evenly spaced between 0 and 1 for any other number.')
                        #' I will always start from value zero anyway.')
    parser.add_argument('-c', '--covariance-type', default=None,
                        help=
                        'Type of covariance matrix.'
                        ' Ignored when distribution is not Gaussian.'
                        ' Could be one of:'
                        ' diag (diagonal), full, spherical, tied, or grouped.'
                        ' "grouped" makes sense only if "-g" argument is provided;'
                        ' it means covariance matrix will be full inside groups'
                        ' and filled with zeros between groups.'
                        ' Note that unlike other options,'
                        ' grouped option applies only to the initial covariance matrix,'
                        ' it can end up as full one.'
                        ' Defaults to diag; if you choose grouped but don\'t provide "-g"'
                        ' it will be full.')
    parser.add_argument('--debug', action='store_true',
                        help=
                        'If you want I can save all intermediate results.'
                        ' Warning: it will (probably) be a lot of large files. Decide wisely.')
    parser.add_argument('--debug-prefix', default=None,
                        help=
                        ' If you chose "--debug" option,'
                        ' you can provide here some prefix for the result files.'
                        ' In particular, the prefix can contain desired path where I should save the results,'
                        ' e.g. "../intermediate_results/my_prefix".'
                        ' If you don\'t specify this argument, I will create directory "[output_prefix]_results"'
                        ' and I will save the results there, without any prefix.')
    args = parser.parse_args()
    args = check_args(args)
    return args

def check_args(args):
    distribution = args.distribution.lower()
    distribution = DISTRIBUTIONS_REVERSE.get(distribution)
    if distribution is None:
        sys.exit("Unknown distribution: %s."
                 " Supported distributions are: %s."
                 " You can use following names: %s;"
                 " upper or lower case." %
                 (args.distribution,
                  str(list(DISTRIBUTIONS.keys())),
                  str(list(DISTRIBUTIONS_REVERSE.keys()))))
    args.distribution = distribution
    if args.distribution == "NB" and args.control is not None:
        raise ValueError("Control files are supported only for Gauss distribution. Sorry.")
    if args.quantiles is None:
        if args.number_of_states == 3:
            args.quantiles = [0, 0.5, 0.99]
        else:
            args.quantiles = np.linspace(0, 1, args.number_of_states)
    if args.debug:
        if args.debug_prefix is None:
            args.debug_prefix = args.output_prefix + "_results/"
            os.mkdir(args.debug_prefix)
        elif not args.debug_prefix.endswith("_"):
            # TODO: maybe prefix is simply a directory name?
            # Then we don't want _. To check.
            args.debug_prefix += "_"
    if args.scores not in POSSIBLE_SCORES:
        raise ValueError("Unknown --scores argument."
                         " I can recognise the following names: %s."
                         " You wanted %s. Is it on the list above?"
                         " No, it's not. So I don't recognise it." %
                         (POSSIBLE_SCORES, args.scores))
    return args

def get_covariance_type(args):
    if args.distribution == "NB" and args.covariance_type is not None:
        logging.info("Argument covariance type is ignored for Negative Binomial distribution. Just sayin.")
    if args.distribution == "Gauss" and args.covariance_type is None:
        args.covariance_type = 'diag'
    grouped = False
    if args.covariance_type == "grouped":
        grouped = True
        args.covariance_type = "full"
    if args.distribution == "Gauss" and args.covariance_type not in COVARIANCE_TYPES:
        sys.exit("Covariance type must be one of: %s. I don't understand your option: %s." %
                 (str(COVARIANCE_TYPES), args.covariance_type))
    return grouped

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
    # to fix, sth not working after 2.7 -> 3.6 transfer
    #stdout_logger = logging.getLogger('STDOUT')
    #stream_to_logger = StreamToLogger(stdout_logger, logging.INFO)
    #sys.stdout = stream_to_logger
    #stderr_logger = logging.getLogger('STDERR')
    #stream_to_logger = StreamToLogger(stderr_logger, logging.ERROR)
    #sys.stderr = stream_to_logger

    # this works, but error messages have additional newlines
    sys.stdout.write = logging.debug
    sys.stderr.write = logging.error

    # To teoretycznie powinno zadzialac jesli dobrze rozumiem dokumentacje
    # logging.captureWarnings(True)
    # ale nie dziala

    logging.info("Command used: %s", " ".join(sys.argv))
    logging.info("Creating data structure...")
    covariance_grouped = get_covariance_type(arguments)
    model = Model(number_of_states=arguments.number_of_states,
                  distribution=arguments.distribution,
                  #random_seed=arguments.random_seed,
                  random_seed=None,
                  debug_prefix=arguments.debug_prefix,
                  covariance_type=arguments.covariance_type)
    logging.debug("Random seed: %d", model.random_seed)
    logging.info("Reading in data...")
    model.read_in_files(arguments.infiles, resolution=arguments.resolution)
    logging.debug("Window size: %i", model.data.window_size)
    logging.debug("Chromosome names: %s", str(model.data.chromosome_names))
    logging.debug("Chromosome ends: %s", str(model.data.chromosome_ends))
    if arguments.distribution == "Gauss":
        logging.debug("Covariance type: %s", str(arguments.covariance_type))
    logging.info("All files read in.")
    logging.info("Preparing data for fitting.")
    if arguments.threshold != 0:
        logging.info("Filtering data...")
        model.filter_training_data(arguments.threshold)
    if arguments.control is not None:
        model.read_in_files(arguments.control, resolution=arguments.resolution,
                            add=True)
        model.normalise_data()
    if arguments.merge:
        logging.debug("Merging files...")
        model.merge_data()
    if arguments.groups:
        logging.debug("I will initialise grouped means")
        model.initialise_grouped_means(arguments.groups, arguments.quantiles)
        if covariance_grouped and arguments.distribution == "Gauss":
            logging.debug("I will initialise grouped covars")
            model.initialise_grouped_covars(arguments.groups)
    elif arguments.means:
        logging.info("Initialising given means...")
        model.initialise_constant_means(arguments.means)
    else:
        model.initialise_individual_means(arguments.quantiles)
    #logging.debug("Quantiles:")
    #logging.debug(arguments.quantiles)
    logging.info("Fitting model...")
    model.fit_HMM()
    logging.info("Predicting states...")
    model.predict_states()
    peaks = model.which_state_is_peaks()
    logging.info("Peaks: state %d", peaks)
    if arguments.save_peaks:
        model.score_peaks(peaks)
        model.save_state(arguments.output_prefix, peaks, "_peaks.bed",
                         save_score=True, which_score=arguments.scores)
        model.save_peaks_as_tab(arguments.output_prefix, peaks)
    if arguments.save_all_states:
        model.save_all_states(arguments.output_prefix)
    model.write_stats_to_file(arguments.output_prefix)
    # for debugging:
    #with open("%s_datamatrix.tab" % arguments.output_prefix, "w") as output:
    #    model.write_matrix_to_file(output)
    logging.info("...done.")

if __name__ == '__main__':
    main()
