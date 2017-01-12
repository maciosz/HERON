#!/usr/bin/python3.4
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='infile', action='store',
                        help='input file')
    return parser.parse_args()

def read_bedgraph_file(filename):
    return open(filename)

def main():
    arguments = parse_arguments()
    infile = read_bedgraph_file(arguments.infile)

if __name__=='__main__':
    main()

