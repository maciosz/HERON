Find peaks using HMM from hmmlearn.

Preliminary test version of beta version.

The program attempts to identify sites of enrichment
 in the signal from NGS data;
 it takes coverages or bams of mapped reads as an input
 and returns bed file with coordinates of peaks,
 i.e. sites of presumed enrichment.

### Installation:

No need to install. Just make sure this version of hmmlearn
 is in the same directory as peakcaller.
 Also you need numpy, scipy, pysam.

### Current usage:

./peakcaller.py -i sample1.bedgraph sample2.bedgraph ...

possibly with

-s - number of states (default 3)

-o - prefix to output files

-l - level of logging

-d - distribution of emissions (Gauss / NB; defaults to NB)

-t - threshold; t promils windows with highest values
     will be excluded from training.
     They will be included in finding states step.
     Defaults to 1. 0 mean no thresholding.

-r - resolution, that is the desired window size.
     Used only for reading bams, has no effect on bedgraphs.
     By default it's 200.

--dont-save - do not save one of the states as peaks

Initialising means options:

-m - list of constants

-q - quantiles

-g - your samples are divided into two groups;
     provide order of these groups here
     using 0s and 1s.


### Current output:

 generates
 file '[prefix]\_all\_states.bed' with all states,
 marked in the 3rd column and
 files [prefix]\_state\_[x].bed.
 Also, generates [prefix].log with log messages
 and [prefix]\_stats.txt with some statistics.
 Checks which state has the highest mean coverage over samples,
 assumes it denotes peaks and copies file [prefix]\_state\_x.bed to [prefix]\_peaks.bed.
 Unless you add --dont-save argument, then it doesn't.
 ...actually, it doesn't do that at the moment,
 so --dont-save has no effect.

Warning.
Assumes that bedgraphs 
 start with coordinate 0
 (bedgraphs are indeed 0-relative,
 but technically they might start from something else)
 (ok, to be fair it will manage with bedgraphs that start with multiplication of resolution,
 like 200; it will just add missing windows).
 (it won't now, because it would have to assume some resolution. I removed that useless feature.)
 Requires bedgraphs to have fixed resolution,
 exits otherwise.
 When using multiple bedgraphs,
 requires them to have identical coordinates;
 exits otherwise.
 (Actually, no exiting at the moment. I like to live dangerously.)
 If there are some floats in the bedgraph
 and distribution is set to Negative Binomial,
 program will convert them to integers
 (we don't mind floats with Gauss).
 Can read bams, but do it much longer than bedgraphs.

Tested on python 2.7. Won't work with python3.
(Actually, it might. Worth checking.)
