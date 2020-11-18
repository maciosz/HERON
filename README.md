# Hidden Markov Model peakcaller for NGS data

The program attempts to identify sites of enrichment
 in the signal from Next Generation Sequencing data;
 it takes coverages or bams of mapped reads as an input
 and returns bed file with coordinates of peaks,
 i.e. sites of presumed enrichment.

*Warning.*
 This is preliminary test version of beta version,
 still under development.
 Because of this, some features might not be fully tested or even implemented.
 In case of any issues, questions, suggestions or bugs please let me know:
 a.macioszek at mimuw.edu.pl.

## Requirements

 - python3 (tested on python3.6)
 - python packages:
   - numpy
   - scipy
   - sklearn
   - pysam

These packages can be installed using pip.

Make sure your default python is python3.
 If it's not and you want to keep it that way,
 either modify the first line of `peakcaller.py` and add `3` at the end:

```
#!/usr/bin/env python3
```

or run it always explicitly with `python3`, i.e. `python3 peakcaller.py`.

The peakcaller uses also python package `hmmlearn`, but it's modified in comparison to the original one,
 so there is no point in installing it;
 our version is included in the repositorium, in `hmmlearn` directory.
 If you already have `hmmlearn` package, that's ok;
 the peakcaller will use our version anyway.
 The license under which `hmmlearn` is distributed and the authors that contributed to it
 are available in the `hmmlearn` directory;
 we are using the version from 09.2019.
 You can check out the original package [here](https://github.com/hmmlearn/hmmlearn).

## Installation

No need to install.
 Just download the whole repository.
 The script you want to run is peakcaller.py.
 Do not move it from this directory;
 it needs to be in the same directory as the rest of the scripts
 and `hmmlearn` directory.
 You can add executive rights to it using command:

```
chmod +x peakcaller.py
```

Then you can run it with just `./peakcaller.py`, instead of `python peakcaller.py`.
 

## Simple usage

```
./peakcaller.py -i sample.bam
```
or
```
./peakcaller.py -i sample.bedgraph
```

possibly with

```
-o - prefix to output files

--control - control (input) files
```

For more possible arguments, go to [Advanced usage](https://github.com/maciosz/peakcaller#advanced-usage) section.

## Current output

Generates file `[prefix]_peaks.bed` with coordinates of predicted peaks in bed format.
 If you specify `--save-all-states` option
 it will also generate seperate file `[prefix]_state_[x].bed` for each state
 and one extra `[prefix]_all_states.bed` with state marked in the 3rd column.
 Also generates `[prefix].log` file with log messages
 and `[prefix]_stats.txt` with some statistics.

## How it works

If you only want to find peaks using default parameters you can probably skip this section.

Peakcaller uses Hidden Markov Models to discover sites of enrichment.
 Hidden Markov Model used here by default has three states.
 We assume each state emits signal (read coverage) from some distribution;
 currently Gaussian and negative binomial are supported.
 Provided data is divided into windows
 (with resolution either determined by bedgraph resolution,
 or specified by user via `-r` argument).
 We assume every window is in one of the three states,
 and the summaric (for negative binomial) or mean (for Gaussian) coverage
 in this window is the emitted value.

We estimate initial parameters of HMM in the following way:
 We calculate 0, 0.5 and 0.99 quantile of data and set them as initial means.
 Sample covariance matrix is set as initial covariance matrix.
 If distribution is set to negative binomial,
 parameters p and r are calculated from means and covariance.
 With this initial parameters, we can start training our HMM.
 We use Expectation Maximasation algorithm, for HMMs known as Baum-Welch algorithm.
 Briefly, the algorithms iteratively tries to improve our estimates of model's parameters
 (transition matrix, initial probabilities and parameters of distributions).
 
After fitting HMM to data, the states can be arranged according to their mean.
 In most cases (specifically always when we have only one sample)
 there is one state that has the highest mean among all samples
 and one with the lowest.
 Thus, sorted states can be interpreted as "no signal", "background / noise" and "peaks / enrichment".

In the last step, Viterbi algorithm is used to determine the most likely sequence of states;
 i.e. assign state to every window.
 Usually we are most interested in windows in "peak" state;
 coordinates of those are considered peaks.
 By default, only peaks are saved;
 you can choose however to save coordinates of all states.

You can use different number of states than 3,
 but we found that 3 gives usually the best results.
 You can also modify how means should be initialised;
 you can either provide constant means or different levels of quantiles. 

## Multiple files

You can provide multiple files as input,
 if you expect them to have more or less simmilar peaks.
 For example, if you have many replicates of the same experiment
 you can provide them all at once
 and I will return one set of peaks for all of them at once.
 There are two advantages of doing so:
 first, if you identify seperate set of peaks for every replicate
 chances are the sets will be slightly different
 and it won't be obvious how to determine a consensus set of peaks.
 Running peakcaller once and obtaining only one ready set of peaks
 removes this obstacle.
 Second, peaks with low enrichment can be hard to detect in a single sample;
 they can be intepreted as noise.
 However, if multiple samples have a weak peak in the same position
 the peakcaller can be more certain that there really is a peak in this position.
 It is not very likely that artificial peaks originating from noise
 will be highly repetitive between replicates.

You can also provide multiple sample that are divided into some groups:
 e.g. 3 samples from tissue A and 2 samples from tissue B,
 or samples from different grades of tumour.
 Then you can use `-g` parameter to tell peakcaller which sample is in which group.
 The peakcaller will attempt to identify
 set of peaks common for all the groups
 and sets of peaks unique for every group.
 But Achtung! This feature is not fully implemented yet.
 The peaks will be called, but you will have to interpret which state represent which set by yourself,
 basing on state means saved in `[prefix]_stats.txt` file.

## Advanced usage

Here is a list of all possible arguments.
 Note that the peakcaller is still in development,
 so some arguments are more for the developer rather than the user
 (like debugging options).
 Also some features are not yet fully supported or sufficiently tested.

#####  -i / --infiles

Input files. .bam and .bedgraph formats are supported.
 Bedgraphs should have fixed resolution and start from 0.
 If there are some floats in the bedgraph
 and distribution is set to Negative Binomial,
 program will convert them to integers
 (we don't mind floats with Gauss).
 You can provide multiple files here;
 see [Multiple files](https://github.com/maciosz/peakcaller#multiple-files) section.

##### -o / --output-prefix

Prefix to output files. Defaults to none.

##### --control

Control file(s), usually called input.
 Just like input files, can be either bam of bedgraph.
 Either provide one file and I will reuse it for all samples,
 or one for every sample.
 Currently normalisation on input means:
 for every window, coverage in the sample is divided by coverage in the control
 and logarithm is taken.

#####  -r / --resolution

Resolution to use. Ignored when input files are bedgraphs. Defaults to 600.

### You probably don't want to use these:

##### -d / --distribution

Distribution of emissions used in HMM; can be either Gaussian or negative binomial.
 Defaults to Gaussian.
 You can use following abbrevations: "G", "g", "Gauss", "normal"
 or "n", "N", "NB", "negativebinomial".
 Note that NB will probably be much slower than Gaussian.

##### -s / --number-of-states

Number of states in Hidden Markov Model. Defaults to 3 for simple peakcalling
 and 5 for peakcalling with samples divided into two groups.
 
##### -l / --logging
  
Level of logging: c (critical), e (error), w (warning), i (info) or d (debug).
 Defaults to i.

##### --dont-save

Should the state with highest mean be saved as peaks?
 By default it will. If you specify this, it won't.

##### --save-all-states

Should all states be saved?
 If you specify this and not `--dont-save`,
 the peak state would be saved twice.

##### --debug

If you want I can save all intermediate results:
 estimates of parameters in every iteration, posterior probabilities etc.
 Warning: it will (probably) be a lot of large files.
 Decide wisely.

##### --debug-prefix

If you chose `--debug` option, you can provide here some prefix for the result files.
 In particular, the prefix can contain desired path where I should save the results,
 e.g. `../intermediate_results/my_prefix`.
 If you don't specify this argument, I will create  directory `[output_prefix]_results`
 and I will save the results there, without any prefix.

##### -c / --covariance-type

Type of covariance matrix. Ignored when distribution is not Gaussian.
 Could be one of: diag (diagonal), full, spherical, tied, or grouped.
 "grouped" makes sense only if "-g" argument is provided;
 it means covariance matrix will be technically full,
 but initialised as full inside groups and filled with zeros between groups.
 Note that unlike other options, grouped option applies only to the initial covariance matrix, it can end up as totally full one.
 Defaults to full; if you choose grouped but don't provide "-g" it will also be full.

#### Means initialising options:

##### -m / --means 

Initial means. By default I will estimate them, but you can provide your own estimates here.
  When you have p samples and k states you should provide
 either k means (then I will use these means for all samples)
 or p * k means (first all the means for the first state, then for the second etc.).

##### -q / --quantiles

What quantiles should I use as no-signal, background and enrichment?
 Or as any other states, if you don't want 3 states.
 Defaults to 0,0.5,0.99 for 3 states or
 evenly spaced between 0 and 1 for any other number.

#### Not fully implemented yet:

##### -g / --groups

Are your samples divided into groups?
 E.g. tumor grades, tissues, different experiment types.
 If so, specify here the order of samples using 0s, 1s and so on.
 For example, -g 0 1 1 0 0 means that 1., 4. and 5. sample
 are from one group and 2. and 3. are from another.
 You can specify more than two groups.


