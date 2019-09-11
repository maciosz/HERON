import numpy as np
from data import Data

def test_multiple_chromosomes():
    data = Data()
    data.add_data_from_files(["simple_test_with_multiple_chromosomes.bedgraph"])
    assert data.chromosome_names == ['chr1', 'chr2', 'chr3', 'chr4']
    assert data.chromosome_ends == [2300, 2900, 2500, 2300]
    assert data.numbers_of_windows == [23, 29, 25, 23]

def test_various_chromosome_ends():
    data = Data()
    data.add_data_from_files(["simple_test_with_various_chromosome_ends.bedgraph"])
    assert data.chromosome_names == ['chr1', 'chr2', 'chr3', 'chr4']
    assert data.chromosome_ends == [570, 940, 870, 504]
    assert data.numbers_of_windows == [6, 10, 9, 6]

def test_multiple_bedgraphs():
    data = Data()
    data.add_data_from_files(['1.bedgraph', '2.bedgraph'])
    assert data.matrix.shape[1] == 2

def test_filter():
    data = Data()
    data.add_data_from_files(["simple_test2.bedgraph"])
    data.filter_data([3])
    assert np.all(data.matrix == np.array([0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0]).reshape((11, 1)))
    data = Data()
    data.add_data_from_files(["simple_test2.bedgraph"])
    data.filter_data([1])
    assert np.all(data.matrix == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((11, 1)))

def test_split():
    data = Data()
    data.add_data_from_files(["simple_test2.bedgraph"])
    data.split_data([3])
    assert np.all(data.matrix == np.array([0, 0, 0, 2, 2, 2, 0, 0, 0]).reshape((9, 1)))
    assert data.chromosome_names == ['chr1_0']
    assert data.chromosome_ends == [900]
    assert data.numbers_of_windows == [9]
    data = Data()
    data.add_data_from_files(["simple_test2.bedgraph"])
    data.split_data([1])
    assert np.all(data.matrix == np.array([0, 0, 0, 0, 0, 0]).reshape((6, 1)))
    assert data.chromosome_names == ['chr1_0', 'chr1_3']
    assert data.chromosome_ends == [300, 300]
    assert data.numbers_of_windows == [3, 3]

def test_bam_reading():
    # TODO: finish me
    # testing values, mean...
    data = Data()
    data.add_data_from_files(["test.bam"], resolution=100)
    assert data.chromosome_names == ['chr21']
    assert data.chromosome_ends == [46709983]
    assert data.numbers_of_windows == [467100]

"""
def test_peak_calling():
    data = Data()
    data.add_data_from_bedgraph("simple_test.bedgraph")
    states = data.predict_states()
    correct_states = [[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                      [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]]
    #assert states.all(correct_states[0]) or states.all(correct_states[1])
    assert list(states) in correct_states
"""
"""
chr1	0	100	0
chr1	100	200	0
chr1	200	300	0
chr1	300	400	2
chr1	400	500	2
chr1	500	600	2
chr1	600	700	0
chr1	700	800	0
chr1	800	900	0
chr1	900	1000	4
chr1	1000	1100	4
"""
