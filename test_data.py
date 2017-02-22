from data import Data

def test_readling_bedgraphs():
    data = Data()
    for infile in ['sample1.bedgraph', 'sample2.bedgraph']:
        data.add_data_from_bedgraph(infile)
    assert len(data.matrix) == 2
    assert len(data.matrix[0]) == 100
    #assert Data.chromosome_names = []
    
def test_peak_calling():
    data = Data()
    data.add_data_from_bedgraph("simple_test.bedgraph")
    states = data.predict_states()
    correct_states = [[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                      [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]]
    #assert states.all(correct_states[0]) or states.all(correct_states[1])
    assert list(states) in correct_states

