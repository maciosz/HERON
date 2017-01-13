import peakcaller
import coverage

def test_readling_bedgraph():
    bedgraph = peakcaller.read_bedgraph_file('sample.bedgraph')
    assert isinstance(bedgraph, coverage.Coverage)
    assert len(bedgraph.lines) == 100
