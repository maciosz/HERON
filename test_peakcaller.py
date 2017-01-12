import peakcaller

def test_readling_bedgraph():
    bedgraph = peakcaller.read_bedgraph_file('sample.bedgraph')
    assert isinstance(bedgraph.read(), str)
