from hw2skeleton import cluster
from hw2skeleton import io
import os
import numpy as np

def test_similarity():
    filename_a = os.path.join("data", "276.pdb")
    filename_b = os.path.join("data", "4629.pdb")

    activesite_a = io.read_active_site(filename_a)
    activesite_b = io.read_active_site(filename_b)
    
    assert cluster.compute_similarity(activesite_a, activesite_b) == float("inf")
    assert round(cluster.compute_similarity(activesite_a, activesite_a),1) == 0.0

def test_partition_clustering():
    # tractable subset
    pdb_ids = [276, 4629, 10701]

    active_sites = []
    for id in pdb_ids:
        filepath = os.path.join("data", "%i.pdb"%id)
        active_sites.append(io.read_active_site(filepath))

    clusters = cluster.cluster_by_partitioning(active_sites,1)[0]
    assert [int(c.name) for c in clusters] == [276, 4629, 10701]

def test_hierarchical_clustering():
    # tractable subset
    pdb_ids = [276, 4629, 10701]

    active_sites = []
    for id in pdb_ids:
        filepath = os.path.join("data", "%i.pdb"%id)
        active_sites.append(io.read_active_site(filepath))
    
    clusters = []
    for clust in cluster.cluster_hierarchically(active_sites,1):
        elems = []
        for e in clust:
            elems.append([int(n.name) for n in e])
        clusters.append(elems)
    
    assert clusters == [[[276], [4629], [10701]], [[276], [10701, 4629]], [[276, 10701, 4629]]]
