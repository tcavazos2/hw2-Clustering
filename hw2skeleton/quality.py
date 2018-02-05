# Code to test cluster quality
from .cluster import *

def silhouette_score(clusters_in, active_sites):
    """
    Calculate the silhouette score for a list of clusters.
    Measures how similar objects are to their own cluster and
    how different they are from others
    """
    mat = output_similarity_matrix(active_sites)
    all_sils = np.array([])
    
    clusters = [x for x in clusters_in if x != []]
    if len(clusters) == 1: return float("-inf")
    for clust in clusters:
        atom_inds = [active_sites.index(atom) for atom in clust]
        sub_same_clust = mat[np.ix_(atom_inds, atom_inds)]
        a_i = np.mean(sub_same_clust, axis=0)
        avgs = None
        
        for clust2 in clusters:
            if clust != clust2:
                atom_inds_other = [active_sites.index(atom2) for atom2 in clust2]
                sub_other_clust = mat[np.ix_(atom_inds, atom_inds_other)]
                sub_b = np.mean(sub_other_clust, axis=1)
                if type(avgs) == type(None): avgs = sub_b
                else: avgs = np.column_stack((avgs,sub_b))
        
        if avgs.ndim > 1:
            b_i = np.amax(avgs, axis=1)
        
        else: b_i = avgs
        s_i = (b_i - a_i)/np.amax(np.column_stack((a_i, b_i)),axis=1)
        all_sils = np.append(all_sils, s_i)
    return np.mean(all_sils)

def rand_index(clusters_1, clusters_2):
    """
    Measure to compare two clusterings by calculating:
    TP - # pairs in same cluster in both clusterings
    TN - # pairs in different clusters in both clusterings
    FP - # pairs in the same cluster in cluster 1 but different in cluster 2
    FN - # pairs in different clusters in cluster 1 but same in cluster 2
    """
    return 0
