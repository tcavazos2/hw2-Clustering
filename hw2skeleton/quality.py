# Code to test cluster quality
from .cluster import *
from scipy.special import comb

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

def contingency_table(c1, c2):
    """
    Create contigency table to track similarities between two clusters
    """
    table = np.ndarray(shape=(len(c1), len(c2)))
    for i in range(0, len(c1)):
        c1_num = [c.name for c in c1[i]]
        for j in range(0, len(c2)):
            c2_num = [c.name for c in c2[j]]
            table[i][j] = len(np.intersect1d(c1_num, c2_num))
    return table

def adjusted_rand_index(c1, c2, n):
    """
    Measure to compare two clusterings by calculating:
    TP - # pairs in same cluster in both clusterings
    TN - # pairs in different clusters in both clusterings
    FP - # pairs in the same cluster in cluster 1 but different in cluster 2
    FN - # pairs in different clusters in cluster 1 but same in cluster 2
    """
    c1 = [x for x in c1 if x != []]
    table = contingency_table(c1,c2)
    index = np.sum(comb(table,2))
    max_index = (np.sum(comb(np.sum(table, axis=1),2))+np.sum(comb(np.sum(table, axis=0),2)))/2
    expected_index = (np.sum(comb(np.sum(table, axis=1),2))*np.sum(comb(np.sum(table, axis=0),2)))/comb(n,2)
    
    return (index-expected_index)/(max_index-expected_index)
