from .utils import Atom, Residue, ActiveSite
import matplotlib.pyplot as plt
import numpy as np
from .helpers import *
from Bio import pairwise2
import rmsd
from sklearn.decomposition import PCA
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

def compute_similarity(site_a, site_b):
    """
    Compute the similarity between two given ActiveSite instances.

    Input: two ActiveSite instances
    Output: the similarity between them (a floating point number)
    """
    # Get strings of single letter aa residues
    s_a = output_aa_string(site_a.residues)
    s_b = output_aa_string(site_b.residues)
    
    # Align strings using local alignment algorithm which relies
    # on dynamic programming to compute all possible alignments and
    # returns the highest scoring alignment. 
    
    # Local alignment aims to find the max alignment for substrings
    # of two larger strings.
    # Matches = +1
    # Mismatches, gaps = +0
    
    alignments = pairwise2.align.localxx(s_a, s_b) # perform alignment
    if len(alignments) == 0: return float("inf") # return INF if no alignment found
    align_a, align_b, s = alignments[0][:3] # extract first alignment
        
    # Output indices where nucleotides in alignment match
    inds_a, inds_b = match(align_a, align_b)
    
    if len(inds_a) < 2: return float("inf")
        
    # Create matrix of coordinates for atom CA
    V = create_coord_matrix(site_a, inds_a)
    W = create_coord_matrix(site_b, inds_b)
     
    # Center and rotate Ca matrices then calculate Root-Mean-Square-Deviation (RMSD)
    # It measures the average distance between backbone atoms of two
    # superimposed proteins.

    # The greater the RMSD, the less similar the proteins are.
    # A RMSD equal to 0 represents identical proteins.

    # Each protein is a matrix containing x, y, and z coordinates for each CA atom
    # The rows of the two matrices are matching residues obtained from the alignment

    # To minimize RMSD you must first center the coordinates on the origin so the
    # two vectors can be near each other.
    V -= rmsd.centroid(V)
    W -= rmsd.centroid(W)

    # Then find the optimal rotation for matrix W that aligns it best with V
    # This is the Kabasch algorithm which works by calculating a covariance matrix
    # and then finding the singular value decomposition (SVD) of the cov. matrix
    # Last, find the optimal rotation matrix which is the dot product of V and W
    # optimized by lowest RMSD
    return rmsd.kabsch_rmsd(V,W)

def output_similarity_matrix(active_sites):
    """
    Calculate RMSD for all pairwise active sites. This distance measure
    is converted into a similarity metric by dividing by the max element and
    subtracting 1

    Input: list of active sites from PDB files
    Output: similarity matrix for active sites
    """
    # Create empty pairwise matrix 
    mat = np.empty([len(active_sites), len(active_sites)])
    # For every pair calculate the RMSD
    for (x,y), value in np.ndenumerate(mat):
        mat[x][y] = compute_similarity(active_sites[x], active_sites[y])
    # Infinite values means proteins had less than 3 similar amino acids, set to none
    mat[np.isinf(mat)] = None
    # Find max value in array for normalization
    max_val = np.nanmax(mat)
    # Make none values max value
    mat[np.isnan(mat)] = max_val
    # Get normalized dissimilarity matrix
    norm_mat = mat/max_val
    # Convert dissimilarity matrix to similarity by subtracting 1
    norm_mat_sim = 1 - norm_mat
    return norm_mat_sim

def cluster_by_partitioning(active_sites,k):
    """
    Cluster a given set of ActiveSite instances using a partitioning method.

    Input: a list of ActiveSite instances
    Output: a clustering of ActiveSite instances
            (this is really a list of clusters, each of which is list of
            ActiveSite instances)
    """
    cost_max = float("-inf")
    mat = output_similarity_matrix(active_sites)
    
    # randomly choose k medoids
    centers = initialize_k_mediods(mat, k)
    # assign elements to cluster medoid with max similarity
    clusters = assign_k_clusters(mat, centers)
    # calculate cost of clustering (sum of similarity of points to cluster)
    cost = calculate_cost(mat, centers, clusters)
    # iterate until cost does not increase
    while cost_max < cost:
        cost_max = cost
        # Loop through medoids and all elements not in medoids
        for i in range(0, len(centers)):
            m = centers[i]
            for o in range(len(active_sites)):
                if o != m:
                    # replace medoid with element and re-calculate clusters
                    # and cost
                    centers[i] = o
                    clusters_temp = assign_k_clusters(mat, centers)
                    cost_swap = calculate_cost(mat, centers, clusters_temp)
                    # if cost increases then replace clusters
                    if cost_swap > cost: 
                        cost = cost_swap
                        clusters = clusters_temp
                    # if cost decreases or stays the same leave center
                    else: centers[i] = m
    return output_cluster_list(active_sites, clusters)

def cluster_hierarchically(active_sites,k):
    """
    Cluster the given set of ActiveSite instances using a hierarchical algorithm.                                                                  #

    Input: a list of ActiveSite instances
    Output: a list of clusterings
            (each clustering is a list of lists of Sequence objects)
    """
    # Create similarity matrix
    mat_original = output_similarity_matrix(active_sites)
    mat = output_similarity_matrix(active_sites)
    # Fill diagonals with -infinity 
    np.fill_diagonal(mat, float("-inf"))
    
    # Create cluster array to keep track of number of clusters
    vals = [np.array([v]) for v in range(len(active_sites))]
    keys = np.arange(0,len(active_sites))
    clusters = dict(zip(keys, vals))
    all_clusters = []

    all_clusters.append(output_cluster_list(active_sites, clusters.values()))
    # Group the most similar elements until you only have one more cluster
    while len(clusters) > k:
        # Get most similar clusters
        i,j = np.unravel_index(mat.argmax(), mat.shape)
        # Get two clusters
        c_i = clusters.get(i)
        c_j = clusters.get(j)
        # Add new combined cluster
        c_new = list(clusters.keys())[-1]+1
        clusters[c_new] = np.append(c_i, c_j)
        
        # Add new row/column to similarity matrix
        new_dist = dist_HC(active_sites, clusters,c_new, mat_original)
        new_col = np.append(new_dist, float("-inf"))
        mat = np.vstack([mat, new_dist])
        mat = np.column_stack([mat, new_col])
        # Replace row/column with negative infinitys that correspond to 
        # most similar elements
        mat[i], mat[j] = float("-inf"), float("-inf")
        mat[:,j], mat[:,i] = float("-inf"), float("-inf")
        # Drop most similar elements from cluster
        clusters.pop(i)
        clusters.pop(j)
        all_clusters.append(output_cluster_list(active_sites, clusters.values()))
    return all_clusters
