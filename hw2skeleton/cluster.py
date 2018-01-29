from .utils import Atom, Residue, ActiveSite
import numpy as np
from .helpers import *
from Bio import pairwise2
import rmsd

def compute_similarity(site_a, site_b):
    """
    Compute the similarity between two given ActiveSite instances.

    Input: two ActiveSite instances
    Output: the similarity between them (a floating point number)
    """
    similarity = 0.0
    # Get strings of single letter aa residues
    s_a = output_aa_string(site_a.residues)
    s_b = output_aa_string(site_b.residues)
    # Align strings
    alignments = pairwise2.align.localxx(s_a, s_b)
    align_a, align_b, s = alignments[0][:3]
        
    # Output indices where nucleotides in alignment match
    inds_a, inds_b = match(align_a, align_b)
    
    if len(inds_a) > 2:
        # Create matrix of coordinates for atom CA
        V = create_coord_matrix(site_a, inds_a)
        W = create_coord_matrix(site_b, inds_b)
     
        # Center and rotate Ca matrices then calculate RMSD
        V -= rmsd.centroid(V)
        W -= rmsd.centroid(W)
        similarity = rmsd.kabsch_rmsd(V,W)
    else: similarity = float("inf")

    return similarity

def cluster_by_partitioning(active_sites):
    """
    Cluster a given set of ActiveSite instances using a partitioning method.

    Input: a list of ActiveSite instances
    Output: a clustering of ActiveSite instances
            (this is really a list of clusters, each of which is list of
            ActiveSite instances)
    """
    # Fill in your code here!

    return []


def cluster_hierarchically(active_sites):
    """
    Cluster the given set of ActiveSite instances using a hierarchical algorithm.                                                                  #

    Input: a list of ActiveSite instances
    Output: a list of clusterings
            (each clustering is a list of lists of Sequence objects)
    """

    # Fill in your code here!

    return []
