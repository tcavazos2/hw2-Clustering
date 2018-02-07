# File containing helper functions for computing similarity and clustering PDB active sites

import numpy as np 
import random
import matplotlib.pyplot as plt

def output_aa_string(residues):
    """
    Convert 3 letter amino acid code to single letter amino acids
    Input: list of residues in three letter codes
    Output: string of residues in single code
    """
    # Dictionary of 3 letter to 1 letter AA conversion
    aa_dict = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

    s = ''
    for res in residues:
        s = s + aa_dict.get(res.type)
    return s

def match(a1, a2):
    """
    Find indexes of matches for both alignment a and alignment b
    Input: two strings after alignment
    Outputs: indexes where match occurs in string 1 and string 2
    """
    inds_1, inds_2 = [], []
    i, j = 0, 0
    for s in range(0, len(a1)):
        increment = lambda a, x, ind: x+1 if a[ind] != "-" else x
        if a1[s] is a2[s] and a1[s] != "-":
            inds_1.append(i)
            inds_2.append(j)
        i = increment(a1,i,s)
        j = increment(a2,j,s)
    return inds_1, inds_2

def create_coord_matrix(site, index):
    """
    Create a matrix of CA atom x,y,z coordinates for every residue contained
    in an active site
    Inputs: active site and list of indices where two alignments match 
    Output: matrix with x,y, and z coordinates for every CA atom 
    """
    mat = np.empty([0,3])
    for res in np.take(site.residues, index):
        ind = [i for i,x in enumerate(res.atoms) if x.type == "CA" or x.type == "CA A"][0]
        x,y,z = res.atoms[ind].coords
        mat = np.append(mat, [[x,y,z]], axis=0)
    return mat

def initialize_k_mediods(data, k):
    """
    Randomly select k points and set as cluster centers
    Input: similarity matrix and # of clusters
    Output: centers for clusters
    """
    return random.sample(range(len(data)), k) 

def assign_k_clusters(data, centers):
    """
    Assign each data point to its nearest cluster center
    Input: matrix and k centers
    Output: cluster assignments for points in matrix
    """
    clusters = []
    center_data = np.take(data, centers, axis=0)
    best_center = np.argmax(center_data, axis=0)
    for i in range(len(centers)):
        inds = [ind for ind in np.where(best_center == i)[0]]
        clusters.append(inds)
    return clusters

def calculate_cost(data, centers, clusters):
    """
    Calculate the sum of similarities of cluster elements to centers
    """
    total = 0
    for i in range(len(centers)):
        total = total + np.sum(data[centers[i]][clusters[i]]) 
    return total

def recalculate_centers(data, k, clusters):
    """
    Recalculate cluster centers for data
    Input: matrix, number of clusters, and cluster assignments
    Output: new cluster centers
    """
    centers = []
    for k_i in range(k):
        inds = [i for i, j in enumerate(clusters) if j == k_i]
        n = np.take(data, inds, axis=0)
        if len(inds) == 0:
            i = np.random.randint(len(data))
            centers.append((data[i,0], data[i,1]))

        elif len(inds) < 2: 
            centers.append((n[0][0], n[0][1]))
        else:
            result = np.sum(n, axis=1)/len(inds)
            centers.append((result[0], result[0]))
    return centers

def dist_HC(active_sites, clusters,c_new, data):
    """
    Output the distance of the new cluster to all other clusters
    by computing the average distance 
    """
    new_arr = np.array([])
    for i in range(c_new):
        if type(clusters.get(i)) == type(None):
            new_arr = np.append(new_arr,float("-inf"))
        else:
            c = clusters.get(i)
            sub = data[np.ix_(clusters.get(c_new), c)]
            d = (np.ma.masked_invalid(sub).sum())/(len(clusters.get(c_new))*len(c))
            new_arr = np.append(new_arr, d)
    return new_arr

def output_cluster_list(active_sites, clusters):
    """
    Output the active site names in list of atoms format
    """
    atoms = []
    for c in clusters:
        clust = []
        for elem in c:
            clust.append(active_sites[elem])
        atoms.append(clust)
    return atoms
