import numpy as np 

def output_aa_string(residues):
    """
    Convert 3 letter amino acid code to single letter amino acids
    """
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
    Output index of string that matches for both alignment a and alignment b
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
    Create a matrix for an active sites CA x,y,z coordinates.
    """
    mat = np.empty([0,3])
    for res in np.take(site.residues, index):
        ind = [i for i,x in enumerate(res.atoms) if x.type == "CA"][0]
        x,y,z = res.atoms[ind].coords
        mat = np.append(mat, [[x,y,z]], axis=0)
    return mat

def tm_score(V, W, Ln):
    d0 = round((1.24*((Ln-15.0)**(1./3.)) - 1.8).real,1)
    di = np.sqrt(np.sum(np.power(np.subtract(V, W), 2), axis = 1))
    score = np.sum(1/(np.power(np.divide(di, d0), 2) + 1))/Ln
    return score


