import sys
from .io import read_active_sites, write_clustering, write_mult_clusterings
from .cluster import cluster_by_partitioning, cluster_hierarchically, compute_similarity
from .quality import *

# Some quick stuff to make sure the program is called correctly
if len(sys.argv) < 4:
    print("Usage: python -m hw2skeleton [-P| -H] <pdb directory> <output file>")
    sys.exit(0)

active_sites = read_active_sites(sys.argv[2])

# Choose clustering algorithm
if sys.argv[1][0:2] == '-P':
    print("Clustering using Partitioning method")
    sil_score_out = open("sil_scores_P.txt", "w")
    for k in range(2,18):
        print(k)
        clustering = cluster_by_partitioning(active_sites,k)
        write_clustering(sys.argv[3]+"_"+str(k)+".txt", clustering)
        
        score = silhouette_score(clustering, active_sites)
        sil_score_out.write(str(k)+"\t"+str(score)+"\n")
    sil_score_out.close()

if sys.argv[1][0:2] == '-H':
    sil_score_out = open("sil_scores_H.txt", "w")
    print("Clustering using hierarchical method")
    clusterings = cluster_hierarchically(active_sites)
    write_mult_clusterings(sys.argv[3], clusterings)

    for i in range(0, len(clusterings)):
        score = silhouette_score(clusterings[i], active_sites)
        ind = len(clusterings)-i
        sil_score_out.write(str(ind)+"\t"+str(score)+"\n")
    sil_score_out.close()

