import os
import networkx as nx
import scipy.sparse as sp

def search_cycle(dir_adj):
    dir_adj = nx.from_scipy_sparse_matrix(A=dir_adj, create_using=nx.DiGraph)
    cycles = list(nx.algorithms.cycles.simple_cycles(dir_adj))
    num_cycle = len(cycles)
    q = []
    for i in range(num_cycle):
        q.append(len(cycles[i]))
    q = set(q)

    return q

dataset_path = './data'
dataset_name = 'cornell'
dataset_path = os.path.join(dataset_path, dataset_name)
dir_adj = sp.load_npz(os.path.join(dataset_path, 'adj.npz'))
dir_adj = dir_adj.tocsc()

q = search_cycle(dir_adj)
print(q)