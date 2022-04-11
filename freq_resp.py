import os
import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigvalsh

def norm_feat(feature):
    feature = feature.astype(dtype=np.float64)
    if sp.issparse(feature):
        row_sum = feature.sum(axis=1).A1
        row_sum_inv = np.power(row_sum, -1)
        row_sum_inv[np.isinf(row_sum_inv)] = 0.
        deg_inv = sp.diags(row_sum_inv, format='csc')
        norm_feature = deg_inv.dot(feature)
    else:
        row_sum_inv = np.power(np.sum(feature, axis=1), -1)
        row_sum_inv[np.isinf(row_sum_inv)] = 0.
        deg_inv = np.diag(row_sum_inv)
        norm_feature = deg_inv.dot(feature)
        norm_feature = np.array(norm_feature, dtype=np.float64)

    return norm_feature

def load_webkb_data(dataset_name):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)

    feature = sp.load_npz(os.path.join(dataset_path, 'features.npz'))
    feature = feature.tocsc()
    n_feat = feature.shape[1]
    feature = norm_feat(feature)
    feature = feature.astype(dtype=np.float32)

    adj = sp.load_npz(os.path.join(dataset_path, 'adj.npz'))
    adj = adj.tocsc()

    label = np.genfromtxt(os.path.join(dataset_path, 'labels.csv'))
    n_class = 5

    idx_train = np.genfromtxt(os.path.join(dataset_path, 'idx_train.csv'))
    idx_valid = np.genfromtxt(os.path.join(dataset_path, 'idx_valid.csv'))
    idx_test = np.genfromtxt(os.path.join(dataset_path, 'idx_test.csv'))

    return feature, adj, label, idx_train, idx_valid, idx_test, n_feat, n_class

def calc_mag_gso(dir_adj, gso_type, q):
    if sp.issparse(dir_adj):
        id = sp.identity(dir_adj.shape[0], format='csc')
        # Symmetrizing an adjacency matrix
        adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)
        #adj = 0.5 * (dir_adj + dir_adj.transpose())
        
        if q != 0:
            dir = dir_adj.transpose() - dir_adj
            trs = np.exp(1j * 2 * np.pi * q * dir.toarray())
            trs = sp.csc_matrix(trs)
        else:
            trs = id # Fake
    
        if gso_type == 'sym_renorm_mag_adj' or gso_type == 'rw_renorm_mag_adj' \
            or gso_type == 'neg_sym_renorm_mag_adj' or gso_type == 'neg_rw_renorm_mag_adj' \
            or gso_type == 'sym_renorm_mag_lap' or gso_type == 'rw_renorm_mag_lap':
            adj = adj + id
    
        if gso_type == 'sym_norm_mag_adj' or gso_type == 'sym_renorm_mag_adj' \
            or  gso_type == 'neg_sym_norm_mag_adj' or gso_type == 'neg_sym_renorm_mag_adj' \
            or gso_type == 'sym_norm_mag_lap' or gso_type == 'sym_renorm_mag_lap':
            row_sum = adj.sum(axis=1).A1
            row_sum_inv_sqrt = np.power(row_sum, -0.5)
            row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
            deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')
            # A_{sym} = D^{-0.5} * A * D^{-0.5}
            sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

            if q == 0:
                sym_norm_mag_adj = sym_norm_adj
            elif q == 0.5:
                sym_norm_mag_adj = sym_norm_adj.multiply(trs.real)
            else:
                sym_norm_mag_adj = sym_norm_adj.multiply(trs)
            
            if gso_type == 'neg_sym_norm_mag_adj' or gso_type == 'neg_sym_renorm_mag_adj':
                gso = -1 * sym_norm_mag_adj
            elif gso_type == 'sym_norm_mag_lap' or gso_type == 'sym_renorm_mag_lap':
                sym_norm_mag_lap = id - sym_norm_mag_adj
                gso = sym_norm_mag_lap
            else:
                gso = sym_norm_mag_adj
        
        elif gso_type == 'rw_norm_mag_adj' or gso_type == 'rw_renorm_mag_adj' \
            or gso_type == 'neg_rw_norm_mag_adj' or gso_type == 'neg_rw_renorm_mag_adj' \
            or gso_type == 'rw_norm_mag_lap' or gso_type == 'rw_renorm_mag_lap':
            row_sum = adj.sum(axis=1).A1
            row_sum_inv = np.power(row_sum, -1)
            row_sum_inv[np.isinf(row_sum_inv)] = 0.
            deg_inv = sp.diags(row_sum_inv, format='csc')
            # A_{rw} = D^{-1} * A
            rw_norm_adj = deg_inv.dot(adj)

            if q == 0:
                rw_norm_mag_adj = rw_norm_adj
            elif q == 0.5:
                rw_norm_mag_adj = rw_norm_adj.multiply(trs.real)
            else:
                rw_norm_mag_adj = rw_norm_adj.multiply(trs)

            if gso_type == 'neg_rw_norm_mag_adj' or gso_type == 'neg_rw_renorm_mag_adj':
                gso = -1 * rw_norm_mag_adj
            elif gso_type == 'rw_norm_mag_lap' or gso_type == 'rw_renorm_mag_lap':
                rw_norm_mag_lap = id - rw_norm_mag_adj
                gso = rw_norm_mag_lap
            else:
                gso = rw_norm_mag_adj

        else:
            raise ValueError(f'{gso_type} is not defined.')
    
    else:
        id = np.identity(dir_adj.shape[0])
        # Symmetrizing an adjacency matrix
        adj = np.maximum(dir_adj, dir_adj.T)
        #adj = 0.5 * (dir_adj + dir_adj.T)

        if q != 0:
            dir = dir_adj.T - dir_adj
            trs = np.exp(1j * 2 * np.pi * q * dir)
        else:
            trs = id # Fake

        if gso_type == 'sym_renorm_mag_adj' or gso_type == 'rw_renorm_mag_adj' \
            or gso_type == 'sym_renorm_mag_lap' or gso_type == 'rw_renorm_mag_lap':
            adj = adj + id

        if gso_type == 'sym_norm_mag_adj' or gso_type == 'sym_renorm_mag_adj' \
            or gso_type == 'sym_norm_mag_lap' or gso_type == 'sym_renorm_mag_lap':
            row_sum = np.sum(adj, axis=1)
            row_sum_inv_sqrt = np.power(row_sum, -0.5)
            row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
            deg_inv_sqrt = np.diag(row_sum_inv_sqrt)
            # A_{sym} = D^{-0.5} * A * D^{-0.5}
            sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

            if q == 0:
                sym_norm_mag_adj = sym_norm_adj
            elif q == 0.5:
                sym_norm_mag_adj = np.multiply(sym_norm_adj, trs.real)
            else:
                sym_norm_mag_adj = np.multiply(sym_norm_adj, trs)

            if gso_type == 'sym_norm_mag_lap' or gso_type == 'sym_renorm_mag_lap':
                sym_norm_mag_lap = id - sym_norm_mag_adj
                gso = sym_norm_mag_lap
            else:
                gso = sym_norm_mag_adj

        elif gso_type == 'rw_norm_mag_adj' or gso_type == 'rw_renorm_mag_adj' \
            or gso_type == 'rw_norm_mag_lap' or gso_type == 'rw_renorm_mag_lap':
            row_sum = np.sum(adj, axis=1).A1
            row_sum_inv = np.power(row_sum, -1)
            row_sum_inv[np.isinf(row_sum_inv)] = 0.
            deg_inv = np.diag(row_sum_inv)
            # A_{rw} = D^{-1} * A
            rw_norm_adj = deg_inv.dot(adj)

            if q == 0:
                rw_norm_mag_adj = rw_norm_adj
            elif q == 0.5:
                rw_norm_mag_adj = np.multiply(rw_norm_adj, trs.real)
            else:
                rw_norm_mag_adj = np.multiply(rw_norm_adj, trs)

            if gso_type == 'rw_norm_mag_lap' or gso_type == 'rw_renorm_mag_lap':
                rw_norm_mag_lap = id - rw_norm_mag_adj
                gso = rw_norm_mag_lap
            else:
                gso = rw_norm_mag_adj

        else:
            raise ValueError(f'{gso_type} is not defined.')

    return gso

dataset = 'wisconsin'
gso_type = 'sym_renorm_mag_lap'
q = 1/6
feature, adj, label, idx_train, idx_val, idx_test, n_feat, n_class = load_webkb_data(dataset)
gso = calc_mag_gso(adj, gso_type, q)

if sp.issparse(gso):
    gso = gso.toarray()
    if q == 0 or q == 0.5:
        gso = gso.astype(np.float64)
    else:
        gso = gso.astype(np.complex128)
    eigval = eigvalsh(a=gso).real
    eigval = np.sort(eigval)
else:
    if q == 0 or q == 0.5:
        gso = gso.astype(np.float64)
    else:
        gso = gso.astype(np.complex128)
    eigval = eigvalsh(a=gso).real
    eigval = np.sort(eigval)

csv_name = dataset + '_sym_renorm_mag_lap_' + str(q) + '_ev.csv'
np.savetxt(fname=csv_name, X=eigval)