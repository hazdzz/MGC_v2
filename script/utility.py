import math
import numpy as np
import scipy.sparse as sp
import torch

def calc_mag_gso(dir_adj, gso_type, q):
    if sp.issparse(dir_adj):
        id = sp.identity(dir_adj.shape[0], format='csc')
        # Symmetrizing an adjacency matrix
        adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)
        #adj = 0.5 * (dir_adj + dir_adj.transpose())
        
        if q != 0:
            dir = dir_adj - dir_adj.transpose()
            trs = np.exp(1j * 2 * np.pi * q * dir.toarray())
            trs = sp.csc_matrix(trs)
            if q == 0.25:
                trs = id + 1j * trs.imag
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

        if q != 0 and q != 0.25:
            dir = dir_adj - dir_adj.T
            trs = np.exp(1j * 2 * np.pi * q * dir)
            if q == 0.25:
                trs = id + 1j * trs.imag
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

def calc_mgc_feature(gso, feature, K, device):
    # LinearRank
    if device == torch.device('cpu'):
        if K == 1:
            emb = feature
        else:
            tmp = 2 / (K + 1) * feature
            emb = tmp # 0-th term
            for k in range(K-1):
                # (k+1)-th term
                tmp = (K - k - 1) / (K - k) * gso.dot(tmp)
                emb = emb + tmp
    else:
        if K == 1:
            emb = feature
        else:
            if gso.is_sparse:
                tmp = 2 / (K + 1) * feature
                emb = tmp # 0-th term
                for k in range(K-1):
                    # (k+1)-th term
                    tmp = (K - k - 1) / (K - k) * torch.sparse.mm(gso, tmp)
                    emb = torch.add(emb, tmp)
            else:
                tmp = 2 / (K + 1) * feature
                emb = tmp # 0-th term
                for k in range(K-1):
                    # (k+1)-th term
                    tmp = (K - k - 1) / (K - k) * torch.mm(gso, tmp)
                    emb = torch.add(emb, tmp)

    return emb

def cnv_sparse_mat_to_coo_tensor(sp_mat, dtype, device):
    # Convert a compressed sparse row (csr) or compressed sparse column (csc) matrix to a hybrid sparse coo tensor
    sp_coo_mat = sp_mat.tocoo()
    i = torch.from_numpy(np.vstack((sp_coo_mat.row, sp_coo_mat.col)))
    v = torch.from_numpy(sp_coo_mat.data)
    s = torch.Size(sp_coo_mat.shape)

    if dtype == np.complex64:
        return torch.sparse_coo_tensor(indices=i, values=v, size=s, dtype=torch.complex64, device=device, requires_grad=False)
    elif dtype == np.float32:
        return torch.sparse_coo_tensor(indices=i, values=v, size=s, dtype=torch.float32, device=device, requires_grad=False)
    else:
        raise TypeError(f'ERROR: The dtype of {sp_mat} is {sp_mat.dtype}, not been applied in implemented models.')

def calc_accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double().sum()
    accuracy = correct / len(labels)

    return accuracy