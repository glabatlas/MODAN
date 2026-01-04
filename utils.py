import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from functools import partial

## Dataset Config Begin
cancers = ['BRCA']

subtypes = {'BRCA':['Basal','Her2']}

mut_thres = {'BRCA': 0.02}
## Dataset Config End

def get_pair_exclusivity(gene1: np.ndarray, gene2: np.ndarray):
    if gene1.sum() + gene2.sum() == 0:
        mex_cov_sqrt = 0
    else:
        mex_cov_sqrt = (gene1 | gene2).sum() / ((gene1.sum() + gene2.sum()) * len(gene1))**0.5
    return mex_cov_sqrt

def get_single_module_exclusivity(data_mut: pd.DataFrame, data_methy: pd.DataFrame, data_expr: pd.DataFrame, module_index: list):
    mut_sel = data_mut.values[module_index]
    methy_sel = data_methy.values[module_index]
    expr_sel = data_expr.values[module_index]
    exclusivity = 0
    for i in range(len(mut_sel)):
        for j in range(i + 1, len(mut_sel)):
            exc_mut = get_pair_exclusivity(mut_sel[i,:], mut_sel[j,:])
            exc_methy = get_pair_exclusivity(methy_sel[i,:], methy_sel[j,:])
            exc_expr = get_pair_exclusivity(expr_sel[i,:], expr_sel[j,:])
            exclusivity += (exc_mut + exc_methy + exc_expr) / 3
    if len(mut_sel) > 1:
        exclusivity /= 0.5 * len(mut_sel) * (len(mut_sel) - 1)
    else:
        exclusivity = 0
    return exclusivity

def get_all_module_exclusivity(data_mut: pd.DataFrame, data_methy: pd.DataFrame, data_expr: pd.DataFrame, clustering_labels: np.ndarray):
    exclusivity = np.zeros((clustering_labels.max()+1))
    for c in range(clustering_labels.max()+1):
        exclusivity[c] = get_single_module_exclusivity(data_mut, data_methy, data_expr, np.where(clustering_labels == c))
    return exclusivity

def get_multi_dataset_exclusivity(data_mut: list, data_methy: list, data_expr: list, datasets: list, clustering_labels: np.ndarray):
    exclusivity = np.zeros((clustering_labels.max()+1,len(datasets)))
    for d in range(len(datasets)):
        exclusivity[:,d] = get_all_module_exclusivity(data_mut[d], data_methy[d], data_expr[d], clustering_labels)
    return exclusivity

def get_edge_weight(data: np.ndarray, ppi_idx: np.ndarray):
    genes_0 = data[ppi_idx[:,0]]
    genes_1 = data[ppi_idx[:,1]]
    mex = (genes_0 | genes_1).sum(axis=1) / (genes_0.sum(axis=1) + genes_1.sum(axis=1))
    mex = np.nan_to_num(mex, nan=0, posinf=0, neginf=0)
    cov = (genes_0 | genes_1).sum(axis=1) / genes_1.shape[1]
    gene_exclusivity = (mex*cov)**0.5
    return gene_exclusivity

def get_single_module_mut_exclusivity(data_mut: pd.DataFrame, module_index: list):
    mut_sel = data_mut.values[module_index]

    exclusivity = 0
    for i in range(len(mut_sel)):
        for j in range(i + 1, len(mut_sel)):
            exclusivity += get_pair_exclusivity(mut_sel[i,:], mut_sel[j,:])
    if len(mut_sel) > 1:
        exclusivity /= 0.5 * len(mut_sel) * (len(mut_sel) - 1)
    else:
        exclusivity = 0
    return exclusivity

def get_all_module_mut_exclusivity(data_mut: pd.DataFrame, clustering_labels: np.ndarray):
    exclusivity = np.zeros((clustering_labels.max()+1))
    for c in range(clustering_labels.max()+1):
        exclusivity[c] = get_single_module_mut_exclusivity(data_mut, np.where(clustering_labels == c))
    return exclusivity

def get_gaussian_distance(features: np.ndarray, ppi_idx: np.ndarray) -> np.ndarray:
    distance = squareform(pdist(features))
    gaussian_distance = 1 - np.exp(-distance**2/(10000))
    np.fill_diagonal(gaussian_distance, 0)
    return squareform(gaussian_distance)

def get_NI_qualified_genes(cancer: str, subtype: str):
    ppi = pd.read_csv('./gene_sample_overlapped/'+cancer+'/ppi.csv',header=None,index_col=None)

    data_mut_total = pd.read_csv('./gene_sample_overlapped/'+cancer+'/'+subtype+'/mut.csv',header=0,index_col=0)
    data_methy_total = pd.read_csv('./gene_sample_overlapped/'+cancer+'/'+subtype+'/0.05/methy_mut.csv',header=0,index_col=0)
    data_expr_total = pd.read_csv('./gene_sample_overlapped/'+cancer+'/'+subtype+'/0.05/expr_mut.csv',header=0,index_col=0)

    def idconv(idtable,idx):
        return idtable.loc[np.array(idx)].values.flatten()

    symbol_idx = pd.DataFrame(range(len(data_mut_total.index)),index=data_mut_total.index)
    symbol2idx = partial(idconv, symbol_idx)
    ppi_weighted = pd.DataFrame([symbol2idx(ppi.iloc[:,0]),symbol2idx(ppi.iloc[:,1])]).T.to_numpy()

    ppi_weight_mut = get_edge_weight(data_mut_total.values, ppi_weighted)
    ppi_weight_methy = get_edge_weight(data_methy_total.values, ppi_weighted)
    ppi_weight_expr = get_edge_weight(data_expr_total.values, ppi_weighted)
    ppi_weighted = np.append(ppi_weighted, ((ppi_weight_mut + ppi_weight_methy + ppi_weight_expr)/3).reshape(-1,1), axis=1)

    NI = pd.Series(0,index=data_mut_total.index,dtype=float)
    ppi_matrix_weighted = np.zeros((len(data_mut_total),len(data_mut_total)))
    ppi_matrix_weighted[ppi_weighted[:,0].astype(int),ppi_weighted[:,1].astype(int)] = ppi_weighted[:,2]
    ppi_matrix_weighted = np.maximum(ppi_matrix_weighted,ppi_matrix_weighted.T)

    for i in range(len(data_mut_total)):
        NI.iloc[i] = NIcal(i, ppi_matrix_weighted, ppi_weighted, data_mut_total.values)

    qualified_genes = NI.iloc[NI.values > 0].index.to_numpy()
    return qualified_genes

def NIcal(node_index, ppi_matrix_weighted, ppi_weighted, data_mut):
    neighbor_weight_sum = ppi_matrix_weighted[:,node_index].sum()
    node_neighbors = np.union1d(ppi_weighted[ppi_weighted[:,0]==node_index,1], ppi_weighted[ppi_weighted[:,1]==node_index,0])
    node_neighbors = np.union1d(node_neighbors, node_index).astype(int)
    cov = data_mut[node_index,:].sum()/data_mut.shape[1]
    NI = neighbor_weight_sum / (len(node_neighbors) - 1) * cov
    return NI

def get_module_node_influence(cancer: str, subtype: str, module_index: list):
    ppi = pd.read_csv('./gene_sample_overlapped/'+cancer+'/ppi.csv',header=None,index_col=None)

    data_mut_total = pd.read_csv('./gene_sample_overlapped/'+cancer+'/'+subtype+'/mut.csv',header=0,index_col=0)
    data_methy_total = pd.read_csv('./gene_sample_overlapped/'+cancer+'/'+subtype+'/0.05/methy_mut.csv',header=0,index_col=0)
    data_expr_total = pd.read_csv('./gene_sample_overlapped/'+cancer+'/'+subtype+'/0.05/expr_mut.csv',header=0,index_col=0)

    def idconv(idtable,idx):
        return idtable.loc[np.array(idx)].values.flatten()

    symbol_idx = pd.DataFrame(range(len(data_mut_total.index)),index=data_mut_total.index)
    symbol2idx = partial(idconv, symbol_idx)
    ppi_weighted = pd.DataFrame([symbol2idx(ppi.iloc[:,0]),symbol2idx(ppi.iloc[:,1])]).T.to_numpy()

    ppi_weight_mut = get_edge_weight(data_mut_total.values, ppi_weighted)
    ppi_weight_methy = get_edge_weight(data_methy_total.values, ppi_weighted)
    ppi_weight_expr = get_edge_weight(data_expr_total.values, ppi_weighted)
    ppi_weighted = np.append(ppi_weighted, ((ppi_weight_mut + ppi_weight_methy + ppi_weight_expr)/3).reshape(-1,1), axis=1)

    NI = np.zeros(len(module_index))
    ppi_matrix_weighted = np.zeros((len(data_mut_total),len(data_mut_total)))
    ppi_matrix_weighted[ppi_weighted[:,0].astype(int),ppi_weighted[:,1].astype(int)] = ppi_weighted[:,2]
    ppi_matrix_weighted = np.maximum(ppi_matrix_weighted,ppi_matrix_weighted.T)

    for i in range(len(module_index)):
        NI[i] = NIcal(symbol2idx(module_index[i]), ppi_matrix_weighted, ppi_weighted, data_mut_total.values)
    return NI

def get_all_module_influence(cancer: str, subtype: str, all_module_index: list):
    ppi = pd.read_csv('./gene_sample_overlapped/'+cancer+'/ppi.csv',header=None,index_col=None)

    data_mut_total = pd.read_csv('./gene_sample_overlapped/'+cancer+'/'+subtype+'/mut.csv',header=0,index_col=0)
    data_methy_total = pd.read_csv('./gene_sample_overlapped/'+cancer+'/'+subtype+'/0.05/methy_mut.csv',header=0,index_col=0)
    data_expr_total = pd.read_csv('./gene_sample_overlapped/'+cancer+'/'+subtype+'/0.05/expr_mut.csv',header=0,index_col=0)

    def idconv(idtable,idx):
        return idtable.loc[np.array(idx)].values.flatten()

    symbol_idx = pd.DataFrame(range(len(data_mut_total.index)),index=data_mut_total.index)
    symbol2idx = partial(idconv, symbol_idx)
    ppi_weighted = pd.DataFrame([symbol2idx(ppi.iloc[:,0]),symbol2idx(ppi.iloc[:,1])]).T.to_numpy()

    ppi_weight_mut = get_edge_weight(data_mut_total.values, ppi_weighted)
    ppi_weight_methy = get_edge_weight(data_methy_total.values, ppi_weighted)
    ppi_weight_expr = get_edge_weight(data_expr_total.values, ppi_weighted)
    ppi_weighted = np.append(ppi_weighted, ((ppi_weight_mut + ppi_weight_methy + ppi_weight_expr)/3).reshape(-1,1), axis=1)

    module_influence = np.zeros(len(all_module_index))
    ppi_matrix_weighted = np.zeros((len(data_mut_total),len(data_mut_total)))
    ppi_matrix_weighted[ppi_weighted[:,0].astype(int),ppi_weighted[:,1].astype(int)] = ppi_weighted[:,2]
    ppi_matrix_weighted = np.maximum(ppi_matrix_weighted,ppi_matrix_weighted.T)

    for i in range(len(module_influence)):
        module_influence[i] = MIcal(all_module_index[i], ppi_weighted, data_mut_total)
    
    return module_influence
    

def MIcal(module_index, ppi_weighted, data_mut):
    neighbor_edges_index = set()
    for i in range(len(module_index)):
        neighbor_edges_index = neighbor_edges_index.union(
            set(
                np.where((ppi_weighted[:,0] == module_index[i])|(ppi_weighted[:,1] == module_index[i]))[0].tolist()
                )
            )
    for i in range(len(module_index)):
        for j in range(len(module_index)):
            neighbor_edges_index = neighbor_edges_index - set(np.where((ppi_weighted[:,0] == module_index[i])&(ppi_weighted[:,1] == module_index[j]))[0].tolist())
    cov = (data_mut.values[module_index,:].sum(axis=1) > 0).sum()/data_mut.shape[1]
    MI = ppi_weighted[list(neighbor_edges_index), 2].sum()/len(neighbor_edges_index) * cov
    return MI