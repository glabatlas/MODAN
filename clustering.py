import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
import os
import warnings
import networkx as nx
from utils import *
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

warnings.filterwarnings('ignore', category=RuntimeWarning)

for cancer in cancers:
    for subtype in subtypes[cancer]:
        qualified_genes = get_NI_qualified_genes(cancer, subtype)
        exc_net = pd.read_csv('./exclusivity_network/' + cancer + '_' + subtype + '.csv',index_col=None,header=0)
        exc_net_genes = np.sort(np.union1d(exc_net.values[:,0],exc_net.values[:,1]))

        data_mut = pd.read_csv('./gene_sample_overlapped/'+cancer+'/'+subtype+'/mut.csv',header=0,index_col=0)
        data_methy = pd.read_csv('./gene_sample_overlapped/'+cancer+'/'+subtype+'/0.05/methy_mut.csv',header=0,index_col=0)
        data_expr = pd.read_csv('./gene_sample_overlapped/'+cancer+'/'+subtype+'/0.05/expr_mut.csv',header=0,index_col=0)

        data_mut = data_mut[data_mut.index.isin(exc_net_genes)].sort_index()
        data_methy = data_methy[data_methy.index.isin(exc_net_genes)].sort_index()
        data_expr = data_expr[data_expr.index.isin(exc_net_genes)].sort_index()


        out = np.loadtxt('./results/features/' + cancer + '_' + subtype + '_features.csv', delimiter=',')
        print(len(out))
        out = out[np.isin(data_mut.index, qualified_genes)]
        print(len(out))
        data_mut = data_mut[data_mut.index.isin(qualified_genes)].sort_index()
        data_methy = data_methy[data_methy.index.isin(qualified_genes)].sort_index()
        data_expr = data_expr[data_expr.index.isin(qualified_genes)].sort_index()

        ppi = pd.read_csv('./gene_sample_overlapped/'+cancer+'/ppi.csv',header=None,index_col=None)
        ppi = ppi[(ppi.iloc[:,0].isin(data_mut.index)&ppi.iloc[:,1].isin(data_mut.index)==1)]
        def idconv(idtable,idx):
            return idtable.loc[np.array(idx)].values.flatten()
        symbol_idx = pd.DataFrame(range(len(data_mut.index)),index=data_mut.index)
        symbol2idx = partial(idconv, symbol_idx)
        ppi_idx = pd.DataFrame([symbol2idx(ppi.iloc[:,0]),symbol2idx(ppi.iloc[:,1])]).T.to_numpy()

        nclu_start = round(len(out)/4/10)*10
        nclu_end = round(len(out)/2/10)*10

        distance_array = get_gaussian_distance(out, ppi_idx)
        Z = linkage(distance_array, method='ward')
        n_clus = np.arange(nclu_start, nclu_end+1, 10)
        si_clus = np.zeros(len(n_clus))
        ch_clus = np.zeros(len(n_clus))
        db_clus = np.zeros(len(n_clus))
        for i in range(len(n_clus)):
            clustering_labels = fcluster(Z,  t=n_clus[i], criterion='maxclust') - 1
            si_clus[i] = silhouette_score(out, clustering_labels)
            ch_clus[i] = calinski_harabasz_score(out, clustering_labels)
            db_clus[i] = davies_bouldin_score(out, clustering_labels)

        sc_clus = 0.65 * si_clus + 0.25 * np.log10(ch_clus) + 0.1 * np.log10(1/db_clus)
        n_cluster = n_clus[np.argmax(sc_clus)]
        
        print(n_cluster,si_clus[np.argmax(sc_clus)])
        clustering_labels = fcluster(Z, t=n_cluster, criterion='maxclust')
        gene_name_cluster = []
        for cluster_no in range(0,clustering_labels.max()+1):
            gene_name_cluster.append(data_mut.index[clustering_labels == cluster_no].to_list())
        max_cluster_length = max(len(cluster) for cluster in gene_name_cluster)

        cur_exclusivity = get_all_module_exclusivity(data_mut, data_methy, data_expr, clustering_labels)

        module_index_list = []
        for i in range(max(clustering_labels)+1):
            module_index_list.append(list(np.where(clustering_labels==i)[0]))
        cur_influence = get_all_module_influence(cancer, subtype, module_index_list)
        gene_name_cluster_padded = list()
        for cluster in gene_name_cluster:
            gene_name_cluster_padded.append(cluster + (max_cluster_length - len(cluster)) * [''])
        cluster_genes = pd.DataFrame(gene_name_cluster_padded)
        clustering_results = pd.concat([pd.DataFrame((cur_exclusivity).reshape((-1,1))),pd.DataFrame((cur_influence).reshape((-1,1))),cluster_genes],axis=1)
        clustering_results = clustering_results[clustering_results.iloc[:,3]!='']

        clustering_results.columns = [f'column_{i}' for i in range(clustering_results.shape[1])]
        clustering_results = clustering_results.sort_values(by=['column_0','column_1'],ascending=False)
        if not os.path.exists('./results/'):
            os.makedirs('./results/')
        if not os.path.exists('./results/modules/'):
            os.makedirs('./results/modules/')
        if not os.path.exists('./results/modules/' + cancer):
            os.makedirs('./results/modules/' + cancer)
        if not os.path.exists('./results/modules/' + cancer + '/' + subtype):
            os.makedirs('./results/modules/' + cancer + '/' + subtype)
        clustering_results.to_csv('./results/modules/'  + cancer + '/' + subtype + '/initial_results.csv',header=None,index=None)


for cancer in cancers:
    data_mut = list()
    data_methy = list()
    data_expr = list()
    for i in range(len(subtypes[cancer])):
        data_mut.append(pd.read_csv('./gene_sample_overlapped/'+cancer+'/'+subtypes[cancer][i]+'/mut.csv',header=0,index_col=0))
        data_methy.append(pd.read_csv('./gene_sample_overlapped/'+cancer+'/'+subtypes[cancer][i]+'/0.05/methy_mut.csv',header=0,index_col=0))
        data_expr.append(pd.read_csv('./gene_sample_overlapped/'+cancer+'/'+subtypes[cancer][i]+'/0.05/expr_mut.csv',header=0,index_col=0))
    symbol_idx = pd.DataFrame(range(len(data_mut[0].index)),index=data_mut[0].index)
    ppi = pd.read_csv('./gene_sample_overlapped/'+cancer+'/ppi.csv',header=None,index_col=None)
    def idconv(idtable,idx):
        return idtable.loc[np.array(idx)].values.flatten()
    symbol2idx = partial(idconv, symbol_idx)
    ppi_idx = pd.DataFrame([symbol2idx(ppi.iloc[:,0]),symbol2idx(ppi.iloc[:,1])]).T.to_numpy()
    res_table = list()
    for s in range(len(subtypes[cancer])):
        ppi_weighted = pd.DataFrame([symbol2idx(ppi.iloc[:,0]),symbol2idx(ppi.iloc[:,1])]).T.to_numpy()
        ppi_weight_mut = get_edge_weight(data_mut[s].values, ppi_weighted)
        ppi_weight_methy = get_edge_weight(data_methy[s].values, ppi_weighted)
        ppi_weight_expr = get_edge_weight(data_expr[s].values, ppi_weighted)
        ppi_weighted = np.append(ppi_weighted, ((ppi_weight_mut + ppi_weight_methy + ppi_weight_expr)/3).reshape(-1,1), axis=1)
        res_table.append(pd.read_csv('./results/modules/'+cancer+'/'+subtypes[cancer][s]+'/initial_results.csv',header=None))
        module_list = []
        for i in range(len(res_table[s])):
            module = res_table[s].values[i,2::]
            module_genes = []
            for ii in range(len(module)):
                if isinstance(module[ii],str) and module[ii] != '':
                    module_genes.append(module[ii])
            module_genes_idx = symbol2idx(np.array(module_genes))
            module_list.append(module_genes_idx)
        module_del_idx = []
        module_add = []
        while True:
            module_del_idx = []
            module_add = []
            for i in range(len(module_list)):
                current_module = module_list[i]
                current_exclusivity = get_single_module_exclusivity(data_mut[s], data_methy[s], data_expr[s], current_module)
                if len(current_module) >= 3:
                    new_weight = np.zeros(len(current_module))
                    for l in range(len(current_module)):
                        current_module_genes_idx_del = np.delete(current_module, l)
                        new_weight[l] = get_single_module_exclusivity(data_mut[s], data_methy[s], data_expr[s], current_module_genes_idx_del)
                    if ((new_weight > current_exclusivity).sum() > 0):
                        new_module_genes_idx = np.delete(current_module, np.argmax(new_weight))
                        G = nx.Graph()
                        current_module_ppi_idx = ppi_idx[(np.isin(ppi_idx[:,0],new_module_genes_idx))&(np.isin(ppi_idx[:,1],new_module_genes_idx))]
                        G.add_edges_from(current_module_ppi_idx)
                        new_added = 0
                        for component in list(nx.connected_components(G)):
                            if len(component) >= 2:
                                new_added += 1
                        if new_added == 0:
                            new_weight[np.argmax(new_weight)] = 0
                            continue
                        else:
                            for component in list(nx.connected_components(G)):
                                module_add.append(list(component))
                            module_del_idx.append(i)
                            break
                        del G
            module_del_idx = sorted(module_del_idx, reverse=True)
            for del_index in module_del_idx:
                del module_list[del_index]
            for single_module_add in module_add:
                module_list.append(single_module_add)
            if len(module_del_idx) == 0:
                break

        exclusivity_new = np.zeros(len(module_list))
        for i in range(len(module_list)):
            exclusivity_new[i] = get_single_module_exclusivity(data_mut[s], data_methy[s], data_expr[s], module_list[i])
        influence_new = get_all_module_influence(cancer, subtypes[cancer][s], module_list)
        module_genename_list = []
        for i in range(len(module_list)):
            module_genename_list.append(data_mut[s].index[module_list[i]].to_list())

        max_cluster_length = max(len(cluster) for cluster in module_genename_list)
        gene_name_cluster_padded = list()
        for cluster in module_genename_list:
            gene_name_cluster_padded.append(cluster + (max_cluster_length - len(cluster)) * [''])
        cluster_genes = pd.DataFrame(gene_name_cluster_padded)
        clustering_results = pd.concat([pd.DataFrame((exclusivity_new).reshape(-1,1)),pd.DataFrame((influence_new).reshape(-1,1)),cluster_genes],axis=1)
        clustering_results = clustering_results[clustering_results.iloc[:,3]!='']
        clustering_results.columns = [f'column_{i}' for i in range(clustering_results.shape[1])]
        clustering_results = clustering_results.sort_values(by=['column_0','column_1'],ascending=False)
        clustering_results.to_csv('./results/modules/' + cancer + '/' + subtypes[cancer][s] + '/final_results.csv',header=None,index=None)