import pandas as pd
import numpy as np
from functools import partial
import networkx as nx
import warnings
import os
from utils import get_edge_weight

warnings.filterwarnings('ignore', category=RuntimeWarning)


def get_me_network(cancer, subtypes):
    ppi = pd.read_csv('./datasets/'+cancer+'/ppi.csv',header=None,index_col=None)

    data_mut = list()
    data_methy = list()
    data_expr = list()
    for s, subtype in enumerate(subtypes):
        data_mut.append(pd.read_csv('./datasets/'+cancer+'/'+subtype+'/mut.csv',header=0,index_col=0))
        data_methy.append(pd.read_csv('./datasets_generate/'+cancer+'/'+subtype+'/methy_mut.csv',header=0,index_col=0))
        data_expr.append(pd.read_csv('./datasets_generate/'+cancer+'/'+subtype+'/expr_mut.csv',header=0,index_col=0))

    def idconv(idtable,idx):
        return idtable.loc[np.array(idx)].values.flatten()

    ppi_symbol_idx = pd.DataFrame(range(len(data_mut[0].index)),index=data_mut[0].index)
    symbol2idx = partial(idconv, ppi_symbol_idx)
    ppi_idx = pd.DataFrame([symbol2idx(ppi.iloc[:,0]),symbol2idx(ppi.iloc[:,1])]).T.to_numpy()

    for s, subtype in enumerate(subtypes):
        data_mut[s] = data_mut[s].values
        data_methy[s] = data_methy[s].values
        data_expr[s] = data_expr[s].values

    edge_exclusivity = np.zeros((len(ppi_idx),len(subtypes)))
    for s, subtype in enumerate(subtypes):
        edge_exclusivity[:,s] = (get_edge_weight(data_mut[s], ppi_idx) + get_edge_weight(data_methy[s], ppi_idx) + get_edge_weight(data_expr[s], ppi_idx))/3

    for s, subtype in enumerate(subtypes):
        exclusivity_others = np.delete(edge_exclusivity, s, axis=1)
        exclusive_edges = (edge_exclusivity[:,s] > exclusivity_others.std(axis=1) + exclusivity_others.mean(axis=1)).astype(int)
        exclusivity_network = ppi.copy()
        exclusivity_network['Weight'] = edge_exclusivity[:,s]
        exclusivity_network = exclusivity_network.iloc[(exclusive_edges==1)]
        if len(exclusivity_network) > 0:
            G = nx.Graph()
            G.add_edges_from(exclusivity_network.values[:,0:2])
            exclusivity_network_genes = sorted(nx.connected_components(G), key=len, reverse=True)[0]
            exclusivity_network_genes = np.array(list(exclusivity_network_genes))
            exclusivity_network = exclusivity_network.iloc[np.isin(exclusivity_network.values[:,0],exclusivity_network_genes),:]
            exclusivity_network = exclusivity_network.iloc[np.isin(exclusivity_network.values[:,1],exclusivity_network_genes),:]
            print('cancer {} subtype {}: {} exclusive edge(s).'.format(cancer,subtype,len(exclusivity_network)))
            if not os.path.exists('./me_network/'):
                os.makedirs('./me_network/')
            exclusivity_network.to_csv('./me_network/' + cancer + '_' + subtype + '.csv',index=None)
        
        else:
            print('cancer {} subtype {}: 0 exclusive edge.'.format(cancer,subtype))
