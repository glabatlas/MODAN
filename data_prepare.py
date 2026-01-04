import numpy as np
import pandas as pd
import networkx as nx
import os
from utils import *
import shutil

for cancer in cancers:
    ppi = pd.read_table('./datasets/ppi_network.txt',header=None)
    
    data = list()
    data.append(pd.read_csv('./datasets/' + cancer + '/' + cancer + '_mut.csv',header=0,index_col=0))
    data.append(pd.read_csv('./datasets/' + cancer + '/' + cancer + '_methy.csv',header=0,index_col=0))
    data.append(pd.read_csv('./datasets/' + cancer + '/' + cancer + '_expr.csv',header=0,index_col=0))
    
    data[0] = data[0][data[0].mean(axis=1)>=mut_thres[cancer]]

    data[1] = data[1][data[1].index.isin(data[0].index)]
    data[2] = data[2][data[2].index.isin(data[1].index)]
    ppi = ppi[ppi[0].isin(data[2].index)&ppi[1].isin(data[2].index)]
    
    G = nx.Graph()
    G.add_edges_from(ppi.values)
    
    ppi_genes = sorted(nx.connected_components(G), key=len, reverse=True)[0]
    ppi_genes = np.array(list(ppi_genes))
    ppi_genes = ppi_genes.astype(str)
    
    data[0] = data[0][data[0].index.isin(ppi_genes)]
    data[1] = data[1][data[1].index.isin(ppi_genes)]
    data[2] = data[2][data[2].index.isin(ppi_genes)]
    data[0].sort_index(inplace=True)
    data[1].sort_index(inplace=True)
    data[2].sort_index(inplace=True)
    ppi = ppi[ppi.iloc[:,0].isin(ppi_genes)&ppi.iloc[:,1].isin(ppi_genes)]

    print(len(set(ppi.values[:,0].flatten())|set(ppi.values[:,1].flatten())),len(data[0]))
    
    if not os.path.exists('./gene_overlapped/'):
        os.makedirs('./gene_overlapped/')
    if not os.path.exists('./gene_overlapped/' + cancer + '/'):
        os.makedirs('./gene_overlapped/' + cancer + '/')

    if not os.path.exists('./gene_sample_overlapped/'):
        os.makedirs('./gene_sample_overlapped/')
    if not os.path.exists('./gene_sample_overlapped/' + cancer):
        os.makedirs('./gene_sample_overlapped/' + cancer)

    ppi.to_csv('./gene_sample_overlapped/' + cancer + '/ppi.csv',index=None,header=None)
    data[0].to_csv('./gene_overlapped/' + cancer + '/mut.csv')
    data[1].to_csv('./gene_overlapped/' + cancer + '/methy.csv')
    data[2].to_csv('./gene_overlapped/' + cancer + '/expr.csv')

#sample split
    subtype_label = pd.read_csv('./datasets/' + cancer + '/' + cancer + '_subtype_label.csv',index_col=None,header=None)
    subtype_label.dropna(inplace=True)
    
    data[1].columns = [col.replace('.', '-')[0:15] for col in data[1].columns]
    data[2].columns = [col.replace('.', '-')[0:15] for col in data[2].columns]

    subtype_data = list()
    for i in range(len(subtypes[cancer])):
        subtype_data.append(subtype_label[subtype_label.iloc[:,1]==subtypes[cancer][i]])
    
    for i in range(len(subtypes[cancer])):
        if not os.path.exists('./gene_overlapped/' + cancer + '/' + subtypes[cancer][i]):
            os.makedirs('./gene_overlapped/' + cancer + '/' + subtypes[cancer][i])
        data[0].iloc[:,data[0].columns.isin(subtype_data[i].iloc[:,0].values)].to_csv('./gene_overlapped/'+cancer+'/'+subtypes[cancer][i]+'/mut.csv')
        data[1].iloc[:,data[1].columns.isin(subtype_data[i].iloc[:,0].values)].to_csv('./gene_overlapped/'+cancer+'/'+subtypes[cancer][i]+'/methy.csv')
        data[2].iloc[:,data[2].columns.isin(subtype_data[i].iloc[:,0].values)].to_csv('./gene_overlapped/'+cancer+'/'+subtypes[cancer][i]+'/expr.csv')

#sample_split_overlap

for cancer in cancers:
    for i in range(len(subtypes[cancer])):
        data = list()
        data.append(pd.read_csv('./gene_overlapped/'+cancer+'/'+subtypes[cancer][i]+'/mut.csv',header=0,index_col=0))
        data.append(pd.read_csv('./gene_overlapped/'+cancer+'/'+subtypes[cancer][i]+'/methy.csv',header=0,index_col=0))
        data.append(pd.read_csv('./gene_overlapped/'+cancer+'/'+subtypes[cancer][i]+'/expr.csv',header=0,index_col=0))
        
        data[1] = data[1].iloc[:,data[1].columns.isin(data[0].columns)]
        data[2] = data[2].iloc[:,data[2].columns.isin(data[1].columns)]
        data[0] = data[0].iloc[:,data[0].columns.isin(data[2].columns)]
        data[1] = data[1].iloc[:,data[1].columns.isin(data[2].columns)]
    
        for j in range(len(data)):
            data[j] = data[j].sort_index(axis=1)
        print(data[0].shape,data[1].shape,data[2].shape)    


        if not os.path.exists('./gene_sample_overlapped/' + cancer + '/' + subtypes[cancer][i]):
            os.makedirs('./gene_sample_overlapped/' + cancer + '/' + subtypes[cancer][i])
        data[0].to_csv('./gene_sample_overlapped/' + cancer+'/' + subtypes[cancer][i] + '/mut.csv')
        data[1].to_csv('./gene_sample_overlapped/' + cancer+'/' + subtypes[cancer][i] + '/methy.csv')
        data[2].to_csv('./gene_sample_overlapped/' + cancer+'/' + subtypes[cancer][i] + '/expr.csv')