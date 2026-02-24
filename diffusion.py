import pandas as pd
import numpy as np
from functools import partial
import os
import warnings
from utils import *

warnings.filterwarnings('ignore')

#influence integration
ratio = 0.05
mut_quantile = 0.998

def idconv(idtable,idx):
    return idtable.loc[np.array(idx)].values.flatten()

def diffusion_one_time(A, Pn, P, ratio):
    Pn = ratio*np.matmul(A,Pn)+(1-ratio)*P
    return Pn

def diffusion(cancer, subtypes):
    ppi = pd.read_csv('./datasets/'+cancer+'/ppi.csv',header=None)

    data = list()

    for subtype in subtypes:
        data = list()
        data.append(pd.read_csv('./datasets/'+cancer+'/'+subtype+'/mut.csv',header=0,index_col=0))
        data.append(pd.read_csv('./datasets/'+cancer+'/'+subtype+'/methy.csv',header=0,index_col=0))
        data.append(pd.read_csv('./datasets/'+cancer+'/'+subtype+'/expr.csv',header=0,index_col=0)) 
        data_normal = list()
        data_normal.append(pd.read_csv('./datasets/'+cancer+'/methy_normal.csv',header=0,index_col=0))
        data_normal.append(pd.read_csv('./datasets/'+cancer+'/expr_normal.csv',header=0,index_col=0))
        ppi_symbol_idx = pd.DataFrame(range(len(data[0].index.values)),index=data[0].index.values)
        symbol2idx = partial(idconv, ppi_symbol_idx)
        ppi_idx = pd.DataFrame([symbol2idx(ppi.iloc[:,0]),symbol2idx(ppi.iloc[:,1])]).T.to_numpy()
        for i in range(1,len(data)):
            P_final = np.zeros((data[0].shape[0],data[0].shape[1]))
            for j in range(data[i].shape[1]):
                node_weights = data[i].values[:,j]/data_normal[i-1].mean(axis=1)
                node_weights = np.abs(np.log2(node_weights))
                node_weights_num = np.nan_to_num(node_weights,copy=True,posinf=0,neginf=0)
                np.nan_to_num(node_weights,copy=False,posinf=max(node_weights_num),neginf=0)
                ppi_weights_0 = node_weights[ppi.iloc[:,0]].to_numpy()
                ppi_weights_1 = node_weights[ppi.iloc[:,1]].to_numpy()
                ppi_weights = pd.DataFrame([ppi_weights_0,ppi_weights_1]).T.mean(axis=1)
                ppi_weighted_net = np.zeros((len(data[i]),len(data[i])))
                np.put(ppi_weighted_net,ppi_idx[:,0]*len(data[0])+ppi_idx[:,1],ppi_weights.to_numpy())
                ppi_weighted_net /= np.max(ppi_weighted_net)
                P = data[0].iloc[:,j].values.reshape(-1,1)
                Pc = P
                Pn = diffusion_one_time(ppi_weighted_net,P,P,ratio)
                while np.linalg.norm(Pn-Pc) > 0.0001:
                    Pc = Pn
                    Pn = diffusion_one_time(ppi_weighted_net,Pn,P,ratio)
                P_final[:,j] = Pn.flatten()
                print(subtype+'_'+str(i)+':'+str(j)+'/'+str(data[i].shape[1]))
            P_final = pd.DataFrame(P_final,index=data[0].index,columns=data[0].columns)
            P_mut = np.zeros((data[0].shape[0],data[0].shape[1]))
            for k in range(P_final.shape[1]):
                Pn = P_final.values[:,k]
                if Pn[data[0].values[:,k]==0].sum() > 0:
                    Pn_over = (Pn>np.quantile(Pn[(Pn!=0).flatten()&(data[0].values[:,k]==0)],mut_quantile))
                    Pn_over = np.logical_or(Pn_over.flatten(),data[0].values[:,k]==1)
                    Pn_res = np.logical_not(Pn_over)
                    Pn[Pn_over] = 1
                    Pn[Pn_res] = 0
                else:
                    Pn = data[0].values[:,k]
                P_mut[:,k] = Pn.flatten()
            P_final = pd.DataFrame(P_final,index=data[0].index,columns=data[0].columns).astype(int)
            if not os.path.exists('./datasets_generate/'):
                os.makedirs('./datasets_generate/')
            if not os.path.exists('./datasets_generate/'+cancer+'/'):
                os.makedirs('./datasets_generate/'+cancer+'/')
            if not os.path.exists('./datasets_generate/'+cancer+'/'+subtype):
                os.makedirs('./datasets_generate/'+cancer+'/'+subtype)
            if i == 1:
                P_final.to_csv('./datasets_generate/'+cancer+'/'+subtype+'/methy_mut.csv')
            else:
                P_final.to_csv('./datasets_generate/'+cancer+'/'+subtype+'/expr_mut.csv')
