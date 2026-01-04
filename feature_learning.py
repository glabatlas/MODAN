import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import argparse
import gensim

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cancer', type=str, help='Cancer type',required=True)
parser.add_argument('-s', '--subtype', type=str, help="Specific subtype", required=True)

args = parser.parse_args()

walk_length = 60

ppi = pd.read_csv('./exclusivity_network/' + args.cancer + '_' + args.subtype + '.csv',index_col=None,header=0)
ppi.columns = ['source', 'target', 'weight']

vector_size = 20

G = StellarGraph(edges=ppi)

rw = BiasedRandomWalk(G)

weighted_walks = rw.run(nodes=G.nodes(), length=walk_length, n=200, p=1, q=1, weighted=True, seed=42)

class Loss(gensim.models.callbacks.CallbackAny2Vec):
    def __init__(self):
        self.losses = []

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.losses.append(loss)
        print('Epoch {}, total loss {}'.format(len(self.losses), loss))

weighted_model = Word2Vec(weighted_walks, vector_size=vector_size, window=10, min_count=2, sg=1, hs=1, 
                          workers=1, alpha = 0.1, min_alpha = 0.05, epochs=3, compute_loss=True, callbacks=[Loss()], seed=4)

node_vectors = weighted_model.wv

if not os.path.exists('./models/'):
    os.makedirs('./models/')
weighted_model.save('models/'+  args.cancer +'_' + args.subtype + '.model')
out = np.zeros((len(node_vectors.index_to_key),vector_size))

i = 0
for node in sorted(node_vectors.index_to_key):
    out[i,:] = np.array(node_vectors[node])
    i = i + 1

if not os.path.exists('./results/'):
    os.makedirs('./results/')
if not os.path.exists('./results/features/'):
    os.makedirs('./results/features/')
np.savetxt('./results/features/'+args.cancer+'_'+args.subtype+'_features.csv', out, delimiter=',')
