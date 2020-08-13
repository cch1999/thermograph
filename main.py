from nn.nn import Predictor

import dgl
import numpy as np
import pandas as pd
import torch
import pickle
from tqdm import tqdm
from dgllife.model.model_zoo import GCNPredictor
from graphein.construct_graphs import ProteinGraph
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader
from dgl.data.utils import save_graphs, load_graphs, load_labels
from utils import collate

from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"

print(device)

# Load data sets
df = pd.read_csv('data/datasets/Q3214_direct.csv')
df.head()
#df = df.iloc[:50]
print(df)

# Create labels
labels = df["ddg"].tolist()
#print("printing labels")
#print(labels)
#labels = [torch.Tensor(i) for i in labels]
print(labels)
# Split datasets
x_train, x_test, y_train, y_test = train_test_split(df, labels, test_size=0.15)

print(x_train)

#----Initialise protein graphs----

pg = ProteinGraph(granularity='atom', insertions=False, keep_hets=True,
          node_featuriser='meiler',
          get_contacts_path='~/projects/repos/getcontacts',
          pdb_dir='data/pdbs',
          contacts_dir='files/contacts/',
          exclude_waters=True,
          covalent_bonds=True,
          include_ss=True,
          include_ligand=False,
          remove_string_labels=False)

#----Construst graphs----

train_graphs = []
test_graphs = []

i = 0
"""
for index, variant in tqdm(x_train.iterrows()):

    wt = variant["pdb_id"]
    mut = wt + "_" + variant["wild_type"] + str(variant["position"]) + variant["mutant"]

    print(mut)

    wt_path = "data/pdbs/Q3214_Q1744/" + wt + "/" + wt + "_relaxed.pdb"
    mut_path = "data/pdbs/Q3214_Q1744/" + wt + "/" + mut + "_relaxed.pdb"

    train_graphs.append([pg.dgl_graph_from_pdb_code(file_path=wt_path), pg.dgl_graph_from_pdb_code(file_path=mut_path)])
    i = i + 1
    print(i, "/", len(x_train))
    #print(train_graphs[-1][1].ndata)

i = 0

for index, variant in tqdm(x_test.iterrows()):

    wt = variant["pdb_id"]
    mut = wt + "_" + variant["wild_type"] + str(variant["position"]) + variant["mutant"]

    print(mut)

    wt_path = "data/pdbs/Q3214_Q1744/" + wt + "/" + wt + "_relaxed.pdb"
    mut_path = "data/pdbs/Q3214_Q1744/" + wt + "/" + mut + "_relaxed.pdb"

    test_graphs.append([pg.dgl_graph_from_pdb_code(file_path=wt_path), pg.dgl_graph_from_pdb_code(file_path=mut_path)])

    i = i + 1
    print(i, "/", len(x_train))

#Save graphs
#with open("train_graphs.p", "wb") as pickle_file:
#    pickle.dump(train_graphs, pickle_file)
with open("test_graphs.p", "wb") as pickle_file:
    pickle.dump(test_graphs, pickle_file)
"""
with open("train_graphs.p", "rb") as pickle_file:
    train_graphs = pickle.load(pickle_file)

print(train_graphs[1])
print(train_graphs[1][1].ndata)

with open("test_graphs.p", "rb") as pickle_file:
    test_graphs = pickle.load(pickle_file)

#----Create dataloaders----

train_data = list(zip(train_graphs, y_train))
test_data = list(zip(test_graphs, y_test))

train_loader = DataLoader(train_data, batch_size=32, shuffle=True,
                         collate_fn=collate)

test_loader = DataLoader(test_data, batch_size=32, shuffle=True,
                         collate_fn=collate)

#----Inititialise model----

in_feats = 74

gcn_net = Predictor(in_feats, hidden_feats=[32, 32, 32, 32], batchnorm=[True, True, True, True], dropout=[0, 0, 0, 0], predictor_hidden_feats=128, n_tasks=64)

gcn_net.to(device)
loss_fn = MSELoss()
optimizer = torch.optim.Adam(gcn_net.parameters(), lr=0.00005)

epochs = 50

# Training loop
gcn_net.train()
epoch_losses = []

epoch_f1_scores = []
epoch_precision_scores = []
epoch_recall_scores = []

for epoch in range(epochs):
    epoch_loss = 0

    preds = []
    labs = []
    # Train on batch
    for i, (bg, labels) in enumerate(train_loader):
        print(i)
        labels = labels.to(device)
        """
        #print(bg.ndata)
        #print(bg.ndata["h"].shape)
        print(bg.node_attr_schemes())
        graph_feats = torch.cat([bg.ndata.pop("h").to(device), bg.ndata.pop("ss").to(device)], dim=1)
        print(graph_feats.shape)
        print(graph_feats)
        #print(bg)
        """
        #print(bg[1].ndata)
        #print(bg[1])

        #graph_feats = torch.cat(features, dim=1)

        #graph_feats, labels = graph_feats.to(device), labels.to(device)
        y_pred = gcn_net(bg, [bg[0].ndata.pop("h").to(device), bg[1].ndata.pop("h").to(device)])


        preds.append(y_pred.detach().numpy())
        labs.append(labels.detach().numpy())

        labels = labels.reshape(len(labels), 1)

        #print(y_pred)
        #print(labels)

        loss = loss_fn(y_pred, labels)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()

    epoch_loss /= (i + 1)

    """
# Evaluate
gcn_net.eval()
test_loss = 0

preds = []
labs = []
for i, (bg, labels) in enumerate(test_loader):
    labels = labels.to(device)
    graph_feats = bg.ndata.pop('h').to(device)
    graph_feats, labels = graph_feats.to(device), labels.to(device)
    y_pred = gcn_net(bg, graph_feats)

    preds.append(y_pred.detach().numpy())
    labs.append(labels.detach().numpy())
