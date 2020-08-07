import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgllife.model.model_zoo import GCNPredictor


class Predictor(nn.Module):

    def __init__(self, in_feats, hidden_feats=[32, 32], batchnorm=[True, True], dropout=[0, 0], predictor_hidden_feats=32, n_tasks=6):
        super(Predictor, self).__init__()

        self.net1 = GCNPredictor(in_feats=n_feats,
                               hidden_feats=hidden_feats,
                               batchnorm=batchnorm,
                               dropout=dropout,
                               predictor_hidden_feats=predictor_hidden_feats,
                               n_tasks=n_tasks)

        self.net2 = GCNPredictor(in_feats=n_feats,
                               hidden_feats=hidden_feats,
                               batchnorm=batchnorm,
                               dropout=dropout,
                               predictor_hidden_feats=predictor_hidden_feats,
                               n_tasks=n_tasks)

        self.predict = nn.Linear(n_tasks*2, 1)

    def forward(self, g):

        wt = self.net1(g[0])
        mut = self.net2(g[1])

        flattened = torch.cat(wt, mut)

        return self.predict(flattened)
