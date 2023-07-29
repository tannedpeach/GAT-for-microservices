import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from layers import GraphConvolution
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.utils import softmax

# GAT Implementation 1


class GAT(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, hidden_dim5, hidden_dim6, heads=16):
        super().__init__()
#
#         # single-head attention
#         self.gat1 = GATv2Conv(input_feat_dim, hidden_dim1, heads=1)  # , add_self_loops=False)
#         self.gat2 = GATv2Conv(hidden_dim1, hidden_dim2, heads=1)
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=1e-4)
#
#         self.dgat1 = GATv2Conv(hidden_dim2, hidden_dim1, heads=1)  # , add_self_loops=False)
#         self.dgat2 = GATv2Conv(hidden_dim1, input_feat_dim, heads=1)
#
#
#         # multi-head attention
#         # self.gat1 = GATv2Conv(input_feat_dim, hidden_dim1, heads=heads)#, add_self_loops=False)
#         # self.gat2 = GATv2Conv(hidden_dim1*heads, hidden_dim2, heads=1)#, add_self_loops=False)
#         # self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=1e-4)
#         #
#         # self.dgat1 = GATv2Conv(hidden_dim2, hidden_dim1, heads=heads)#, add_self_loops=False)
#         # self.dgat2 = GATv2Conv(hidden_dim1*heads, input_feat_dim, heads=1)#, add_self_loops=False)
#
        # 5 layer encoder-decoder
        self.gat1 = GATv2Conv(input_feat_dim, hidden_dim1, heads=heads)  # , add_self_loops=False)
        self.gat2 = GATv2Conv(hidden_dim1 * heads, hidden_dim2, heads=25)  # , add_self_loops=False)
        self.gat3 = GATv2Conv(hidden_dim2 * 25, hidden_dim3, heads=16)
        self.gat4 = GATv2Conv(hidden_dim3 * 16, hidden_dim4, heads=1)
        self.gat5 = GATv2Conv(hidden_dim4 * 1, hidden_dim5, heads=1)
        # self.gat6 = GATv2Conv(hidden_dim5 * 1, hidden_dim6, heads=1)
        #self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=1e-4)

        self.dgat1 = GATv2Conv(hidden_dim5, hidden_dim4, heads=heads)  # , add_self_loops=False)
        self.dgat2 = GATv2Conv(hidden_dim4 * heads, hidden_dim3, heads=25)  # , add_self_loops=False)
        self.dgat3 = GATv2Conv(hidden_dim3 * 25, hidden_dim2, heads=16)
        self.dgat4 = GATv2Conv(hidden_dim2 * 16, hidden_dim1, heads=1)
        self.dgat5 = GATv2Conv(hidden_dim1 * 1, input_feat_dim, heads=1)
        # self.dgat6 = GATv2Conv(hidden_dim1 * 1, input_feat_dim, heads=1)

    # def encode(self, x, adj):
    #     hidden1 = self.gat1(x, adj)
    #     hidden2 = self.gat2(hidden1, adj)
    #     return hidden2
    #
    # def decode(self, hidden, adj):
    #     hidden1 = self.dgat1(hidden, adj)
    #     recon = self.dgat2(hidden1, adj)
    #     return recon

    def forward(self, x, adj):
        adj = adj.long()
        #print("size:", adj.shape())
        h = F.dropout(x, p=0.4, training=self.training)
        h = self.gat1(h, adj)
        h = F.relu(h)
        h = F.dropout(h, p=0.4, training=self.training)
        h = self.gat2(h, adj)
        h = F.relu(h)
        h = F.dropout(h, p=0.4, training=self.training)
        h = self.gat3(h, adj)
        h = F.relu(h)
        h = F.dropout(h, p=0.4, training=self.training)
        h = self.gat4(h, adj)
        h = F.relu(h)
        h = F.dropout(h, p=0.4, training=self.training)
        h = self.gat5(h, adj)
        # h = F.relu(h)
        # h = F.dropout(h, p=0.4, training=self.training)
        # h = self.gat6(h, adj)
        h = torch.nn.functional.softmax(h)

        h1 = F.dropout(h, p=0.4, training=self.training)
        #print("dgat1 adj", adj.size())
        #print("dgat1 h", h.size())
        h1 = self.dgat1(h1, adj)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=0.4, training=self.training)
        #print(adj.size())
        #print(h1.size())
        h1 = self.dgat2(h1, adj)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=0.4, training=self.training)
        h1 = self.dgat3(h1, adj)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=0.4, training=self.training)
        h1 = self.dgat4(h1, adj)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=0.4, training=self.training)
        h1 = self.dgat5(h1, adj)
        # h1 = F.relu(h1)
        # h1 = F.dropout(h1, p=0.4, training=self.training)
        # h1 = self.dgat6(h1, adj)
        h1 = torch.nn.functional.softmax(h1)

        return h1, h
# End of GAT Implementation 1


# gcn start


# class GCNAE(nn.Module):
#     def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
#          super(GCNAE, self).__init__()
#          self.encgc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
#          self.encgc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
#
#          self.decgc1 = GraphConvolution(hidden_dim2, hidden_dim1, dropout, act=F.relu)
#          self.decgc2 = GraphConvolution(hidden_dim1, input_feat_dim, dropout, act=lambda x: x)
#
#     def encode(self, x, adj):
#          hidden1 = self.encgc1(x, adj)
#          hidden2 = self.encgc2(hidden1, adj)
#          return hidden2
#
#     def decode(self, hidden, adj):
#          hidden1 = self.decgc1(hidden, adj)
#          recon = self.decgc2(hidden1, adj)
#          return recon
#
#     def forward(self, x, adj):
#          enc = self.encode(x, adj)
#          dec = self.decode(enc, adj)
#          return dec, enc
# # gcn end
class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj
