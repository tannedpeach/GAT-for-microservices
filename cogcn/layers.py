import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GATv2Conv
import tensorflow as tf


# class GraphAttention(torch.nn.Module):
#     """Graph Attention Network"""
#
#     def __init__(self, dim_in, dim_h, dim_out, heads):
#         super().__init__()
#         self.dim_in = dim_in
#         self.dim_h = dim_h
#         self.dim_out = dim_out
#         self.heads = heads
#         self.weight = Parameter(torch.FloatTensor(dim_in, dim_out))
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         torch.nn.init.xavier_uniform_(self.weight)
#
#     def forward(self, x, edge_index):
#         h = F.dropout(x, p=0.6, training=self.training)
#         h = self.gat1(x, edge_index)
#         h = F.relu(h)
#         h = F.dropout(x, p=0.6, training=self.training)
#         h = self.gat2(h, edge_index)
#         h = lambda h: h

class GraphConvolution(Module):
#     """
#     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
#     """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
