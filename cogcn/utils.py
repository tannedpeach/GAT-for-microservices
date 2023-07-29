import pickle as pkl
import os
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import itertools
from sklearn.metrics import roc_auc_score, average_precision_score
from matplotlib import pyplot as plt
from torch_geometric.data import Data



def load_adj(dataset):
    adj_file = os.path.join(dataset, "struct.csv")

    # Load create adjacency matrix
    adj = pd.read_csv(adj_file, header=None)
    adj = adj.values
    adj = nx.to_networkx_graph(adj)
    adj = nx.adjacency_matrix(adj)
    return adj


# for gcn
# def load_data_cma(dataset):
#     adj_file = os.path.join(dataset, "struct.csv")
#     feat_file = os.path.join(dataset, "content.csv")
#
#     # Load reate adjacency matrix
#     adj = pd.read_csv(adj_file, header=None)
#     adj = adj.values
#     adj = nx.from_numpy_matrix(adj)
#     adj = nx.adjacency_matrix(adj)
#     print("Adjacency matrix shape:", adj.shape)
#
#     # Load features
#     feat = pd.read_csv(feat_file, header=None)
#     feat = feat.values
#     features = torch.FloatTensor(feat)
#     print("Features shape:", features.shape)
#
#     return adj, features

# for gat

# for node2vec
def create_edgelist(dataset):
    adj_file = os.path.join(dataset, "struct.csv")
    feat_file = os.path.join(dataset, "content.csv")

    # Load create adjacency matrix
    adj = pd.read_csv(adj_file, header=None)
    adj = adj.values
    adj = nx.to_networkx_graph(adj)
    # adj = nx.adjacency_matrix(adj)
    # adj = nx.to_pandas_edgelist(adj)
    # adj1 = []
    nx.write_edgelist(adj, "test.edgelist", data=False)
    # G = nx.read_edgelist("test.edgelist")
    # adj = nx.to_edgelist(adj)


def load_data_cma(dataset):
    adj_file = os.path.join(dataset, "struct.csv")
    feat_file = os.path.join(dataset, "content.csv")

    # Load create adjacency matrix
    adj = pd.read_csv(adj_file, header=None)
    adj = adj.values
    adj = nx.to_networkx_graph(adj)
    # adj = nx.adjacency_matrix(adj)
    # adj = nx.to_pandas_edgelist(adj)
    adj1 = []
    adj = list(nx.to_edgelist(adj))  # , nodelist=list(range(1,27)))
    for x in adj:
        x = list(x)
        x.pop(2)
        adj1.append(x)

    # print("Adjacency matrix shape:", adj.shape)
    adj = torch.tensor(adj1, dtype=torch.long)
    adj = adj.t().contiguous()

    # Load features
    feat = pd.read_csv(feat_file, header=None)
    feat = feat.values
    features = torch.FloatTensor(feat)
    print("Features shape:", features.shape)

    return adj, features


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def plot_losses(losses, epoch_mark):
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        # plt.plot(losses[i])
        plt.plot(losses[:][i])
        # plt.axvline(epoch_mark, color='r')
        # plt.axvline(epoch_mark * 2, color='g')
    plt.show()


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


def cluster_size(cluster_list, mem):
    K = len(cluster_list)
    cluster_size = [0] * K
    for c in cluster_list:
        for x in mem:
            if x == c:
                cluster_size[c] += 1
    return cluster_size


def structural_modularity(adj, mem, cluster_list):
    print("mem: ", mem)
    print("adj", adj)
    # compute uk
    i = -1
    K = len(cluster_list)
    uk_list = [0] * K
    for key, val in adj.items():
        i += 1
        for v in val:
            print(i, key, v)
            if mem[v] == mem[i]:
                for c in cluster_list:
                    if mem[v] == mem[i] == c:
                        # print("v: ", v)
                        # print("i: ", i)
                        uk_list[c] += 1

    print("uk_list:", uk_list)

    # compute number of members in each cluster
    cluster_size = [0] * K
    cluster_size_square = []
    for c in cluster_list:
        for x in mem:
            if x == c:
                cluster_size[c] += 1
    print("cluster_size: ", cluster_size)

    # compute sigma(k1, k2)
    pair_order_list = itertools.permutations(cluster_list, 2)
    d = dict.fromkeys(pair_order_list, 0)
    pair_mult = []
    for pair in dict.keys(d):
        x = pair[0]
        y = pair[1]
        # compute Nk1*Nk2
        mult = cluster_size[pair[0]] * cluster_size[pair[1]]
        pair_mult.append(mult)
        for key, val in adj.items():
            if mem[key] == x:
                for v in val:
                    if mem[v] == y:
                        d[pair] += 1

    print("d", d)
    print("pair_mult", pair_mult)

    for s in cluster_size:
        cluster_size_square.append(s * s)

    print("cluster_size_square", cluster_size_square)

    sigma = list(d.values())
    print("sigma", sigma)
    test_square = np.power(np.sum(cluster_size), 2)
    struct_mod = 10 * (((1 / K) * (np.sum(uk_list) / (2 * np.sum(cluster_size_square)))) - (
            (2 / (K * (K - 1))) * (np.sum(sigma) / (2 * np.sum(pair_mult)))))

    # struct_mod = ((1 / K) * (np.sum(uk_list/2) / test_square)) - (
    #         (2 / (K * (K - 1))) * (np.sum(sigma) / (2 * np.sum(pair_mult))))
    print("struct mod", struct_mod)
    return struct_mod


def NED(N, size):
    K = len(size)
    ned_list = []
    e = 0.5
    for i in range(K):
        if (size[i] >= ((1 - e) * (N / K)) and size[i] <= ((1 + e) * (N / K))):
            ned_list.append(size[i])
    ned_num = (1 / N) * np.sum(ned_list)
    print("N: ", N)
    print("ned_list", ned_list)
    print("ned: ", ned_num)
    return ned_num


def IFN(mem, results, k):
    ifn_list = []
    for j in range(k):
        count = 0;
        for row in range(0, len(results)):
            for i in range(0, len(results)):
                if results[row][i] == 1 and mem[i] == j:
                    count += 1
        ifn_list.append(count);
    ifn = (1 / k) * np.sum(ifn_list)
    print("ifn ", ifn)


def AIC(mem, results, k):
    outgoing = []
    for i in range(k):
        count = 0;
        for j in range(0, len(results)):
            if mem[j] == i:
                for m in range(0, len(results)):
                    if (results[j][m] == 1):
                        count += 1
        outgoing.append(count)
    print("mem: ", mem)
    print("outgoing: ", outgoing)

    incoming = []
    for A in range(k):
        count = 0
        for B in range(k):
            if (B != A):
                for j in range(0, len(results)):
                    if (mem[j] == B):
                        for n in range(0, len(results)):
                            if (mem[n] == A):
                                if (results[j][n] == 1):
                                    count += 1
        incoming.append(count)
    incoming_sum = np.sum(incoming)
    outgoing_sum = np.sum(outgoing)
    aic = (incoming_sum + outgoing_sum) / k
    print("aic: ", aic)


def AIC2(adj, mem, cluster_list):
    k = len(cluster_list)
    out = []
    inc = []
    for c in cluster_list:
        out_count = 0
        for key, val in adj.items():
            if mem[key] == c:
                for v in val:
                    out_count += 1
        out.append(out_count)

    for c in cluster_list:
        inc_count = 0
        for key, val in adj.items():
            for v in val:
                if mem[v] == c:
                    inc_count += 1
        inc.append(inc_count)

    incoming_sum = np.sum(inc)
    outgoing_sum = np.sum(out)
    aic = incoming_sum / k
    print("adj: ", adj)
    print("incoming: ", inc)
    print("outgoing: ", out)
    print("aic2: ", aic/2)


def IRN(adj, mem, cluster_list):
    irn_clus = []
    for c1 in cluster_list:
        count = 0
        for c2 in cluster_list:
            if c1!=c2:
                for key, val in adj.items():
                    if mem[key] == c1:
                        for v in val:
                            if mem[v] == c2:
                                count += 1
        irn_clus.append(count)
    irn = np.sum(irn_clus)
    print("IRN_clus: ", irn_clus)
    print("IRN: ", irn)

# def NEDPlot():
#     N = 4
#     ind = np.arange(N)
#     width = 0.25
#
#     daytrader = [0.2054794520547945, 0.136986301369863, 0.0821917808219178]
#     bar1 = plt.bar(ind, daytrader, width, color='r')
#
#     pbw = [0.25, 0.5357142857142857, 0.21428571428571427]
#     bar2 = plt.bar(ind + width, pbw, width, color='g')
#
#     acmeair = [0.38461538461538464, 0.38461538461538464, 0.15384615384615385]
#     bar3 = plt.bar(ind + width * 2, acmeair, width, color='b')
#
#     dietapp = [0.8, 0.25, 0.1]
#     bar4 = plt.bar(ind + width * 2, dietapp, width, color='yellow')
#
#     # plt.xlabel("Dates")
#     plt.ylabel('NED')
#     plt.title("Players Score")
#
#     plt.xticks(ind + width, ['Daytrader', 'PBW', 'AcmeAir', 'DietApp'])
#     plt.legend((bar1, bar2, bar3, bar4), ('Multi-Head GAT', 'Single-Head GAT', 'coGCN'))
#     plt.show()




