# train for gat

from __future__ import division
from __future__ import print_function

import argparse
import time
import sys
import os
import pickle
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch import optim
import tensorflow as tf
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping
import networkx as nx
import csv

from model import GAT
from optimizer import compute_structure_loss, compute_attribute_loss, update_o1, update_o2
from utils import load_data_cma, preprocess_graph, plot_losses, load_adj, EarlyStopping, structural_modularity, NED, cluster_size, IFN, AIC, AIC2, IRN
from kmeans import Clustering

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
# 5000 epochs is where the pretrain loss starts hovering around 5.0 - 6.0
parser.add_argument('--k', type=int, default=6, help='Number of clusters.')
parser.add_argument('--preepochs', type=int, default=350, help='Number of epochs to pre-train.')
parser.add_argument('--clusepochs', type=int, default=1, help='Number of epochs to pre-train for clustering.')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=64, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=32, help='Number of units in hidden layer 2.')
parser.add_argument('--hidden3', type=int, default=32, help='Number of units in hidden layer 3.')
parser.add_argument('--hidden4', type=int, default=32, help='Number of units in hidden layer 4.')
parser.add_argument('--hidden5', type=int, default=32, help='Number of units in hidden layer 5.')
parser.add_argument('--hidden6', type=int, default=32, help='Number of units in hidden layer 6.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--lambda1', type=float, default=0.1, help='Structure loss weight.')
parser.add_argument('--lambda2', type=float, default=0.1, help='Attribute loss weight.')
parser.add_argument('--lambda3', type=float, default=0.8, help='Clustering loss weight.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default=None, help='type of dataset.')
parser.add_argument('--outfile', type=str, default='embeddings', help='output embeddings file.')

args = parser.parse_args()
early_stopping = EarlyStopping()


def gae_for(args):
    print("Using {} dataset".format(args.dataset_str))
    adj_norm, features = load_data_cma(args.dataset_str)
    n_nodes, feat_dim = features.shape
    adj = load_adj(args.dataset_str)

    # Some preprocessing
    adj = preprocess_graph(adj)




    model = GAT(feat_dim, args.hidden1, args.hidden2, args.hidden3, args.hidden4, args.hidden5, args.hidden6)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-20)

    # gamma = 0.98
    gamma = 1.8
    schedule_update_interval = 400
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # initialize all outlier scores with the equal values summing to 1
    init_value = [1. / n_nodes] * n_nodes
    o_1 = torch.FloatTensor(init_value)  # structural outlier
    o_2 = torch.FloatTensor(init_value)  # attribute outlier

    lossfn = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    lambda1 = args.lambda1 / (args.lambda1 + args.lambda2)
    lambda2 = args.lambda2 / (args.lambda1 + args.lambda2)

    kmeans = Clustering(args.k)

    # PRETRAIN ON STRUCTURE AND ATTRIBUTE LOSSES, NO OUTLIER LOSS
    for epoch in range(args.preepochs):
        model.train()
        optimizer.zero_grad()

        recon, embed = model(features, adj_norm)

        structure_loss = compute_structure_loss(adj, embed, o_1)
        attribute_loss = compute_attribute_loss(lossfn, features, recon, o_2)

        loss = lambda1 * structure_loss + lambda2 * attribute_loss
        # early_stopping(loss)

        # Update the functions F and G (embedding network)
        loss.backward()
        early_stopping(loss)
        cur_loss = loss.item()
        optimizer.step()



        if (epoch + 1) % 100 == 0:
            # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss), "time=", "{:.5f}".format(time.time() - t), "Pretrain:",pretrain)
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss), "lr=",
                  "{:.5f}".format(scheduler.get_last_lr()[0]))

        # early_stopping(loss)
        # if early_stopping.early_stop:
        #     break

    # Initialize clusters
    recon, embed = model(features, adj_norm)
    kmeans.cluster(embed)

    sloss = []
    aloss = []
    l = []
    closs = []

    # TRAIN ON ALL THREE LOSES WITH OUTLIER UPDATES
    for epoch in range(args.epochs):
        # Update the values of O_i1 and O_i2
        o_1 = update_o1(adj, embed)
        o_2 = update_o2(features, recon)

        if (epoch + 1) % schedule_update_interval == 0:
            scheduler.step()

        model.train()
        optimizer.zero_grad()

        recon, embed = model(features, adj_norm)

        kmeans.cluster(embed)

        structure_loss = compute_structure_loss(adj, embed, o_1)
        attribute_loss = compute_attribute_loss(lossfn, features, recon, o_2)
        clustering_loss = kmeans.get_loss(embed)

        loss = (args.lambda1 * structure_loss) + (args.lambda2 * attribute_loss) + (args.lambda3 * clustering_loss)
        # loss = (args.lambda1 * structure_loss) + (args.lambda2 * attribute_loss)

        # early_stopping(loss)


        # Update the functions F and G (embedding network)
        loss.backward()
        early_stopping(loss)
        cur_loss = loss.item()
        optimizer.step()



        if (epoch + 1) % 100 == 0:
            # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss), "time=", "{:.5f}".format(time.time() - t), "Pretrain:",pretrain)
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss), "lr=",
                  "{:.5f}".format(scheduler.get_last_lr()[0]))
            # early_stopping(loss)
        # early_stopping(loss)
        sloss.append(structure_loss.detach().numpy())
        aloss.append(attribute_loss.detach().numpy())
        closs.append(clustering_loss.detach().numpy())
        l.append(loss.detach().numpy())

        # early_stopping(loss)

    losses = [sloss, aloss, closs, l]

    # Extract embeddings
    # adj_norm = preprocess_graph(adj)
    recon, embed = model(features, adj_norm)
    embed = embed.detach().cpu().numpy()

    embfile = os.path.join(args.dataset_str, args.outfile + ".pkl")
    with open(embfile, "wb") as f:
        pickle.dump(embed, f)

    o_1 = o_1.detach().cpu().numpy()
    o_2 = o_2.detach().cpu().numpy()
    outlfile = os.path.join(args.dataset_str, args.outfile + "_outliers.pkl")
    with open(outlfile, "wb") as f:
        pickle.dump([o_1, o_2], f)

    membfile = os.path.join(args.dataset_str, args.outfile + "_membership.pkl")
    with open(membfile, "wb") as f:
        pickle.dump(kmeans.get_membership(), f)
    print("membership", kmeans.get_membership())
    adj_test = load_adj(args.dataset_str)
    graph_test = nx.to_networkx_graph(adj_test)
    adj_test = nx.convert.to_dict_of_lists(graph_test)
    structural_modularity(adj_test, kmeans.get_membership(), [0, 1, 2, 3, 4, 5])
    size = cluster_size([0, 1, 2, 3, 4, 5], kmeans.get_membership())
    NED(n_nodes, size)


    structfile = os.path.join("C:/Users/tanis/Desktop/cogcn-main/cogcn/data/apps/",args.dataset_str+"/struct.csv")
    results = []
    with open(structfile) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  # change contents to floats
        for row in reader:  # each row is a list
            results.append(row)
    # for row in range(0, len(results)):
    #     for i in range(0, len(results)):
    #         if results[row][i]==1:
    #             print("row, i", row, i)
    IFN(kmeans.get_membership(), results, 6)
    AIC(kmeans.get_membership(), results, 6)
    plot_losses(losses, epoch)
    AIC2(adj_test, kmeans.get_membership(), [0, 1, 2, 3, 4, 5])
    IRN(adj_test, kmeans.get_membership(), [0, 1, 2, 3, 4, 5])
    # NEDPlot();


if __name__ == '__main__':
    gae_for(args)

#

# train for gcn

# from __future__ import division
# from __future__ import print_function
#
# import argparse
# import time
# import sys
# import os
# import pickle
# import numpy as np
# import scipy.sparse as sp
# import torch
# import torch.nn as nn
# from torch import optim
# from matplotlib import pyplot as plt
# import networkx as nx
# import csv
#
# from model import GCNAE
# from optimizer import compute_structure_loss, compute_attribute_loss, update_o1, update_o2
# from utils import load_data_cma, preprocess_graph, plot_losses, structural_modularity, load_adj, cluster_size, IFN, NED, AIC, AIC2, IRN
# from kmeans import Clustering
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
# parser.add_argument('--seed', type=int, default=42, help='Random seed.')
# # 5000 epochs is where the pretrain loss starts hovering around 5.0 - 6.0
# parser.add_argument('--k', type=int, default=6, help='Number of clusters.')
# parser.add_argument('--preepochs', type=int, default=350, help='Number of epochs to pre-train.')
# parser.add_argument('--clusepochs', type=int, default=1, help='Number of epochs to pre-train for clustering.')
# parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
# parser.add_argument('--hidden1', type=int, default=64, help='Number of units in hidden layer 1.')
# parser.add_argument('--hidden2', type=int, default=32, help='Number of units in hidden layer 2.')
# parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
# parser.add_argument('--lambda1', type=float, default=0.1, help='Structure loss weight.')
# parser.add_argument('--lambda2', type=float, default=0.1, help='Attribute loss weight.')
# parser.add_argument('--lambda3', type=float, default=0.8, help='Clustering loss weight.')
# parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
# parser.add_argument('--dataset-str', type=str, default=None, help='type of dataset.')
# parser.add_argument('--outfile', type=str, default='embeddings', help='output embeddings file.')
#
# args = parser.parse_args()
#
# def gae_for(args):
#     print("Using {} dataset".format(args.dataset_str))
#     adj, features = load_data_cma(args.dataset_str)
#     n_nodes, feat_dim = features.shape
#
#     # Some preprocessing
#     adj_norm = preprocess_graph(adj)
#
#     model = GCNAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)
#
#     gamma = 0.98
#
#
#     # initialize all outlier scores with the equal values summing to 1
#     init_value = [1./n_nodes] * n_nodes
#     o_1 = torch.FloatTensor(init_value) # structural outlier
#     o_2 = torch.FloatTensor(init_value) # attribute outlier
#
#     lossfn = nn.MSELoss(reduction='none')
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)
#
#     schedule_update_interval = 400
#     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
#
#     lambda1 = args.lambda1 / (args.lambda1 + args.lambda2)
#     lambda2 = args.lambda2 / (args.lambda1 + args.lambda2)
#
#     kmeans = Clustering(args.k)
#
#     # PRETRAIN ON STRUCTURE AND ATTRIBUTE LOSSES, NO OUTLIER LOSS
#     for epoch in range(args.preepochs):
#         model.train()
#         optimizer.zero_grad()
#
#         recon, embed = model(features, adj_norm)
#
#         structure_loss = compute_structure_loss(adj_norm, embed, o_1)
#         attribute_loss = compute_attribute_loss(lossfn, features, recon, o_2)
#
#         loss = lambda1 * structure_loss + lambda2 * attribute_loss
#
#         # Update the functions F and G (embedding network)
#         loss.backward()
#         cur_loss = loss.item()
#         optimizer.step()
#
#         if (epoch+1) % 100 == 0:
#             #print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss), "time=", "{:.5f}".format(time.time() - t), "Pretrain:",pretrain)
#             print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss), "lr=", "{:.5f}".format(scheduler.get_last_lr()[0]))
#
#
#     # Initialize clusters
#     recon, embed = model(features, adj_norm)
#     kmeans.cluster(embed)
#
#     sloss = []
#     aloss = []
#     l = []
#     closs = []
#     # TRAIN ON ALL THREE LOSES WITH OUTLIER UPDATES
#     for epoch in range(args.epochs):
#         # Update the values of O_i1 and O_i2
#         o_1 = update_o1(adj_norm, embed)
#         o_2 = update_o2(features, recon)
#
#         if (epoch+1) % schedule_update_interval == 0:
#             scheduler.step()
#
#         model.train()
#         optimizer.zero_grad()
#
#         recon, embed = model(features, adj_norm)
#
#         kmeans.cluster(embed)
#
#         structure_loss = compute_structure_loss(adj_norm, embed, o_1)
#         attribute_loss = compute_attribute_loss(lossfn, features, recon, o_2)
#         clustering_loss = kmeans.get_loss(embed)
#
#         loss = (args.lambda1 * structure_loss) + (args.lambda2 * attribute_loss) + (args.lambda3 * clustering_loss)
#         #loss = (args.lambda1 * structure_loss) + (args.lambda2 * attribute_loss)
#
#         # Update the functions F and G (embedding network)
#         loss.backward()
#         cur_loss = loss.item()
#         optimizer.step()
#
#         if (epoch+1) % 100 == 0:
#             #print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss), "time=", "{:.5f}".format(time.time() - t), "Pretrain:",pretrain)
#             print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss), "lr=", "{:.5f}".format(scheduler.get_last_lr()[0]))
#         sloss.append(structure_loss.detach().numpy())
#         aloss.append(attribute_loss.detach().numpy())
#         closs.append(clustering_loss.detach().numpy())
#         l.append(loss.detach().numpy())
#
#     losses = [sloss, aloss, closs, l]
#     plot_losses(losses, epoch)
#
#
#     # Extract embeddings
#     adj_norm = preprocess_graph(adj)
#     recon, embed = model(features, adj_norm)
#     embed = embed.detach().cpu().numpy()
#
#     embfile = os.path.join(args.dataset_str, args.outfile+".pkl")
#     with open(embfile,"wb") as f:
#         pickle.dump(embed, f)
#
#     o_1 = o_1.detach().cpu().numpy()
#     o_2 = o_2.detach().cpu().numpy()
#     outlfile = os.path.join(args.dataset_str, args.outfile+"_outliers.pkl")
#     with open(outlfile,"wb") as f:
#         pickle.dump([o_1, o_2], f)
#
#     membfile = os.path.join(args.dataset_str, args.outfile+"_membership.pkl")
#     with open(membfile,"wb") as f:
#         pickle.dump(kmeans.get_membership(), f)
#     adj_test = load_adj(args.dataset_str)
#     graph_test = nx.to_networkx_graph(adj_test)
#     adj_test = nx.convert.to_dict_of_lists(graph_test)
#     print("adj_test:", adj_test)
#     structural_modularity(adj_test, kmeans.get_membership(), [0, 1, 2, 3, 4, 5])
#     structfile = os.path.join("C:/Users/tanis/Desktop/cogcn-main/cogcn/data/apps/", args.dataset_str + "/struct.csv")
#     results = []
#     with open(structfile) as csvfile:
#         reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  # change contents to floats
#         for row in reader:  # each row is a list
#             results.append(row)
#     size = cluster_size([0, 1, 2, 3, 4, 5], kmeans.get_membership())
#     NED(n_nodes, size)
#     IFN(kmeans.get_membership(), results, 6)
#     AIC(kmeans.get_membership(), results, 6)
#     AIC2(adj_test, kmeans.get_membership(), [0, 1, 2, 3, 4, 5])
#     IRN(adj_test, kmeans.get_membership(), [0, 1, 2, 3, 4, 5])
#
#
# if __name__ == '__main__':
#     gae_for(args)
