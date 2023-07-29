'''
Reference implementation of node2vec. 
Author: Aditya Grover
For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
from utils import create_edgelist
from kmeans import Clustering
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from utils import load_data_cma, preprocess_graph, plot_losses, load_adj, EarlyStopping, structural_modularity, NED, cluster_size, IFN, AIC, AIC2, IRN
import os
import csv
from kmeans import Clustering

def parse_args():
    '''
    Parses the node2vec arguments.
    '''

    parser = argparse.ArgumentParser(description="Run node2vec.")
    parser.add_argument('--k', type=int, default=6, help='Number of clusters.')
    parser.add_argument('--dataset-str', type=str, default=None, help='type of dataset.')

    parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='emb/karate.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


def read_graph():
    '''
    Reads the input network in networkx.
    '''
    create_edgelist(args.dataset_str)
    fh = open("C:/Users/tanis/Desktop/cogcn-main/cogcn/test.edgelist", "rb")
    EL = nx.read_edgelist(fh)
    print(EL)
    if args.weighted:
        G = nx.read_edgelist("C:/Users/tanis/Desktop/cogcn-main/cogcn/test.edgelist", nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist("C:/Users/tanis/Desktop/cogcn-main/cogcn/test.edgelist", nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G


def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, window=args.window_size, min_count=0, sg=1, workers=args.workers)
    model.build_vocab(walks, progress_per=10000)
    model.train(walks, total_examples=model.corpus_count, epochs=30, report_delay=1)
    # model.save_word2vec_format(args.output)

    return model


def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''

    nx_G = read_graph()
    G = node2vec.Graph(nx_G, args.undirected, args.p, args.q)
    print("G", nx_G.size())
    G.preprocess_transition_probs()
    print("G", nx_G.size())

    walks = G.simulate_walks(args.num_walks, args.walk_length)
    model = learn_embeddings(walks)
    emb_df = (
        pd.DataFrame(
            [model.wv.get_vector(str(n)) for n in nx_G.nodes()],
            index=nx_G.nodes
        )
    )
    X = emb_df.values
    # kmeans = Clustering(args.k)
    # kmeans.cluster(X)
    # return kmeans.get_membership()
    # clustering = SpectralClustering(
    #     n_clusters=5,
    #     assign_labels='discretize',
    #     random_state=0
    # ).fit(X)

    clustering = KMeans(n_clusters=args.k,  n_init=5, max_iter=250)
    clustering.fit(X)
    M = clustering.labels_
    # M = ['\0', '\0', 5, 5, 4, 3, 3, '\0', 4, 2, 2, 2, 2, 2, 0, 0, 0, 4, 1, 1]

    print("nodes", nx_G.nodes())



    # kmeans = Clustering(args.k)
    # kmeans.cluster(model)
    # M1 = kmeans.get_membership();

    print(M)

    adj_norm, features = load_data_cma(args.dataset_str)
    n_nodes, feat_dim = features.shape
    size = cluster_size([0, 1, 2, 3, 4, 5], M)
    structfile = os.path.join("C:/Users/tanis/Desktop/cogcn-main/cogcn/data/apps/", args.dataset_str + "/struct.csv")
    results = []
    with open(structfile) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  # change contents to floats
        for row in reader:  # each row is a list
            results.append(row)

    adj_test = load_adj(args.dataset_str)
    graph_test = nx.to_networkx_graph(adj_test)
    adj_test = nx.convert.to_dict_of_lists(graph_test)
    structural_modularity(adj_test, M, [0, 1, 2, 3, 4, 5])
    NED(n_nodes, size)
    IFN(M, results, 6)
    AIC(M, results, 6)
    AIC2(adj_test, M, [0, 1, 2, 3, 4, 5])
    IRN(adj_test, M, [0, 1, 2, 3, 4, 5])





if __name__ == "__main__":
    args = parse_args()
    main(args)