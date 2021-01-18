# -*- coding: utf-8 -*-

import re
import sys
import pickle as pkl
import numpy as np

import torch
import scipy.sparse as sp
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances
from sklearn import metrics

from sklearn.model_selection import train_test_split


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def get_embedding_matrix(vec_model, tokenizer):
    # values of word_index range from 1 to len
    # embedding_matrix = np.random.random((len(tokenizer.word_index) + 1, vec_model.dim))
    # model.vector_size
    embedding_matrix = np.random.random((len(tokenizer.word_index) + 1, vec_model.vector_size))
    for word, i in tokenizer.word_index.items():
        word = str(word)
        if word.isspace():
            embedding_vector = vec_model['blank']
        elif word not in vec_model.vocab:
            embedding_vector = vec_model['unk']
        else:
            embedding_vector = vec_model[word]
        embedding_matrix[i] = embedding_vector
    return embedding_matrix


def texts_to_idx(texts, tokenizer, max_sentence_length):
    data = np.zeros((len(texts), max_sentence_length), dtype='int32')
    for i, wordTokens in enumerate(texts):
        k = 0
        for _, word in enumerate(wordTokens):
            try:
                if k < max_sentence_length and tokenizer.word_index[word] < tokenizer.num_words:
                    data[i, k] = tokenizer.word_index[word]
                    k = k + 1
            except:
                if k < max_sentence_length:
                    data[i, k] = 0
                    k = k + 1
    return data


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_corpus(dataset_str, source_data_dir="./data/"):
    """
    Loads input corpus from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training docs/words
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test docs as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.train.index => the indices of training docs in original doc list.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'adj']
    objects = []
    for i in range(len(names)):
        with open(source_data_dir + "ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, adj = tuple(objects)

    features = sp.vstack((allx, tx)).tolil()
    labels = np.vstack((ally, ty))

    train_idx_orig = parse_index_file(
        source_data_dir + "{}.train.index".format(dataset_str))
    train_size = len(train_idx_orig)

    val_size = train_size - x.shape[0]
    test_size = tx.shape[0]

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + val_size)
    idx_test = range(allx.shape[0], allx.shape[0] + test_size)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return adj, features, labels, list(idx_train) + list(idx_val), list(idx_test), train_size, test_size


def data_train_split(Y_train, split_size=1.0):
    """
    按照标签分层抽样
    :param Y_train: 数据标签
    :param split_size: 选取的数据的比例
    :return: index_choose 划分出数据的index
    """
    if len(Y_train[0]) > 1:  # one hot representation
        Y = [np.argmax(y) for y in Y_train]
    else:
        Y = Y_train  # index representation
    Y_choose, _, index_choose, _ = train_test_split(Y, range(0, len(Y)), train_size=split_size, stratify=Y, \
                                                    random_state=42)
    return sorted(index_choose)


def train(model, optimizer, X_idx, labels, idx_train, batch_size):
    data_loader = DataLoader(idx_train, batch_size=batch_size, shuffle=True)
    model.train()
    for idx_batch in data_loader:
        optimizer.zero_grad()
        output = model(idx_batch, X_idx[idx_batch])
        preds = torch.sigmoid(output)
        loss_train = F.binary_cross_entropy(preds, labels[idx_batch])
        loss_train.backward()
        optimizer.step()

    loss_train, acc_train = test_on_batch(model, X_idx, idx_train)
    return loss_train, acc_train


def test_on_batch(model, X_idx, idx_test, labels, batch_size=128):
    data_loader = DataLoader(idx_test, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        model.eval()
        output_test = []
        for idx_batch in data_loader:
            output = model(idx_batch, X_idx[idx_batch])
            output_test.append(output)
        output_test = torch.cat(output_test, dim=0)
        preds_test = torch.sigmoid(output_test)
        loss_test = F.binary_cross_entropy(preds_test, labels[idx_test])
        acc_test = accuracy(preds_test, labels[idx_test])

    return loss_test.item(), acc_test


def compute_adj_matrix(input):
    """
    计算邻接矩阵，有不同的计算方式:
    方法1：1 - 词向量均值的similarity（满足：对角线为1，两个结点相似性越高，值越大）
    :param input:
    :return:
    """
    sim_matrix = pairwise_distances(input.tolist(), metric="cosine", n_jobs=6)
    return 1 - sim_matrix


def normalize_adj(adj, tocoo=True):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    if tocoo:
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    else:
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def preprocess_adj(adj, to_dense=True, tocoo=True):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]), tocoo=tocoo)
    # return sparse_to_tuple(adj_normalized)
    if to_dense:
        return adj_normalized.A
    else:
        return adj_normalized


def accuracy(preds, target, detail=False):
    preds = preds.max(1)[1]
    target = target.max(1)[1].long()
    correct = preds.eq(target).double()
    # correct = [1 if target[i][int(p)] == 1 else 0 for i, p in enumerate(preds)]
    if detail:
        print("Test Precision, Recall and F1-Score...")
        print(metrics.classification_report(np.array(target), np.array(preds), digits=4))
        print("Macro average Test Precision, Recall and F1-Score...")
        print(metrics.precision_recall_fscore_support(np.array(target), np.array(preds), average='macro'))
        print("Micro average Test Precision, Recall and F1-Score...")
        print(metrics.precision_recall_fscore_support(np.array(target), np.array(preds), average='micro'))

    return sum(correct) / len(target)


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    # return sparse_to_tuple(features)
    if isinstance(features, np.ndarray):
        return features
    else:
        return features.A


def save_model(model, optimizer, path, epoch, loss):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'epoch': epoch
    }, path + "_ckpt.pt", pickle_protocol=4)
    print("model saved", path + "_ckpt.pt")


def load_model(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.eval()  # 防止预测时修改模型
    return epoch, loss, path, model, optimizer


def get_adj_list(adj):
    adj_lists = dict()
    for i in range(adj.shape[0]):
        adj_lists[i] = set(np.where(adj[i])[0])
    assert len(adj_lists) == adj.shape[0], "adj_lists num != node num"
    return adj_lists


def load_file(data_path):
    with open(data_path, "rb") as f:
        data = pkl.load(f)
    return data
