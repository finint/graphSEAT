# -*- coding: utf-8 -*-

from utils import *
import os
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
from model import SequenceGraphAtt


class CONFIG_PARAS():
    def __init__(self):
        # 参数等常量存放
        self.data_dir = "./data/r52"
        self.data_graph_path = self.data_dir + "/r52_input_base.pkl"
        self.data_seqence_path = self.data_dir + "/r52_sequence.pkl"
        self.data_reduce_size = 1.  # 训练数据占总训练集合比例

        self.random_seed = 42
        self.hidden_enc1 = 100
        self.hidden_enc2 = 200
        self.num_sample1 = 20
        self.num_sample2 = 20
        self.hidden_graph = None
        self.hidden_rnn = 100
        self.att_size = 100  # graph attention size

        self.epochs = 64
        self.lr = 0.005
        self.weight_decay = 0.0
        self.batch_size = 64

        self.cuda = True
        self.device = torch.device("cuda" if self.cuda and torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = CONFIG_PARAS()

    # load features
    adj, features, labels, idx_train, idx_test = load_file(data_path=args.data_graph_path)
    X_idx, embedding_matrix = load_file(data_path=args.data_seqence_path)

    # random.seed(10)
    idx_val = np.array(random.sample(idx_train, int(len(idx_train) * 0.1)))
    idx_train = np.array(list(set(idx_train) - set(idx_val)))
    idx_test = idx_test

    # 邻接矩阵 处理
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_lists = get_adj_list(preprocess_adj(adj))
    args.feature_dim = features.shape[1]
    args.num_classes = labels.shape[1]
    features_layer = torch.nn.Embedding(num_embeddings=features.shape[0], embedding_dim=args.feature_dim)
    features_layer.weight = nn.Parameter(torch.FloatTensor(features), requires_grad=False)
    embedding_matrix = torch.FloatTensor(embedding_matrix)
    embedding_layer = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
    embedding_layer.weight = nn.Parameter(embedding_matrix.to(args.device), requires_grad=False)
    X_idx = torch.LongTensor(X_idx)
    labels = torch.FloatTensor(labels).to(args.device)

    best_acc_list = []
    for i in range(10):
        model = SequenceGraphAtt(features_layer, adj_lists, num_classes=args.num_classes, enc1_hidden=args.hidden_enc1, \
                                 enc2_hidden=args.hidden_enc2, rnn_hidden=args.hidden_rnn, num_sample1=args.num_sample1, \
                                 num_sample2=args.num_sample2, embedding_layer=embedding_layer, cuda=args.cuda,
                                 dropout=0.5, att_size=64)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                               weight_decay=args.weight_decay)
        # 学习率下降
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True,
                                                               patience=3)
        # to tensor
        best_hist = (np.inf, 0)
        stop_count, patient = 0, 5
        for epoch in range(1, args.epochs + 1):
            t = time.time()
            loss_train, acc_train = train(model, optimizer, X_idx, labels, idx_train, \
                                          batch_size=args.batch_size)
            loss_val, acc_val = test_on_batch(model, X_idx, idx_val)
            loss_test, acc_test = test_on_batch(model, X_idx, idx_test)
            scheduler.step(loss_val)
            if loss_val <= best_hist[0]:
                best_hist = (loss_val, acc_val)
                # model, optimizer, path, epoch, loss
                model_save_path = args.data_dir + "/ckpt/seq_dual_att"
                save_model(model=model, optimizer=optimizer, path=model_save_path, epoch=epoch, loss=loss_val)
                stop_count = 0
            else:
                stop_count += 1

            print('Epoch: {:04d}'.format(epoch),
                  'loss_train: {:.4f}'.format(loss_train),
                  'acc_train: {:.4f}'.format(acc_train),
                  'loss_val: {:.4f}'.format(loss_val),
                  'acc_val: {:.4f}'.format(acc_val),
                  'loss_test: {:.4f}'.format(loss_test),
                  'acc_test: {:.4f}'.format(acc_test),
                  'time: {:.4f}s'.format(time.time() - t))
            if stop_count > patient:
                break

        epoch, loss, path, model, optimizer = load_model(model_save_path + "_ckpt.pt", model, optimizer)
        data_loader = DataLoader(idx_test, batch_size=64, shuffle=False)
        with torch.no_grad():
            model.eval()
            output_test = []
            for idx_batch in data_loader:
                output = model(idx_batch, X_idx[idx_batch])
                output_test.append(output)
            output_test = torch.cat(output_test, dim=0)
            preds_test = torch.sigmoid(output_test)
            loss_test = F.binary_cross_entropy(preds_test, labels[idx_test])
            acc_test = accuracy(preds_test, labels[idx_test], detail=True)
        print("best model result on test\n loss:{:.4f} acc{:.4f}".format(loss_test, acc_test))

        best_acc_list.append(acc_test)