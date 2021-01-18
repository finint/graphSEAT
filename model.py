# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable

import random


class SeqAttentionLayer(nn.Module):
    """implementation of sequence attention in paper: Hierarchical Attention Networks for Document Classification"""
    def __init__(self, input_dimension, attention_size, dropout=0., cuda=True):
        super(SeqAttentionLayer, self).__init__()
        self.cuda = cuda
        self.device = torch.device("cuda" if self.cuda and torch.cuda.is_available() else "cpu")
        self.attention_size = attention_size
        # sequence attention
        self.seq_attention = nn.Linear(input_dimension, attention_size)
        self.seq_attention = self.seq_attention.to(self.device)
        # context vector
        self.seq_context_vector = nn.Linear(attention_size, 1, bias=False).to(self.device)
        self.seq_context_vector = self.seq_context_vector.to(self.device)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, input_sequence):
        assert len(input_sequence.size()) == 3
        seq_att = self.seq_attention(input_sequence.to(self.device))
        seq_att = F.tanh(seq_att)
        seq_att = self.seq_context_vector(seq_att).squeeze()
        seq_weights = F.softmax(seq_att)
        weighted_sequence = (input_sequence * seq_weights.unsqueeze(dim=2))
        weighted_sum = weighted_sequence.sum(dim=1)
        return weighted_sum, weighted_sequence


class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """

    def __init__(self, features, features_dim, cuda=False, gcn=False, att_size=100):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()
        self.features = features
        self.feature_dim = features_dim
        self.gcn = gcn
        self.cuda = cuda
        self.device = torch.device("cuda" if self.cuda and torch.cuda.is_available() else "cpu")
        self.neigh_att = SeqAttentionLayer(input_dimension=features_dim, \
                                           attention_size=att_size, dropout=0., cuda=cuda)
    def forward(self, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set
        if num_sample is not None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, num_sample,)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)   # 可以加权重在这里
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list)).cuda()
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        neigh_unique_index = [[unique_nodes[i] for i in samp] for samp in samp_neighs]
        seq_neigh = Variable(torch.zeros(len(samp_neighs), num_sample, self.feature_dim))
        for i, neigh in enumerate(neigh_unique_index):
            seq_neigh[i, :len(neigh)] = embed_matrix[neigh]
        weighted_sum, weighted_sequence = self.neigh_att(seq_neigh.to(self.device))
        to_feats = F.relu(weighted_sum.div(num_neigh))
        return F.dropout(to_feats, p =0.5)


class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features, feature_dim,
            embed_dim, adj_lists, aggregator,
            num_sample=10,
            base_model=None, gcn=False, cuda=False):
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.device = torch.device("cuda" if self.cuda and torch.cuda.is_available() else "cpu")
        self.aggregator.cuda = cuda
        self.weight = nn.Parameter(torch.FloatTensor(\
            embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim).to(self.device), requires_grad=True)
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.
        nodes     -- list of nodes
        """
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes],
                self.num_sample)
        if not self.gcn:
            if self.cuda:
                self_feats = self.features(torch.LongTensor(nodes)).to(self.device)
            else:
                self_feats = self.features(torch.LongTensor(nodes))
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats
        combined = F.relu(self.weight.mm(combined.t()))
        return combined


class SequenceGraphAtt(nn.Module):
    def __init__(self, features_layer, adj_lists, num_classes, enc1_hidden, enc2_hidden, rnn_hidden,\
                 num_sample1, num_sample2, embedding_layer, cuda=False, dropout=0.5, gcn=True, att_size=64):
        super(SequenceGraphAtt, self).__init__()
        agg1 = MeanAggregator(features_layer, features_dim=features_layer.embedding_dim, cuda=cuda, att_size=att_size)
        enc1 = Encoder(features=features_layer, feature_dim=features_layer.embedding_dim, embed_dim=enc1_hidden,
                       adj_lists=adj_lists, aggregator=agg1, gcn=gcn, cuda=cuda)
        agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), features_dim=enc1.embed_dim, cuda=cuda, att_size=att_size)
        enc2 = Encoder(lambda nodes: enc1(nodes).t(), feature_dim=enc1.embed_dim, embed_dim=enc2_hidden,
                       adj_lists=adj_lists, aggregator=agg2, base_model=enc1, gcn=gcn, cuda=cuda)
        enc1.num_sample = num_sample1
        enc2.num_sample = num_sample2

        self.cuda = cuda
        self.device = torch.device("cuda" if self.cuda and torch.cuda.is_available() else "cpu")

        # layers
        self.embedding_layer = embedding_layer
        self.enc = enc2
        self.rnn_hidden = rnn_hidden
        self.rnn_layer = nn.GRU(input_size=features_layer.embedding_dim, hidden_size=rnn_hidden, num_layers=2,
                                batch_first=True, bidirectional=True, dropout=dropout)
        self.att = SeqAttentionLayer(input_dimension=self.enc.embed_dim, attention_size=att_size, dropout=0.5, cuda=cuda)
        self.rnn_layer = self.rnn_layer.to(self.device)

        # weights
        self.weight1 = nn.Parameter(torch.FloatTensor(self.rnn_layer.hidden_size*2, self.enc.embed_dim).to(self.device),\
                                    requires_grad=True)
        init.xavier_uniform(self.weight1)
        self.weight2 = nn.Parameter(torch.FloatTensor(self.enc.embed_dim, num_classes).to(self.device), requires_grad=True)
        init.xavier_uniform(self.weight2)

    def forward(self, nodes, seq_input):
        input_embedded = self.embedding_layer(seq_input.to(self.device))
        rnn_output, rnn_hidden = self.rnn_layer(input_embedded)
        # concatenate normal RNN's last time step(-1) output and reverse RNN's last time step(0) output
        rnn_embeds = torch.cat([rnn_output[:, -1, :self.rnn_hidden], rnn_output[:, 0, self.rnn_hidden:]], dim=1)
        rnn_embeds = F.dropout(rnn_embeds, p=0.5) #.mm(self.weight1)
        graph_embeds = self.enc(nodes)
        combined_embeds, _ = self.att(torch.cat([rnn_embeds.unsqueeze(dim=1), graph_embeds.t().unsqueeze(dim=1)], dim=1))

        scores = combined_embeds.mm(self.weight2)
        return scores

