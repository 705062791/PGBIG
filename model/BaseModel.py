#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math

class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_c, out_c, node_n = 22, seq_len = 35, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_c
        self.out_features = out_c
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        self.weight_seq = Parameter(torch.FloatTensor(seq_len, seq_len))

        self.weight_c = Parameter(torch.FloatTensor(in_c, out_c))

        if bias:
            self.bias = Parameter(torch.FloatTensor(seq_len))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.support = None

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.att.size(1))
        self.weight_c.data.uniform_(-stdv, stdv)
        self.weight_seq.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        #input [b,c,22,35]

        #先进行图卷积再进行空域卷积
        # [b,c,22,35] -> [b,35,22,c] -> [b,35,22,c]
        support     = torch.matmul(self.att, input.permute(0, 3, 2, 1))

        # [b,35,22,c] -> [b,35,22,64]
        output_gcn  = torch.matmul(support, self.weight_c)


        #进行空域卷积
        # [b,35,22,64] -> [b,22,64,35]
        output_fc = torch.matmul(output_gcn.permute(0, 2, 3, 1), self.weight_seq).permute(0, 2, 1, 3).contiguous()


        if self.bias is not None:
            return (output_fc + self.bias)
        else:
            return output_fc

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, channal, p_dropout, bias=True, node_n=22, seq_len = 20):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = channal
        self.out_features = channal

        self.gc1 = GraphConvolution(channal, channal, node_n=node_n, seq_len=seq_len, bias=bias)
        self.bn1 = nn.BatchNorm1d(channal*node_n*seq_len)

        self.gc2 = GraphConvolution(channal, channal, node_n=node_n, seq_len=seq_len, bias=bias)
        self.bn2 = nn.BatchNorm1d(channal*node_n*seq_len)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):

        y = self.gc1(x)
        b, c, n, l = y.shape
        y = y.view(b, -1).contiguous()
        y = self.bn1(y).view(b, c, n, l).contiguous()
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, c, n, l = y.shape
        y = y.view(b, -1).contiguous()
        y = self.bn2(y).view(b, c, n, l).contiguous()
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, in_channal, out_channal, node_n=22, seq_len=20, p_dropout=0.3, num_stage=1 ):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution(in_c=in_channal, out_c=out_channal, node_n=node_n, seq_len=seq_len)
        self.bn1 = nn.BatchNorm1d(out_channal*node_n*seq_len)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(channal=out_channal, p_dropout=p_dropout, node_n=node_n, seq_len=seq_len))

        self.gcbs = nn.ModuleList(self.gcbs)
        self.gc7 = GraphConvolution(in_c=out_channal, out_c=in_channal, node_n=node_n, seq_len=seq_len)
        self.bn2 = nn.BatchNorm1d(in_channal*node_n*seq_len)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()


    def forward(self, x):

        y = self.gc1(x)
        b, c, n, l = y.shape
        y = y.view(b, -1).contiguous()
        y = self.bn1(y).view(b, c, n, l).contiguous()
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        y = self.gc7(y)
        # b, n, f = y.shape
        # y = self.bn2(y.view(b, -1)).view(b, n, f)
        # y = self.act_f(y)
        # y = self.do(y)

        return y + x

class GCN_encoder(nn.Module):
    def __init__(self, in_channal, out_channal, node_n=22, seq_len=20, p_dropout=0.3, num_stage=1 ):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN_encoder, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution(in_c=in_channal, out_c=out_channal, node_n=node_n, seq_len=seq_len)
        self.bn1 = nn.BatchNorm1d(out_channal*node_n*seq_len)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(channal=out_channal, p_dropout=p_dropout, node_n=node_n, seq_len=seq_len))

        self.gcbs = nn.ModuleList(self.gcbs)
        self.gc7 = GraphConvolution(in_c=out_channal, out_c=out_channal, node_n=node_n, seq_len=seq_len)
        self.bn2 = nn.BatchNorm1d(out_channal*node_n*seq_len)
        self.reshape_conv = torch.nn.Conv2d(in_channels=in_channal, out_channels=out_channal, kernel_size=(1, 1))
        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()


    def forward(self, x):

        y = self.gc1(x)
        b, c, n, l = y.shape
        y = y.view(b, -1).contiguous()
        y = self.bn1(y).view(b, c, n, l).contiguous()
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        y = self.gc7(y)
        # b, c, n, l = y.shape
        # y = self.bn2(y.view(b, -1)).view(b, c, n, l).contiguous()
        # y = self.act_f(y)
        # y = self.do(y)

        return y + self.reshape_conv(x)

class GCN_decoder(nn.Module):
    def __init__(self, in_channal, out_channal, node_n=22, seq_len=20, p_dropout=0.3, num_stage=1):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN_decoder, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution(in_c=in_channal, out_c=in_channal, node_n=node_n, seq_len=seq_len)
        self.bn1 = nn.BatchNorm1d(in_channal*node_n*seq_len)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(channal=in_channal, p_dropout=p_dropout, node_n=node_n, seq_len=seq_len))

        self.gcbs = nn.ModuleList(self.gcbs)
        self.gc7 = GraphConvolution(in_c=in_channal, out_c=out_channal, node_n=node_n, seq_len=seq_len)
        self.bn2 = nn.BatchNorm1d(in_channal*node_n*seq_len)

        self.reshape_conv = torch.nn.Conv2d(in_channels=in_channal, out_channels=out_channal, kernel_size=(1, 1))

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, c, n, l = y.shape
        y = y.view(b, -1).contiguous()
        y = self.bn1(y).view(b, c, n, l).contiguous()
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        y = self.gc7(y) + self.reshape_conv(x)

        return y


