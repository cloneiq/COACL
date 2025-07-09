# -*- coding: utf-8 -*-
# @Time    : 2021/3/22
# @Author  : Aspen Stars
# @Contact : aspenstars@qq.com
# @FileName: text_encoder.py
import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import math
import numpy as np

from modules.Transformer import MultiHeadedAttention, PositionalEncoding, PositionwiseFeedForward, Encoder, \
    EncoderLayer, Embeddings, SublayerConnection, clones


class TextEncoder(nn.Module):
    def __init__(self, d_model, d_ff, num_layers, tgt_vocab, num_labels=14, h=3, dropout=0.1):
        super(TextEncoder, self).__init__()
        # TODO:
        #  将eos,pad的index改为参数输入
        self.eos_idx = 0
        self.pad_idx = 0

        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        self.classifier = nn.Linear(d_model, num_labels)

        self.encoder = Encoder(EncoderLayer(d_model, attn, ff, dropout), num_layers)
        self.src_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), position)

    def prepare_mask(self, seq):
        seq_mask = (seq.data != self.eos_idx) & (seq.data != self.pad_idx)
        seq_mask[:, 0] = 1  # bos
        seq_mask = seq_mask.unsqueeze(-2)
        return seq_mask

    def forward(self, src):
        src_mask = self.prepare_mask(src)
        feats = self.encoder(self.src_embed(src), src_mask)
        #print(feats.shape)#torch.Size([16, 60, 512])
        pooled_output = feats[:, 0, :]
        #print(pooled_output.shape)#torch.Size([16, 512])得到的[cls]
        labels = self.classifier(pooled_output)
        return feats, pooled_output, labels


class MHA_FF(nn.Module):
    def __init__(self, d_model, d_ff, h, dropout=0.1):
        super(MHA_FF, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model)
        self.sublayer = SublayerConnection(d_model, dropout)

    def forward(self, x, feats,mask=None):
        #print(mask)
        x = self.sublayer(x, lambda x: self.self_attn(x, feats, feats))#通过多头注意力进行获得新的知识并通过归一化操作进行加入新知识
        return x



class MHA_FF1(nn.Module):
    def __init__(self, d_model, d_ff, h, dropout=0.1):
        super(MHA_FF1, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model)
        self.sublayer = SublayerConnection(d_model, dropout)

    def forward(self, x, feats,mask=None):
        #print(mask)
        x = self.sublayer(x, lambda x: self.self_attn(x, feats, feats))#通过多头注意力进行获得新的知识并通过归一化操作进行加入新知识
        return x



class MHA_FF2(nn.Module):
    def __init__(self, d_model, d_ff, h, dropout=0.1):
        super(MHA_FF2, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model)
        self.sublayer = SublayerConnection(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, feats,mask=None):
        #print(mask)
        #x= self.dropout(self.self_attn(x, feats, feats))
        x = self.sublayer(x, lambda x: self.self_attn(x, feats, feats))#通过多头注意力进行获得新的知识并通过归一化操作进行加入新知识
        return x