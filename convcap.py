import sys

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

#Layers adapted for captioning from https://arxiv.org/abs/1705.03122
def Conv1d(in_channels, out_channels, kernel_size, padding, dropout=0):
    m = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
    std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, 0.1)
    return m

def Linear(in_features, out_features, dropout=0.):
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)

class AttentionLayer(nn.Module):
  def __init__(self, conv_channels, embed_dim):
    super(AttentionLayer, self).__init__()
    self.in_projection = Linear(conv_channels, embed_dim)
    self.out_projection = Linear(embed_dim, conv_channels)
    self.bmm = torch.bmm

  def forward(self, x, wordemb, imgsfeats):
    residual = x

    x = (self.in_projection(x) + wordemb) * math.sqrt(0.5)

    b, c, f_h, f_w = imgsfeats.size()
    y = imgsfeats.view(b, c, f_h*f_w)

    x = self.bmm(x, y)

    sz = x.size()
    x = F.softmax(x.view(sz[0] * sz[1], sz[2]))
    x = x.view(sz)
    attn_scores = x

    y = y.permute(0, 2, 1)

    x = self.bmm(x, y)

    s = y.size(1)
    x = x * (s * math.sqrt(1.0 / s))

    x = (self.out_projection(x) + residual) * math.sqrt(0.5)

    return x, attn_scores

class convcap(nn.Module):
  
  def __init__(self, num_wordclass, num_layers=1, is_attention=True, nfeats=512, dropout=.1):
    super(convcap, self).__init__()
    self.nimgfeats = 4096
    self.is_attention = is_attention
    self.nfeats = nfeats
    self.dropout = dropout 
 
    self.emb_0 = Embedding(num_wordclass, nfeats, padding_idx=0)
    self.emb_1 = Linear(nfeats, nfeats, dropout=dropout)

    self.imgproj = Linear(self.nimgfeats, self.nfeats, dropout=dropout)
    self.resproj = Linear(nfeats*2, self.nfeats, dropout=dropout)

    n_in = 2*self.nfeats 
    n_out = self.nfeats
    self.n_layers = num_layers
    self.convs = nn.ModuleList()
    self.attention = nn.ModuleList()
    self.kernel_size = 5
    self.pad = self.kernel_size - 1
    for i in range(self.n_layers):
      self.convs.append(Conv1d(n_in, 2*n_out, self.kernel_size, self.pad, dropout))
      if(self.is_attention):
        self.attention.append(AttentionLayer(n_out, nfeats))
      n_in = n_out

    self.classifier_0 = Linear(self.nfeats, (nfeats // 2))
    self.classifier_1 = Linear((nfeats // 2), num_wordclass, dropout=dropout)

  def forward(self, imgsfeats, imgsfc7, wordclass):

    attn_buffer = None
    wordemb = self.emb_0(wordclass)
    wordemb = self.emb_1(wordemb)
    x = wordemb.transpose(2, 1)   
    batchsize, wordembdim, maxtokens = x.size()

    y = F.relu(self.imgproj(imgsfc7))
    y = y.unsqueeze(2).expand(batchsize, self.nfeats, maxtokens)
    x = torch.cat([x, y], 1)

    for i, conv in enumerate(self.convs):
      
      if(i == 0):
        x = x.transpose(2, 1)
        residual = self.resproj(x)
        residual = residual.transpose(2, 1)
        x = x.transpose(2, 1)
      else:
        residual = x

      x = F.dropout(x, p=self.dropout, training=self.training)

      x = conv(x)
      x = x[:,:,:-self.pad]

      x = F.glu(x, dim=1)

      if(self.is_attention):
        attn = self.attention[i]
        x = x.transpose(2, 1)
        x, attn_buffer = attn(x, wordemb, imgsfeats)
        x = x.transpose(2, 1)
    
      x = (x+residual)*math.sqrt(.5)

    x = x.transpose(2, 1)
  
    x = self.classifier_0(x)
    x = F.dropout(x, p=self.dropout, training=self.training)
    x = self.classifier_1(x)

    x = x.transpose(2, 1)

    return x, attn_buffer
