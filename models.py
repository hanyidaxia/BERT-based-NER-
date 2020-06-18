import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import settings
from settings import *


class BertBase(nn.Module):
  def __init__(self, bert, classes):
    super().__init__()
    self.bert = bert
    d_model = bert.embeddings.word_embeddings.weight.size(1)
    self.dense = nn.Linear(d_model, classes)
    self.conv1 = nn.Conv1d(128, 128, 5)
    self.conv2 = nn.Conv1d(128, 126, 3, stride = 3)
    # self.conv3= nn.Conv1d(512, 1024, kernel_size=3, stri
    self.fc1 = nn.Linear(254, 4)


  def forward(self, inp, inp_mask):

    output = self.bert(inp, inp_mask)[0]
    # print("1st output  " + str(np.shape(output)))

    output = F.relu(self.conv1(output))
    # print("2nd output  " + str(np.shape(output)))

    output = F.relu(self.conv2(output))

    output = F.relu(self.fc1(output))

    # print("final output  " + str(np.shape(output)))
    #
    # output = self.dense(output[:, 1:-1])
    # print("2nd output  " + str(np.shape(output)))

    # output = self.dropout(output)
    # output = F.relu(self.conv2(output))
    return output

  def init_parameters(self):
    for p in self.dense.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)
