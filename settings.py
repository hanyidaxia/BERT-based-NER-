import sys
import os
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


USE_CUDA = True # Change according to args
LR = 0.0005 # Learning rate
DROPOUT = 0.5
L2_NORM = 0.0005
SHELL_OUT_FILE=sys.stdout


def clones(module, no_of_copies):
  """Produce no_of_copies identical layers."""
  return nn.ModuleList([copy.deepcopy(module) for _ in range(no_of_copies)])

def convert_to_unicode(text):
  if isinstance(text, str):
    return text
  elif isinstance(text, bytes):
    return text.decode('utf-8')
  else:
    raise ValueError('Unsupported string type: %s' % (type(text)))
