from ast import Tuple
import os
import sys
import time
import math
import argparse
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


class Charset(Dataset):
  def __init__(self, names: List[str], maxLen: int):
    self.names = names
    self.maxLen = maxLen
    self.chars = sorted(list(set("".join(names))))
    self.c2i = {ch: i for i, ch in enumerate(self.chars)}
    self.i2c = {i: ch for i, ch in enumerate(self.chars)}

  def __len__(self):
    return len(self.names)

  def contains(self, name):
    return name in self.names

  def get_vocab_size(self):
    return len(self.chars)

  def encode(self, name):
    ix = torch.tensor([self.c2i[ch] for ch in name])
    return ix

  def decode(self, ix):
    return "".join(self.i2c[i] for i in ix)

  def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
    word = self.names[index]
    ix = self.encode(word)
    x = torch.zeros(self.maxLen + 1, dtype=torch.long)
    y = torch.zeros(self.maxLen + 1, dtype=torch.long)
    x[1 : 1 + len(ix)] = ix
    y[len(ix) + 1 :] = -1
    return x, y


def create_datasets(file: str) -> tuple[Charset, Charset]:
  with open(file) as f:
    names = f.read().splitlines()
  names = [name.lower().strip() for name in names]
  names = [name for name in names if name]
  # Use functional python style, "".join(names) -> set -> list
  chars = sorted(list(set("".join(names))))
  maxLen = max(len(name) for name in names)
  testSetSize = len(names) // 10
  rp = torch.randperm(len(names))
  trainNames = [names[i] for i in rp[testSetSize:]]
  testNames = [names[i] for i in rp[:testSetSize]]
  trainSet = Charset(trainNames, maxLen)
  testSet = Charset(testNames, maxLen)
  return trainSet, testSet


### PyTorch implementation
class Bigram(nn.Module):
  def __init__(self, vocab_size: int):
    super().__init__()
    n = vocab_size
    self.logits = nn.Parameter(torch.randn((n, n)))

  def forward(self, idx, targets=None):
    logits = self.logits[idx]
    loss = None
    if targets is not None:
      loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
      )
    return logits, loss


def pytorch(args, testData, trainData):
  model = Bigram(vocab_size=27)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Makemore")
  # Args: Framework (PyTorch, TF, JAX, MLX)
  parser.add_argument("--framework", type=str, default="PyTorch")
  # Args: Model (Bigram, MLP)
  parser.add_argument("--model", type=str, default="Bigram")
  # Backend (CPU, CUDA, Metal)
  parser.add_argument("--backend", type=str, default="cpu")
  # Args: File
  parser.add_argument("--file", type=str, default="names.txt")
  args = parser.parse_args()
