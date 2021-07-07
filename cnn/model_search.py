import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


class MixedOp(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()

    op = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, (1,1), stride=(1, 1), padding="same", bias=False),
      nn.ReLU(inplace=False)
    )
    self._ops.append(op)

    op = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, (3,3), stride=(1, 1), padding="same", bias=False),
      nn.ReLU(inplace=False)
    )
    self._ops.append(op)

    op = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, (5,5), stride=(1, 1), padding="same", bias=False),
      nn.ReLU(inplace=False)
    )
    self._ops.append(op)

  def forward(self, x, alphas):
    no_ops = len(self._ops)
    return sum([alphas[0,i] * self._ops[i](x) for i in range(no_ops)])
    #return sum(alpha * op(x) for alpha, op in zip(alphas, self._ops))

class Cell(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(Cell, self).__init__()
    self.reduction = False
    self._ops = nn.ModuleList()
    op = MixedOp(in_channels, out_channels)
    self._ops.append(op)

  def forward(self, x, alphas):
    return self._ops[0](x, alphas)

class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, use_cuda=False, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self.use_cuda = use_cuda
    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev, C_curr = C_curr, C
    self.cells = nn.ModuleList()
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 1
      cell = Cell(C_prev, C_curr)
      self.cells += [cell]
      C_prev = C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion)
    if self.use_cuda:
      model_new = model_new.cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = self.stem(input)
    for i, cell in enumerate(self.cells):
      alphas = F.softmax(self.alphas_normal, dim=-1)
      s0 = cell(s0, alphas)
    out = self.global_pooling(s0)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = 1
    num_ops = 3

    alphas_normal = 1e-3*torch.randn(k, num_ops)
    if self.use_cuda:
      alphas_normal = alphas_normal.cuda()
    self.alphas_normal = Variable(alphas_normal, requires_grad=True)

    self._arch_parameters = [
      self.alphas_normal,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  # def genotype(self):

  #   def _parse(alphas):
  #     gene = []
  #     n = 2
  #     start = 0
  #     for i in range(self._steps):
  #       end = start + n
  #       W = alphas[start:end].copy()
  #       edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
  #       for j in edges:
  #         k_best = None
  #         for k in range(len(W[j])):
  #           if k != PRIMITIVES.index('none'):
  #             if k_best is None or W[j][k] > W[j][k_best]:
  #               k_best = k
  #         gene.append((PRIMITIVES[k_best], j))
  #       start = end
  #       n += 1
  #     return gene

  #   gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
  #   gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

  #   concat = range(2+self._steps-self._multiplier, self._steps+2)
  #   genotype = Genotype(
  #     normal=gene_normal, normal_concat=concat,
  #     reduce=gene_reduce, reduce_concat=concat
  #   )
  #   return genotype

