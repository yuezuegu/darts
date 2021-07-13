import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


class VariableConv(nn.Module):
  def __init__(self, in_channels, out_channels, window_size, channel_scale_factor):
    super(VariableConv, self).__init__()
    self.op = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, window_size, stride=(1, 1), padding="same", bias=False),
      nn.ReLU(inplace=False)
    )

    self.channel_masks = []
    for i in range(int(out_channels/channel_scale_factor)):
      mask = [0 for j in range(out_channels)]
      mask[0:int(channel_scale_factor*(i+1))] = [1 for j in range(int(channel_scale_factor*(i+1)))]
      self.channel_masks.append(mask)

    self.channel_masks = torch.Tensor(self.channel_masks)

  def forward(self, x, alphas):
    out = self.op(x)
    masks = torch.sum(torch.multiply(alphas, self.channel_masks.transpose(0,1)), dim=1).reshape(1,-1,1,1)
    return torch.multiply(masks, out)
    
class MixedOp(nn.Module):
  def __init__(self, in_channels, out_channels, channel_scale_factor):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()

    # op = nn.Sequential(
    #   nn.Conv2d(in_channels, out_channels, (1,1), stride=(1, 1), padding="same", bias=False),
    #   nn.ReLU(inplace=False)
    # )
    op = VariableConv(in_channels, out_channels, (1,1), channel_scale_factor)
    self._ops.append(op)

    # op = nn.Sequential(
    #   nn.Conv2d(in_channels, out_channels, (3,3), stride=(1, 1), padding="same", bias=False),
    #   nn.ReLU(inplace=False)
    # )
    op = VariableConv(in_channels, out_channels, (3,3), channel_scale_factor)
    self._ops.append(op)

    # op = nn.Sequential(
    #   nn.Conv2d(in_channels, out_channels, (5,5), stride=(1, 1), padding="same", bias=False),
    #   nn.ReLU(inplace=False)
    # )
    op = VariableConv(in_channels, out_channels, (5,5), channel_scale_factor)
    self._ops.append(op)

  def forward(self, x, alphas_normal, alphas_channels):
    no_ops = len(self._ops)
    return sum([alphas_normal[0,i] * self._ops[i](x, alphas_channels) for i in range(no_ops)])

class Cell(nn.Module):
  def __init__(self, in_channels, out_channels, channel_scale_factor):
    super(Cell, self).__init__()
    self.reduction = False
    self._ops = nn.ModuleList()
    op = MixedOp(in_channels, out_channels, channel_scale_factor)
    self._ops.append(op)

  def forward(self, x, alphas_normal, alphas_channels):
    return self._ops[0](x, alphas_normal, alphas_channels)

class Network(nn.Module):

  def __init__(self, init_filters, num_classes, layers, criterion, use_cuda=False, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._init_filters = init_filters
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self.use_cuda = use_cuda
    no_filters = stem_multiplier*init_filters
    
    channel_scale_factor = stem_multiplier
    self.stem = VariableConv(3, no_filters, (3,3), channel_scale_factor)

    # self.stem = nn.Sequential(
    #   nn.Conv2d(3, no_filters, 3, padding=1, bias=False),
    #   nn.BatchNorm2d(no_filters)
    # )

    no_channels, no_filters = no_filters, init_filters
    self.cells = nn.ModuleList()
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        no_filters *= 1
      channel_scale_factor = int(no_filters / init_filters)
      cell = Cell(no_channels, no_filters, channel_scale_factor)
      self.cells += [cell]
      no_channels = no_filters

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(no_channels, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._init_filters, self._num_classes, self._layers, self._criterion)
    if self.use_cuda:
      model_new = model_new.cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = self.stem(input, self.alphas_channels)
    for i, cell in enumerate(self.cells):
      alphas_normal = F.softmax(self.alphas_normal, dim=-1)
      alphas_channels = F.softmax(self.alphas_channels, dim=-1)
      s0 = cell(s0, alphas_normal, alphas_channels)
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

    self.alphas_channels = 1e-3*torch.randn(self._init_filters)
    self.alphas_channels = Variable(self.alphas_channels, requires_grad=True)

    self._arch_parameters = [
      self.alphas_normal,
    ]

  def arch_parameters(self):
    return self._arch_parameters

