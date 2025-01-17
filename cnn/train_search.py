import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from architect import Architect

CIFAR_CLASSES = 10

def main():
	parser = argparse.ArgumentParser("cifar")
	parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
	parser.add_argument('--batch_size', type=int, default=64, help='batch size')
	parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
	parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
	parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
	parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
	parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
	parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
	parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
	parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
	parser.add_argument('--layers', type=int, default=8, help='total number of layers')
	parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
	parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
	parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
	parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
	parser.add_argument('--save', type=str, default='EXP', help='experiment name')
	parser.add_argument('--seed', type=int, default=2, help='random seed')
	parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
	parser.add_argument('--train_portion', type=float, default=0.9, help='portion of training data')
	parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
	parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
	parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
	parser.add_argument('--init_tau', type=float, default=5., help='Initial temperature value for Gumbel softmax')
	parser.add_argument('--tau_anneal_rate', type=float, default=0.956, help='Exponential anneal rate for temperature value for Gumbel softmax')
	args = parser.parse_args()

	os.system("mkdir -p experiments")
	args.save = 'experiments/search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
	utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

	log_format = '%(asctime)s %(message)s'
	logging.basicConfig(stream=sys.stdout, level=logging.INFO,
			format=log_format, datefmt='%m/%d %I:%M:%S %p')
	fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
	fh.setFormatter(logging.Formatter(log_format))
	logging.getLogger().addHandler(fh)

	if torch.cuda.is_available():
		logging.info('gpu device available')
		logging.info('gpu device = %d' % args.gpu)
		device = torch.device('cuda:0')
		torch.cuda.set_device(args.gpu)
		cudnn.benchmark = True
		cudnn.enabled=True
		use_cuda = True
	else:
		device = torch.device('cpu')
		logging.info('using cpu')
		use_cuda = False

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if use_cuda:
		torch.cuda.manual_seed(args.seed)

	logging.info("args = %s", args)

	criterion = nn.CrossEntropyLoss()
	if use_cuda:
		criterion = criterion.cuda()
	model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, args.init_tau, use_cuda)
	if use_cuda:
		model = model.cuda()
	logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

	weight_optimizer = torch.optim.SGD(
			model.parameters(),
			args.learning_rate,
			momentum=args.momentum,
			weight_decay=args.weight_decay)

	alpha_optimizer = torch.optim.Adam(
		model.arch_parameters(), 
		lr=args.arch_learning_rate, 
		betas=(0.5, 0.999), 
		weight_decay=args.arch_weight_decay)

	train_transform, valid_transform = utils._data_transforms_cifar10(args)
	train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

	num_train = len(train_data)
	indices = list(range(num_train))
	split = int(np.floor(args.train_portion * num_train))

	train_queue = torch.utils.data.DataLoader(
			train_data, batch_size=args.batch_size,
			sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
			pin_memory=True, num_workers=2)

	valid_queue = torch.utils.data.DataLoader(
			train_data, batch_size=args.batch_size,
			sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
			pin_memory=True, num_workers=2)

	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
				weight_optimizer, float(args.epochs), eta_min=args.learning_rate_min)

	tau = args.init_tau

	architect = Architect(model, args, use_cuda)

	for epoch in range(args.epochs):
		print(F.softmax(model.alphas_normal, dim=-1))
		print(F.softmax(model.alphas_channels, dim=-1))

		lr = scheduler.get_last_lr()[0]
		logging.info('epoch %d lr %e', epoch, lr)

		# training
		train_acc, train_obj = train_weights(train_queue, model, criterion, weight_optimizer, epoch_early_stop=0.2, args=args)
		logging.info('train_acc %f', train_acc)

		train_acc, train_obj = train_alphas(train_queue, model, architect, criterion, alpha_optimizer, tau, epoch_early_stop=0.2, args=args)
		logging.info('train_acc %f', train_acc)

		# validation
		valid_acc, valid_obj = infer(valid_queue, model, criterion, args)
		logging.info('valid_acc %f', valid_acc)

		# update lr
		scheduler.step()
		
		model.update_tau(args.tau_anneal_rate)

		utils.save(model, os.path.join(args.save, 'weights.pt'))


def train_weights(train_queue, model, criterion, weight_optimizer, epoch_early_stop, args, use_cuda=False):
	objs = utils.AvgrageMeter()
	top1 = utils.AvgrageMeter()
	top5 = utils.AvgrageMeter()

	no_iterations = len(train_queue)
	for step, (input, target) in enumerate(train_queue):
		model.train()
		n = input.size(0)

		input = Variable(input, requires_grad=False)
		if use_cuda:
			input = input.cuda()
		target = Variable(target, requires_grad=False)
		if use_cuda:
			target = target.cuda(non_blocking=True)

		weight_optimizer.zero_grad()
		logits = model(input)
		loss = criterion(logits, target)

		loss.backward()
		nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
		weight_optimizer.step()

		prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
		objs.update(loss.item(), n)
		top1.update(prec1.item(), n)
		top5.update(prec5.item(), n)

		if step % args.report_freq == 0:
			logging.info('train weights: %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

		if step > no_iterations*epoch_early_stop:
			break

	return top1.avg, objs.avg

def train_alphas(train_queue, model, architect, criterion, alpha_optimizer, tau, epoch_early_stop, args, use_cuda=False):
	objs = utils.AvgrageMeter()
	top1 = utils.AvgrageMeter()
	top5 = utils.AvgrageMeter()

	no_iterations = len(train_queue)
	for step, (input, target) in enumerate(train_queue):
		model.train()
		n = input.size(0)

		input = Variable(input, requires_grad=False)
		if use_cuda:
			input = input.cuda()
		target = Variable(target, requires_grad=False)
		if use_cuda:
			target = target.cuda(non_blocking=True)

		architect.step(input, target, alpha_optimizer)

		logits = model(input)
		loss = criterion(logits, target)

		prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
		objs.update(loss.item(), n)
		top1.update(prec1.item(), n)
		top5.update(prec5.item(), n)

		if step % args.report_freq == 0:
			logging.info('training alphas: %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

		if step > no_iterations*epoch_early_stop:
			break

	return top1.avg, objs.avg

def infer(valid_queue, model, criterion, args, use_cuda=False):
	objs = utils.AvgrageMeter()
	top1 = utils.AvgrageMeter()
	top5 = utils.AvgrageMeter()
	model.eval()

	for step, (input, target) in enumerate(valid_queue):
		input = Variable(input, requires_grad=False)
		if use_cuda:
			input = input.cuda()
		target = Variable(target, requires_grad=False)
		if use_cuda:
			target = target.cuda(non_blocking=True)

		logits = model(input)
		loss = criterion(logits, target)

		prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
		n = input.size(0)
		objs.update(loss.item(), n)
		top1.update(prec1.item(), n)
		top5.update(prec5.item(), n)

		if step % args.report_freq == 0:
			logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

	return top1.avg, objs.avg


if __name__ == '__main__':
	main() 

