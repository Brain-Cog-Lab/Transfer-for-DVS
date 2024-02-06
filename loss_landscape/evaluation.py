"""
    The calculation to be performed at each point (modified model), evaluating
    the loss value, accuracy and eigen values of the hessian matrix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd.variable import Variable
from braincog.utils import *
from contextlib import suppress

def accuracy(output, target, topk=(1,)):
    """Compute the top1 and top5 accuracy
    """
    maxk = max(topk)
    batch_size = target.size(0)
    # Return the k largest elements of the given input tensor
    # along a given dimension -> N * k
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def eval_loss(net, criterion, loader, use_cuda=False):
    """
    Evaluate the loss value for a given 'net' on the dataset provided by the loader.

    Args:
        net: the neural net model
        criterion: loss function
        loader: dataloader
        use_cuda: use cuda or not
    Returns:
        loss value and accuracy
    """
    correct = 0
    total_loss = 0
    total = 0 # number of samples
    num_batch = len(loader)

    if use_cuda:
        net.cuda()
    net.eval()

    with torch.no_grad():
        if isinstance(criterion, nn.CrossEntropyLoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                inputs = Variable(inputs)
                targets = Variable(targets)
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()*batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets).sum().item()

        elif isinstance(criterion, nn.MSELoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                inputs = Variable(inputs)

                one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
                one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
                one_hot_targets = one_hot_targets.float()
                one_hot_targets = Variable(one_hot_targets)
                if use_cuda:
                    inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
                outputs = F.softmax(net(inputs))
                loss = criterion(outputs, one_hot_targets)
                total_loss += loss.item()*batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.cpu().eq(targets).sum().item()

    return total_loss/total, 100.*correct/total



def validate(model, loader, loss_fn, args, amp_autocast=suppress):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    closses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    model.cuda()
    model.eval()
    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(loader):
            # inputs = inputs.type(torch.float64)
            last_batch = batch_idx == last_idx
            if not args.prefetcher or args.target_dataset != 'imnet':
                inputs = inputs.type(torch.FloatTensor).cuda()
                target = target.cuda()
            if args.channels_last:
                inputs = inputs.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                if args.model == "Transfer_VGG_SNN":
                    _, _, output_rbg, output_dvs = model(inputs, inputs)
                    output = sum(output_dvs) / len(output_dvs)
                else:
                    output = model(inputs)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            closs = torch.tensor([0.], device=loss.device)

            if not args.distributed:
                spike_rate_avg_layer = model.get_fire_rate().tolist()
                threshold = model.get_threshold()
                threshold_str = ['{:.3f}'.format(i) for i in threshold]
                spike_rate_avg_layer_str = ['{:.3f}'.format(i) for i in spike_rate_avg_layer]
                tot_spike = model.get_tot_spike()

            reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), inputs.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))
            closses_m.update(closs.item(), inputs.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
    return losses_m.avg, top1_m.avg


