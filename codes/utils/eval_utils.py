import numpy as np
import torch

class AverageVarMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val=0
        self.sum2 = 0
        self.avg=0
        self.sum=0
        self.count=0

    def update(self,val,n=1):
        self.val = val
        self.sum2 += (val**2)*n
        self.sum +=val*n
        self.count +=n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    '''Compute the top1 and top k error'''
    maxk = max(topk)
    batch_size = target.size(0)
    _,pred = output.topk(maxk,1,True,True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul(100.0 / batch_size))
    return res

def accuracy_b(output, target, thr = 0.5):
    '''Compute the top1 and top k error'''
    
    batch_size = target.size(0)
    pred = torch.tensor(output>thr,dtype = torch.float32)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

   
    correct_k = correct.view(-1).float().sum(0)
    res=correct_k.mul(100.0 / batch_size)
    
    return res