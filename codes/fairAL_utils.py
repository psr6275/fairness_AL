import torch
import torch.nn as nn
import autograd_hacks
import numpy as np
from torch.utils.data import DataLoader
from load_data import *

def select_examples(clf,select_loader,criterion, grad_z, device, nsample = 32):
    aa = torch.topk(compute_gradsim(clf, select_loader, criterion, grad_z, device),nsample)
    ses = []
    for ts in select_loader.dataset.tensors:
        ses.append(ts[aa[1]])
    return ses,aa[1]



def group_grad(clf, dldic, criterion, device):
    grads={}
    clf.eval()
    for did in dldic.keys():
        print(did)
        grads[did] = cal_meangrad(clf, dldic[did], criterion, device)
    return grads
def cal_meangrad(clf, dataloader, criterion, device):
    
    for i,(x,y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        clf.zero_grad()
        outs = clf(x)
        criterion(outs,y).backward()
        tmp = []
        for param in clf.parameters():
            tmp.append(param.grad.flatten())
        grads_t = torch.cat(tmp)
        if i==0:
            grads = grads_t
        else:
            grads += grads_t
    prgrad_n = torch.norm(grads)
    grads /= prgrad_n
    grads = grads.detach().cpu()
    return grads

def compute_gradsim(clf, select_loader, criterion, grad_z,device):
    clf.eval()
    autograd_hacks.add_hooks(clf)
    sims = []
    for i, (x,y,_) in enumerate(select_loader):
        x = x.to(device)
        y = y.to(device)
        clf.zero_grad()
        clear_backprops(clf)
        outs = clf(x)
        count_backprops(clf)
        criterion(outs,y).backward()
        remove_backprops(clf)
        autograd_hacks.compute_grad1(clf)
        tmp = []
        for j, param in enumerate(clf.parameters()):
            tmp.append(param.grad1.reshape(x.size(0),-1))
        grad_t = torch.cat(tmp,dim=1).cuda()
#         print(grad_t)
#         print(grad_z)
        sims.append(torch.matmul(grad_t,grad_z.cuda()))
    return torch.cat(sims).detach().cpu()

def clear_backprops(model: nn.Module) -> None:
    """Delete layer.backprops_list in every layer."""
    for layer in model.modules():
        if hasattr(layer, 'backprops_list'):
            del layer.backprops_list

def remove_backprops(model: nn.Module):
    """Delete layer.backprops_list in every layer."""
    for layer in model.modules():
        if hasattr(layer, 'backprops_list'):
            del layer.backprops_list[:-1]

def count_backprops(model: nn.Module) -> None:
    """Delete layer.backprops_list in every layer."""
    for layer in model.modules():
        if hasattr(layer, 'backprops_list'):
            print(len(layer.backprops_list))