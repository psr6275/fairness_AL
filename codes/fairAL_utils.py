import torch
import torch.nn as nn
import autograd_hacks
import numpy as np
from torch.utils.data import DataLoader
from load_data import *
import random

class SmoothCrossEntropy(nn.Module):
    def __init__(self):
        super(SmoothCrossEntropy,self).__init__()
    def forward(self,output, target):
        logprobs = torch.nn.functional.log_softmax (output, dim = 1)
        return  -(target * logprobs).sum() / output.shape[0]


def select_examples(clf,select_loader,criterion, grad_z, device, args):
    aa = torch.topk(compute_gradsim(clf, select_loader, criterion, grad_z, device, args),args.AL_batch)
    ses = []
    for ts in select_loader.dataset.tensors:
        ses.append(ts[aa[1]])
    return ses,aa[1]
def select_entexamples(clf,select_loader, grad_z, device, args):
    aa = torch.topk(compute_entropysim(clf, select_loader, grad_z, device, args),args.AL_batch)
    ses = []
    for ts in select_loader.dataset.tensors:
        ses.append(ts[aa[1]])
    return ses,aa[1]

def select_random(clf,select_loader,device, nsample = 32):
    aa = random.sample(list(range(select_loader.dataset.tensors[0].size(0))),nsample)
    ses = []
    for ts in select_loader.dataset.tensors:
        ses.append(ts[aa])
    return ses,torch.tensor(aa)

def group_grad(clf, dldic, criterion, device):
    grads={}
    clf.eval()
    for did in dldic.keys():
        print(did)
        grads[did] = cal_meangrad(clf, dldic[did], criterion, device)
    return grads

def cal_meangrad(clf, dataloader, criterion, device,normalize=True):
    
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
    if normalize:
        grads /= prgrad_n
    grads = grads.detach().cpu()
    return grads

def compute_gradnorm(clf, select_loader, criterion, device,args):
    clf.eval()
    autograd_hacks.add_hooks(clf)
    norms = []
    for i, (x,_,_) in enumerate(select_loader):
        x = x.to(device)
        for j in range(args.num_classes):
            y = torch.ones(x.size(0),1).to(device)*j
            clf.zero_grad()
            clear_backprops(clf)
            outs = clf(x)
            count_backprops(clf)
            criterion(outs,y).backward()
            remove_backprops(clf)
            autograd_hacks.compute_grad1(clf)
            tmp = []
            for param in clf.parameters():
                tmp.append(param.grad1.reshape(x.size(0),-1))
            if args.num_classes <= 2:
                if j==0:
                    grad_t = torch.cat(tmp,dim=1).detach().cpu()
                else:
                    grad_t += torch.cat(tmp,dim=1).detach().cpu()
            else:
                if j==0:
                    grad_t = torch.cat(tmp,dim=1).detach().cpu()*torch.softmax(outs.detach().cpu())[:,j]
                else:
                    grad_t += torch.cat(tmp,dim=1).detach().cpu()*torch.softmax(outs.detach().cpu())[:,j]
#         print(grad_t)
#         print(grad_z)
        norms.append(torch.norm(grad_t,grad_t))
    return torch.cat(norms).detach().cpu()

def compute_gradsim(clf, select_loader, criterion, grad_z,device,args):
    clf.eval()
    autograd_hacks.add_hooks(clf)
    sims = []
    for i, (x,_,_) in enumerate(select_loader):
        x = x.to(device)
        for j in range(args.num_classes):        
            y = torch.ones(x.size(0),1).to(device)*j
            clf.zero_grad()
            clear_backprops(clf)
            outs = clf(x)
            count_backprops(clf)
            criterion(outs,y).backward()
            remove_backprops(clf)
            autograd_hacks.compute_grad1(clf)
            tmp = []
            for k, param in enumerate(clf.parameters()):
                tmp.append(param.grad1.reshape(x.size(0),-1))            
            if args.num_classes<=2:
                if j==0:
                    grad_t = torch.cat(tmp,dim=1).detach().cpu()*outs.detach().cpu()
                else:
                    grad_t += torch.cat(tmp,dim=1).detach().cpu()*outs.detach().cpu()
            else:
                if j==0:
                    grad_t = torch.cat(tmp,dim=1).detach().cpu()*torch.softmax(outs.detach().cpu())[:,j]
                else:
                    grad_t += torch.cat(tmp,dim=1).detach().cpu()*torch.softmax(outs.detach().cpu())[:,j]
#         print(grad_t)
#         print(grad_z)
        sims.append(torch.matmul(grad_t,grad_z.to(device)))
    return torch.cat(sims).detach().cpu()
def compute_gradsim_binary(clf, select_loader, criterion, grad_z,device,args):
    clf.eval()
    autograd_hacks.add_hooks(clf)
    sims = []
    assert args.num_classes ==2
    for i, (x,_,_) in enumerate(select_loader):
        x = x.to(device)
#         y = torch.ones(x.size(0),1).to(device)*j
        clf.zero_grad()
        clear_backprops(clf)
        outs = clf(x)
        outs_ = torch.new_tensor(outs,requires_grad = False)
        count_backprops(clf)
        criterion(outs,outs_).backward()
        remove_backprops(clf)
        autograd_hacks.compute_grad1(clf)
        tmp = []
        for k, param in enumerate(clf.parameters()):
            tmp.append(param.grad1.reshape(x.size(0),-1))            
        grad_t = torch.cat(tmp,dim=1)

#         print(grad_t)
#         print(grad_z)
        sims.append(torch.matmul(grad_t,grad_z.to(device)))
    return torch.cat(sims).detach().cpu()

def auto_grad_compute(clf,x,y,clf_criterion,device):
    autograd_hacks.add_hooks(clf)
    clf.zero_grad()
    clear_backprops(clf)
    outs = clf(x)
    count_backprops(clf)
    clf_criterion(outs,y).backward()
    remove_backprops(clf)
    autograd_hacks.compute_grad1(clf)
    tmp = []
    for k, param in enumerate(clf.parameters()):
        tmp.append(param.grad1.reshape(x.size(0),-1))       
    grad_t = torch.cat(tmp,dim=1).detach().cpu()
    return grad_t

def auto_grad_entropy_compute(clf,x,device):
    autograd_hacks.add_hooks(clf)
    clf.zero_grad()
    clear_backprops(clf)
    # this part is needed....
    torch.autograd.set_detect_anomaly(True)
    outs = clf(x)
    count_backprops(clf)
    # we need to set gradient
    outs.backward(gradient = torch.ones(outs.size()).to(device))
    remove_backprops(clf)
    autograd_hacks.compute_grad1(clf)    
    tmp = []
    for k, param in enumerate(clf.parameters()):
        tmp.append(param.grad1.reshape(x.size(0),-1))       
    grad_t = torch.cat(tmp,dim=1).detach().cpu()
    logits = clf.network(x).detach().cpu()
    return -grad_t*logits

def compute_entropysim(clf, select_loader,grad_z,device,args):
    clf.eval()
    autograd_hacks.add_hooks(clf)
    sims = []
    if args.num_classes <=2:
        criterion = nn.BCELoss()
    else:
        criterion = SmoothCrossEntropy()
        
    for i, (x,_,_) in enumerate(select_loader):
        x = x.to(device)
        grad_t = auto_grad_entropy_compute(clf,x,device)        
#         print(grad_t)
#         print(grad_z)
        sims.append(torch.matmul(grad_t,grad_z.detach().cpu()))
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