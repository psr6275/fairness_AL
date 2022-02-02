import torch
import torch.nn as nn

from . import autograd_hacks


_supported_layers = ['Linear','Conv2d']

def obtain_param_names(clf):
    param_names = []
    itr = 1
    for layer in clf.modules():
        lname = _layer_type(layer)
        if lname in _supported_layers:
            param_names.append(lname+str(itr)+'_w')
            if layer.bias is not None:
                param_names.append(lname+str(itr)+'_b')
            itr += 1
    return param_names
def _layer_type(layer: nn.Module) -> str:
    return layer.__class__.__name__            

def compute_mean_grad_with_label(clf, x,y, criterion, device, sel_idxs):
    clf.to(device).eval()
    x,y = x.to(device), y.to(device)
    clf.zero_grad()
    outs = clf(x)
    criterion(outs,y).backward()
    tmp = []
    for k,param in enumerate(clf.parameters()):
        if k in sel_idxs: 
            tmp.append(param.grad.flatten().detach().cpu())
    return tmp

def compute_mean_grad_wo_label(clf,x,criterion,device, sel_idxs):
    clf.to(device).eval()
    x = x.to(device)
    clf.zero_grad()
    outs = clf(x)
    criterion(outs).backward()
    tmp = []
    for k,param in enumerate(clf.parameters()):
        if k in sel_idxs: 
            tmp.append(param.grad.flatten().detach().cpu())
    return tmp

def compute_mean_grad_dic(clf,dataloader,device,criterion,use_label = False,param_names=None,
                          last_layer = True,sel_idxs=None, normalize=True, dictionary=False):
    clf.to(device)
    ## criterion should have sum type
    assert(criterion.reduction =='sum')
    ## all gradients are obtained when sel_idxs = None
    if param_names is None:
        param_names = obtain_param_names(clf)
        
    if sel_idxs is not None:
        assert(sel_layers in param_names)

    grad_dic = {}
    
    if last_layer:
        sel_idxs=list(range(len(param_names))[-2:])            
    elif sel_idxs is None:
        ## last_layer is False and sel_idxs is None
        sel_idxs = list(range(len(param_names)))
    else:
        ## last_layer is False and sel_idxs is not None
        pass
    
    for i in sel_idxs:
        grad_dic[param_names[i]] = []
    
    
    ## calculate gradient for dataloader
    for i,dt in enumerate(dataloader):
        if use_label:
            grads = compute_mean_grad_with_label(clf,dt[0],dt[1],criterion,device,sel_idxs)
        else:
            grads = compute_mean_grad_wo_label(clf,dt[0],criterion,device,sel_idxs)
        
        for k,j in enumerate(sel_idxs):
            if i==0:
                grad_dic[param_names[j]] = grads[k]
            else:
                grad_dic[param_names[j]] += grads[k]
        del grads, dt
    
    ## obtain concatenated gradient vector
    grad_w = []
    for j in sel_idxs:
        if dictionary == False:
            grad_w.append(grad_dic[param_names[j]])

    if dictionary==False:
        grad_w = torch.cat(grad_w)
#         print(grad_w.shape)
        if normalize:
            grad_w /= torch.norm(grad_w)
        grad_w = grad_w
        return grad_w
    else:
        return grad_dic

def compute_indv_grad_with_label(clf,x,y,criterion,device,sel_idxs):
    clf.to(device).eval()
    x,y = x.to(device), y.to(device)
    autograd_hacks.add_hooks(clf)
    clf.zero_grad()
    autograd_hacks.clear_backprops(clf)
    out = clf(x)
    
    autograd_hacks.count_backprops(clf)
    criterion(out,y).backward()
    autograd_hacks.remove_backprops(clf)
    autograd_hacks.compute_grad1(clf)
    
    tmp=[]
    for k, param in enumerate(clf.parameters()):
        if k in sel_idxs:
            tmp.append(param.grad1.reshape(x.size(0),-1).detach().cpu())
#     grad_t = torch.cat(tmp,dim=1).detach().cpu()
    return tmp

def compute_indv_grad_wo_label(clf,x,criterion,device,sel_idxs):
    clf.to(device).eval()
    x = x.to(device)
    autograd_hacks.add_hooks(clf)
    clf.zero_grad()
    autograd_hacks.clear_backprops(clf)
    out = clf(x)
    
    autograd_hacks.count_backprops(clf)
    criterion(out).backward()
    autograd_hacks.remove_backprops(clf)
    autograd_hacks.compute_grad1(clf)
    
    tmp=[]
    for k, param in enumerate(clf.parameters()):
        if k in sel_idxs:
            tmp.append(param.grad1.reshape(x.size(0),-1).detach().cpu())
    del out
    
    return tmp

def compute_indv_grad_dic(clf,dataloader,device,criterion,use_label = False,param_names=None,
                          last_layer = True,sel_idxs=None, dictionary=False):
    clf.to(device)
    ## dataloader should be "Shuffle =False"
    assert(_is_shuffle(dataloader)==False)
    
    if param_names is None:
        param_names = obtain_param_names(clf)
        
    if sel_idxs is not None:
        assert(sel_layers in param_names)

    grad_dic = {}
    
    if last_layer:
        sel_idxs=list(range(len(param_names))[-2:])            
    elif sel_idxs is None:
        ## last_layer is False and sel_idxs is None
        sel_idxs = list(range(len(param_names)))
    else:
        ## last_layer is False and sel_idxs is not None
        pass
    
    for i in sel_idxs:
        grad_dic[param_names[i]] = []
    

    ## calculate gradient for dataloader    
    for i, (x,y,z) in enumerate(dataloader):
        if use_label:
            grads = compute_indv_grad_with_label(clf,x,y,criterion,device,sel_idxs)
        else:
            grads = compute_indv_grad_wo_label(clf,x,criterion,device,sel_idxs)
            
        for k,j in enumerate(sel_idxs):
            grad_dic[param_names[j]].append(grads[k])
        del grads, x,y,z
    
    
    ## stacks gradients for all instances
    
    for k,j in enumerate(sel_idxs):
        tmp = torch.cat(grad_dic[param_names[j]],axis=0)
        grad_dic[param_names[j]] = tmp
        if k ==0:
            grad_w = tmp
        else:
            grad_w = torch.cat((grad_w, tmp),1)
    if dictionary:
        return grad_dic
    else:
        return grad_w
        

def _obtain_wgrad_from_dictionary(grad_dic, param_names):
    wgrad = []
    for pn in param_names:
        wgrad.append(grad_dic[pn])
    return torch.cat(wgrad,axis=1)

def _is_shuffle(dataloader):
    if dataloader.sampler.__class__.__name__ == 'RandomSampler':
        return True
    else:
        assert(dataloader.sampler.__class__.__name__ == 'SequentialSampler') 
        return False