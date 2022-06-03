import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from .grad_utils import obtain_param_names, _is_shuffle, compute_mean_grad_dic, compute_indv_grad_dic

class BinaryNN(nn.Module):

    def __init__(self, n_features, n_hidden=32, p_dropout=0.2):
        super(BinaryNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.network(x))

class BinaryLR(nn.Module):

    def __init__(self, n_features):
        super(BinaryLR, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.network(x))    
    
class BinaryEntropy(nn.Module):
    def __init__(self,reduction='mean'):
        super(BinaryEntropy,self).__init__()
        self.reduction = reduction
        
    def forward(self,output):
        loss = -output*torch.log(output)  - (1-output)*torch.log(1-output)
        if self.reduction =='mean':
            return loss.mean()
        elif self.reduction =='sum':
            return loss.sum()
        else:
            assert(self.reduction =='none')
            return loss

class IdentityOutput(nn.Module):
    def __init__(self,reduction='mean'):
        super(IdentityOutput,self).__init__()
        self.reduction = reduction
        
    def forward(self, output):
        if self.reduction == 'mean':
            return output.mean()
        elif self.reduction == 'sum':
            return output.sum()
        else:
            assert(self.reduction =='none')
            return output
        
class BinaryEntropy_stable(nn.Module):
    def __init__(self):
        super(BinaryEntropy_stable,self).__init__()
    def forward(self,output,logit):
        max_val = (-logit).clamp_min_(0)
        loss = (1-output)*logit +max_val+ torch.log(torch.exp(-max_val)+torch.exp(-logit-max_val))
        if self.reduction =='mean':
            return loss.mean()
        elif self.reduction =='sum':
            return loss.sum()
        else:
            return loss
        
def construct_model_binary(model_type='NN', n_features = 32, **model_args):
    if model_type == 'NN':
        clf = BinaryNN(n_features, model_args['n_hidden'],model_args['p_dropout'])
    elif model_type =='LR':
        clf = BinaryLR(n_features)
    else:
        raise NotImplementedError("Other models need to be implemented")
    return clf

def compute_indv_entropy(clf, dataloader, device,sel_batch_num):
    assert(_is_shuffle(dataloader)==False)
    
    clf.to(device).eval()
    criterion = BinaryEntropy(reduction='none')
    losses = []
    for dt in dataloader:
        x = dt[0].to(device)
        outs = clf(x)
        loss = criterion(outs)
        losses.append(loss.detach().cpu())
#     print(torch.cat(losses).shape,sel_batch_num)
    max_outs = torch.topk(torch.cat(losses).flatten(),sel_batch_num)
    return max_outs[1]

def compute_similarity_with_grad_binary(clf,val_loader,sel_loader,device,sel_batch_num,
                                        criterion_type = 'identity', param_names=None, 
                                        last_layer=True,sel_idxs=None, normalize=True, ent_num=None): 
    # val_loader consists of validation data for a chosen group
    assert(criterion_type in ['identity','binary_entropy'])
    if criterion_type == 'identity':
        ## identity
        criterion = IdentityOutput(reduction ='sum')
    else:
        ## binary_entropy
        criterion = BinaryEntropy(reduction ='sum')
        
    mean_wgrad = compute_mean_grad_dic(clf, val_loader,device, criterion,
                                      use_label = False, param_names=param_names, last_layer = last_layer,
                                      sel_idxs=sel_idxs, normalize=normalize, dictionary=False)
            
    ind_wgrad = compute_indv_grad_dic(clf, sel_loader, device, criterion, use_label = False, 
                                         param_names=param_names, last_layer = last_layer,
                                         sel_idxs=sel_idxs, dictionary=False)

    sim = torch.matmul(ind_wgrad,mean_wgrad)
#     print(sim.shape)
    max_outs = torch.topk(sim,sel_batch_num)
        
    return max_outs[1]

def compute_similarity_with_grad_binary_sign(clf, whole_loader, val_loader,sel_loader,device,sel_batch_num,
                                        criterion_type = 'identity', param_names=None, 
                                        last_layer=True,sel_idxs=None, normalize=True, ent_num=None): 
    # val_loader consists of validation data for a chosen group
    assert(criterion_type in ['be_sign'])
    if criterion_type == 'identity':
        ## identity
        criterion = IdentityOutput(reduction ='sum')
    else:
        ## binary_entropy
        criterion = BinaryEntropy(reduction ='sum')
        
    whole_wgrad = compute_mean_grad_dic(clf, whole_loader,device, criterion,
                                      use_label = False, param_names=param_names, last_layer = last_layer,
                                      sel_idxs=sel_idxs, normalize=False, dictionary=False)
    val_wgrad = compute_mean_grad_dic(clf, val_loader,device, criterion,
                                      use_label = False, param_names=param_names, last_layer = last_layer,
                                      sel_idxs=sel_idxs, normalize=False, dictionary=False)
    
    mean_wgrad = torch.sign(val_wgrad)-torch.sign(whole_wgrad)
            
    ind_wgrad = compute_indv_grad_dic(clf, sel_loader, device, criterion, use_label = False, 
                                         param_names=param_names, last_layer = last_layer,
                                         sel_idxs=sel_idxs, dictionary=False)

    sim = torch.matmul(ind_wgrad,mean_wgrad)
#     print(sim.shape)
    max_outs = torch.topk(sim,sel_batch_num)
        
    return max_outs[1]

def select_examples_binary(clf, whole_loader, val_loader, sel_loader, device, sel_batch_num, 
                           criterion_type, **sel_args):

    assert(criterion_type in ['identity','binary_entropy','random','entropy','be_sign', 'entropy_be', 'entropy_id'])
    
    if criterion_type in ['identity','binary_entropy']:
        sidxs = compute_similarity_with_grad_binary(clf, val_loader, sel_loader, device, sel_batch_num, 
                                            criterion_type, **sel_args)
    elif criterion_type in ['be_sign']:
        sidxs = compute_similarity_with_grad_binary_sign(clf, whole_loader,val_loader, sel_loader, device, sel_batch_num, 
                                            criterion_type, **sel_args)
    elif criterion_type == 'entropy':
        sidxs = compute_indv_entropy(clf, sel_loader, device,sel_batch_num)
    elif criterion_type == ['entropy_be', 'entropy_id']:
        sidxs0 = compute_indv_entropy(clf, sel_loader, device,sel_args.ent_num)
        if criterion_type == 'entropy_be':
            cr_type = 'binary_entropy'
        else:
            cr_type = 'identity'
        if sel_args.ent_num < sel_batch_num:
            sel_args.ent_num = sel_batch_num
        ds1 = sel_loader.dataset.tensors
        sel_loader2 = DataLoader(NPsDataSet(ds1[0][sidxs0],ds1[1][sidxs0],ds1[2][sidxs0]),batch_size=sel_args.ent_num, shuffle=False)
        sidxs1 = compute_similarity_with_grad_binary(clf, val_loader, sel_loader2, device, sel_batch_num, 
                                            cr_type, **sel_args)
        sidxs = torch.tensor([sidxs0[s1] for s1 in sidxs1])
    else:
        sidxs = torch.randperm(sel_loader.dataset.tensors[0].shape[0])[:sel_batch_num]
    
    return sidxs