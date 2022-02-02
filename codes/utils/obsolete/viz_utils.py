import autograd_hacks
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch


def viz_plot_grad(grad_arrs, c_idx = None,method='pca',lywise=False,metric='euclidean',param_names = None):
    if lywise:
        print("lywise")
        assert param_names is not None
        assert type(grad_arrs) == dict
        for i, pn in enumerate(param_names):
#             print(grad_list[pn])
            grad_arr = grad_arrs[pn]
            draw_plot(grad_arr,c_idx,method,metric,title=pn)
    else:
#         grad_arr = np.array(torch.cat(grad_list,dim=0).numpy(),dtype=np.float32)
#         print(grad_arr.shape)
        draw_plot(grad_arrs,c_idx,method,metric,title="whole grad")

def draw_plot(grad_arr, c_idx, method,metric = 'euclidean',title =None):
    assert method in ['pca','tsne']
    if method == 'pca':
        grad_emb = PCA(n_components=2,whiten=False).fit_transform(grad_arr)
    else:
        grad_emb = TSNE(n_components=2,learning_rate='auto',init = 'random',metric=metric).fit_transform(grad_arr)
    fig,ax = plt.subplots()
    scatter = ax.scatter(grad_emb[:,0],grad_emb[:,1],c=c_idx)
    legend1 = ax.legend(*scatter.legend_elements())
    ax.add_artist(legend1)
    if title is not None:
        plt.title(title+' '+method + ' ' + metric)
    plt.show()

def compute_grad(clf,x,y,criterion,device):
    clf.to(device).eval()
    x,y = x.to(device),y.to(device)
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
        tmp.append(param.grad1.reshape(x.size(0),-1).detach().cpu())
    grad_t = torch.cat(tmp,dim=1).detach().cpu()
    return tmp, grad_t

def compute_grad_entropy(clf,x,criterion,device):
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
        tmp.append(param.grad1.reshape(x.size(0),-1).detach().cpu())
    grad_t = torch.cat(tmp,dim=1).detach().cpu()
    return tmp, grad_t

def compute_predgrad_arrs(clf, dataloader, criterion, device, param_names, entropy=False):
    clf.to(device)
    grad_dic = {}
    for pn in param_names:
        grad_dic[pn]=[]
    wgrad_list = []
    y_list = []
    z_list = []
    out_list = []

    for i, (x,y,z) in enumerate(dataloader):
    #     if i>1:
    #         break
        outs = (clf(x.to(device)).detach()>0.5).float()
        out_list.append(outs.cpu())
        
        if entropy:
            tmp, grt = compute_grad_entropy(clf,x,criterion,device)
        else:
            tmp,grt = compute_grad(clf,x,outs,criterion,device)
        
        clf.to(device)
        
        for j,pn in enumerate(param_names):
            grad_dic[pn].append(tmp[j])
        wgrad_list.append(grt)
        y_list.append(y.detach().cpu().numpy())
        z_list.append(z.detach().cpu().numpy())

    out_arr = np.array(torch.cat(out_list,axis=0).numpy()>0.5,dtype=np.float32).flatten()
    y_arr = np.concatenate(y_list).flatten()
    z_arr = np.concatenate(z_list).flatten()    
    wgrad_arr = np.array(torch.cat(wgrad_list,axis=0).numpy(),dtype=np.float32)
    for pn in param_names:
        grad_dic[pn] = np.array(torch.cat(grad_dic[pn],axis=0),dtype =np.float32)
    
    return out_arr, y_arr, z_arr, wgrad_arr, grad_dic


def compute_grad_arrs(clf, dataloader, criterion, device, param_names):
    clf.to(device)
    grad_dic = {}
    for pn in param_names:
        grad_dic[pn]=[]
    wgrad_list = []
    y_list = []
    z_list = []
    out_list = []

    for i, (x,y,z) in enumerate(dataloader):
    #     if i>1:
    #         break
        tmp,grt = compute_grad(clf,x,y,criterion,device)
        clf.to(device)
        out_list.append(clf(x.to(device)).detach().cpu())
        for j,pn in enumerate(param_names):
            grad_dic[pn].append(tmp[j])
        wgrad_list.append(grt)
        y_list.append(y.detach().cpu().numpy())
        z_list.append(z.detach().cpu().numpy())

    out_arr = np.array(torch.cat(out_list,axis=0).numpy()>0.5,dtype=np.float32).flatten()
    y_arr = np.concatenate(y_list).flatten()
    z_arr = np.concatenate(z_list).flatten()    
    wgrad_arr = np.array(torch.cat(wgrad_list,axis=0).numpy(),dtype=np.float32)
    for pn in param_names:
        grad_dic[pn] = np.array(torch.cat(grad_dic[pn],axis=0),dtype =np.float32)
    
    return out_arr, y_arr, z_arr, wgrad_arr, grad_dic

