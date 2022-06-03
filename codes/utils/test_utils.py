import torch
import torch.nn as nn
import os, glob ## glob has useful functions for directory
import pickle
import matplotlib.pyplot as plt

from .eval_utils import AverageVarMeter, accuracy, accuracy_b, binary_scores
from .data_utils import divide_groupsDL, make_dataloader
from .binary_utils import construct_model_binary
import copy
import numpy as np

class LinearScheduler:
    def __init__(self, start = 0.2, end = 1.0, iters = 10):
        self.start = start
        self.end = end
        self.iters = iters
        self.val = start
    def update(self,it):
        if it> self.iters:
            it = self.iters
        self.val = self.start+(it/self.iters)*(self.end-self.start)
class ConstantScheduler:
    def __init__(self, val = 0.2):
        self.val = val
    def update(self,it):
        return

def average_performance(save_dirs):
    for i,sd in enumerate(save_dirs):
        loss_gs, acc_gs, loss_log, acc_log = load_logs(sd)
        if i==0:
            lossgs = copy.deepcopy(loss_gs)
            accgs = copy.deepcopy(acc_gs)
            losslog = copy.deepcopy(loss_log)
            acclog = copy.deepcopy(acc_log)
        else:
            for gk in lossgs.keys():
                for gik in lossgs[gk].keys():
                    lossgs[gk][gik] = np.row_stack((lossgs[gk][gik],loss_gs[gk][gik]))
                    accgs[gk][gik] = np.row_stack((accgs[gk][gik],acc_gs[gk][gik]))
            for gk in losslog.keys():
                losslog[gk] = np.row_stack((losslog[gk], loss_log[gk]))
                acclog[gk] = np.row_stack((acclog[gk], acc_log[gk]))
                
    for gk in lossgs.keys():
        for gik in lossgs[gk].keys():
            lossgs[gk][gik] = np.mean(lossgs[gk][gik],axis=0)
            accgs[gk][gik] = np.mean(accgs[gk][gik],axis=0)
    for gk in losslog.keys():
        losslog[gk] = np.mean(losslog[gk], axis=0)
        acclog[gk] = np.mean(acclog[gk], axis=0)
    return lossgs, accgs, losslog, acclog    

def plot_compare(arr_list, name_list):
    for i,arr in enumerate(arr_list):
        plt.plot(arr, '-', label=name_list[i])
    plt.legend()
    plt.show()

def plot_results(loss_gs, acc_gs, loss_log, acc_log):
    fig1, axs1 = plt.subplots(1,2)
    fig2, axs2 = plt.subplots(1,2)
    axs1[0].set_title("training loss")
    axs1[1].set_title("training accuracy")
    axs2[0].set_title("test loss")
    axs2[1].set_title("test accuracy")
#     print(loss_gs)
    for k in loss_gs.keys():
        if 'train' in k:
            axs = axs1
        else:
            axs = axs2
        for j in loss_gs[k].keys():
            axs[0].plot(loss_gs[k][j],'--',label=str(k)+'_'+str(j))
            axs[1].plot(acc_gs[k][j],'--',label=str(k)+'_'+str(j))
    for k in loss_log.keys():
        if 'train' in k:
            axs = axs1
        else:
            axs = axs2
#         print(k,loss_log[k])
        axs[0].plot(loss_log[k],'-', label=k)
        axs[1].plot(acc_log[k],'-', label=k)
    
    axs[0].legend(bbox_to_anchor =(0.65, 1.25))
    plt.show()    

def save_logs(result_dir, loss_gs, acc_gs, loss_log, acc_log):
    print("save loss group logs")
    with open(os.path.join(result_dir, "loss_group_logs.pkl"),"wb") as f:
        pickle.dump(loss_gs, f)

    print("save accuracy group logs")
    with open(os.path.join(result_dir, "acc_group_logs.pkl"),"wb") as f:
        pickle.dump(acc_gs, f)
    
    print("save average/worst loss logs")
    with open(os.path.join(result_dir, "loss_avg_worst_logs.pkl"),"wb") as f:
        pickle.dump(loss_log, f)
    
    print("save average/worst accuracy logs")
    with open(os.path.join(result_dir, "acc_avg_worst_logs.pkl"),"wb") as f:
        pickle.dump(acc_log, f)    
    
def load_logs(result_dir):
    print("load loss group logs")
    with open(os.path.join(result_dir, "loss_group_logs.pkl"),"rb") as f:
        loss_gs = pickle.load(f)

    print("load accuracy group logs")
    with open(os.path.join(result_dir, "acc_group_logs.pkl"),"rb") as f:
        acc_gs =  pickle.load(f)
    
    print("load average/worst loss logs")
    with open(os.path.join(result_dir, "loss_avg_worst_logs.pkl"),"rb") as f:
        loss_log = pickle.load(f)
    
    print("load average/worst accuracy logs")
    with open(os.path.join(result_dir, "acc_avg_worst_logs.pkl"),"rb") as f:
        acc_log = pickle.load(f)
    return loss_gs, acc_gs, loss_log, acc_log    
    
def load_AL_config(save_dir, config_file):
    args = torch.load(os.path.join(save_dir, config_file))
    return args

def load_AL_dataloader(save_dir, data_file):
    dataloader = torch.load(os.path.join(save_dir, data_file))
    return dataloader

def load_test_model(save_dir, state_file, args):    
    if args.problem_type == 'binary':
        construct_model = construct_model_binary
        clf_criterion = nn.BCELoss()
    else:
        raise NotImplementedError("need to be implemented for multi-class classification")
    
    clf = construct_model(args.model_type,args.n_features,**args.model_args)
    clf.load_state_dict(torch.load(os.path.join(save_dir,state_file), map_location='cpu'))
    clf.eval()
    print("loaded model is", state_file)
    
    return clf, clf_criterion

def split_AL_loaders(dataloader, args):
    dl_loaders = []
    dl_cum_loaders = []
    st_num = args.init_num
    ds = dataloader.dataset.tensors
    dl_loaders.append(make_dataloader(ds[0][:st_num],ds[1][:st_num],ds[2][:st_num],
                                     args.batch_size, False))
    dl_cum_loaders.append(dl_loaders[0])
    for i in range(args.AL_iters):
        st_num = args.init_num + args.AL_batch*i
        dt_num = args.init_num + args.AL_batch*(i+1)
        dl_loaders.append(make_dataloader(ds[0][st_num:dt_num],ds[1][st_num:dt_num],ds[2][st_num:dt_num],
                                     args.batch_size, False))
        dl_cum_loaders.append(make_dataloader(ds[0][:dt_num],ds[1][:dt_num],ds[2][:dt_num],
                                     args.batch_size, False))
    return dl_loaders, dl_cum_loaders

def obtain_AL_ckpts(save_dir):
    state_files = []
    data_files = []

    for di in os.listdir(save_dir):
        if ".args" in di:
            config_file = di
        elif ".pt" in di:
            state_files.append(di)
        elif ".pkl" in di:
            data_files.append(di)
        elif ".txt" in di:
            setting_file = di
        else:
            print("unknown file: ",di)
    state_files.sort()
    return config_file, state_files, data_files, setting_file

def test_groupwise(clf, data_loader, clf_criterion, device, 
                   AL_select = 'loss', problem_type = 'binary', return_loader = True, test_loader = None):
    clf.to(device).eval()
    if test_loader is None:
        dlTensors = data_loader.dataset.tensors
        dldic = divide_groupsDL(dlTensors[0],dlTensors[1],dlTensors[2])
    else:
        dlTensors = test_loader.dataset.tensors
        dldic = divide_groupsDL(dlTensors[0],dlTensors[1],dlTensors[2])
    del dlTensors
    losss = 0
    accs = 100.0
    gid = list(dldic.keys())[0]

    for did in dldic.keys():
        loss_v, acc_v = test_model(clf, dldic[did],clf_criterion, device, problem_type)
        print("{} : loss {} / acc {}".format(did, loss_v, acc_v))
        
        if AL_select == 'loss':
#             print(losss,loss_v,sid,did)
            if losss <= loss_v:
                print(gid,did)
                gid = did
                losss = loss_v               
        else:
#             print(did,acc_v,accs)
            assert AL_select == 'acc'
#             print(accs > acc_v)
            if accs >= acc_v:
#                 print(sid,did)
                gid = did
                accs = acc_v
        del loss_v, acc_v, did
    if test_loader is not None:
        dlTensors = data_loader.dataset.tensors
        dldic = divide_groupsDL(dlTensors[0],dlTensors[1],dlTensors[2])
        del dlTensors
    worst_dl = dldic[gid]
    del dldic
   
    return gid, worst_dl

def test_group_perf(clf, test_loader, clf_criterion, device, 
                   AL_select = 'loss', problem_type = 'binary', return_loader = True):

    clf.to(device).eval()
        
    dlTensors = test_loader.dataset.tensors
    dldic = divide_groupsDL(dlTensors[0],dlTensors[1],dlTensors[2])
    del dlTensors
        
    losss = 0
    accs = 100.0
    gid = list(dldic.keys())[0]
    
    loss_g = {}
    acc_g = {}
    
    for did in dldic.keys():
        loss_v, acc_v = test_model(clf, dldic[did],clf_criterion, device, problem_type)
        print("{} : loss {} / acc {}".format(did, loss_v, acc_v))

        loss_g[did] = loss_v.item()
        acc_g[did] = acc_v.item()
        
        if losss <= loss_v:
            losss = loss_v.item() 
            

        if accs >= acc_v:
            accs = acc_v.item()

        del loss_v, acc_v, did
        
    return loss_g, acc_g, losss, accs
    
def predict_model(model, test_loader, device,as_np = False):
    model.to(device).eval()
    preds = []
    ys = []
    zs = []
    with torch.no_grad():
        for batch_idx, ts in enumerate(test_loader):
            ys.append(ts[1])
            if len(ts)>2:
                zs.append(ts[2])
            preds.append(model(ts[0].to(device)).detach().cpu())
            del ts
    preds = torch.cat(preds,0)
    ys = torch.cat(ys,0)
    if len(zs)>0:
        zs = torch.cat(zs,0)
    else:
        zs = torch.tensor(zs)
    if as_np:
        return preds.numpy(), ys.numpy(), zs.numpy()
    else:
        return preds, ys, zs
    
def test_binary_model(model, test_loader, device):
    preds, ys, zs = predict_model(model, test_loader, device)
    res = []
    print("="*30)
    print("binary scores for all data")
    print("="*30)
    res.append(binary_scores(preds,ys, prob=True, thres = 0.5))
    if len(zs)>0:
        zu = torch.unique(zs)
        for zi in zu:
            print("="*30)
            print("binary score for group: ",zi.item())
            print("="*30)
            zidx = zs ==zi
            res.append(binary_scores(preds[zidx].view(-1,1),ys[zidx].view(-1,1), 
                          prob=True, thres = 0.5))
    return res

def test_sklearn_model(model, test_loader):
    preds = torch.tensor(model.predict_proba(test_loader.dataset.tensors[0].numpy())[:,1]).view(-1,1)

    ys = test_loader.dataset.tensors[1]
    if len(test_loader.dataset.tensors)>2:
        zs = test_loader.dataset.tensors[2]
    else:
        zs = torch.tensor([])
#     print(preds.shape, ys.shape, zs.shape)
    res = []
    print("="*30)
    print("binary scores for all data")
    print("="*30)
    res.append(binary_scores(preds,ys, prob=True, thres = 0.5))
    if len(zs)>0:
        zu = torch.unique(zs)
        for zi in zu:
            print("="*30)
            print("binary score for group: ",zi.item())
            print("="*30)
            zidx = zs ==zi
            res.append(binary_scores(preds[zidx].view(-1,1),ys[zidx].view(-1,1), 
                          prob=True, thres = 0.5))
    return res
    
def test_model(model, test_loader, criterion, device, problem_type = 'binary'):
    model.to(device).eval()
    losses = AverageVarMeter()
    accs = AverageVarMeter()

    if problem_type == 'binary':
        acc_fn = accuracy_b
    else:
        acc_fn = accuracy
    with torch.no_grad():    
        for batch_idx, ts in enumerate(test_loader):
            x = ts[0].to(device)
            y = ts[1].to(device)
            p_y = model(x)
            loss = criterion(p_y,y)
            acc = acc_fn(p_y.detach().cpu(), y.detach().cpu())
    #         acc = accuracy_b(p_y, y)
            losses.update(loss,x.size(0))
            accs.update(acc,x.size(0))
            del ts,x,y,loss
    model.cpu()    
    return losses.avg.detach().cpu(), accs.avg.detach().cpu()
