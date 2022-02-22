import torch
import torch.nn as nn
import os, glob ## glob has useful functions for directory

from .eval_utils import AverageVarMeter, accuracy, accuracy_b
from .data_utils import divide_groupsDL, make_dataloader
from .binary_utils import construct_model_binary

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
    clf.load_state_dict(torch.load(os.path.join(save_dir,state_file)))
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
        else:
            print("unknown file: ",di)
    state_files.sort()
    return config_file, state_files, data_files

def test_groupwise(clf, data_loader, clf_criterion, device, 
                   AL_select = 'loss', problem_type = 'binary', return_loader = True):
    clf.to(device).eval()
    dlTensors = data_loader.dataset.tensors
    dldic = divide_groupsDL(dlTensors[0],dlTensors[1],dlTensors[2])
    losss = 0
    accs = 100.0
    gid = list(dldic.keys())[0]

    for did in dldic.keys():
        loss_v, acc_v = test_model(clf, dldic[did],clf_criterion, device, problem_type)
        print("{} : loss {} / acc {}".format(did, loss_v, acc_v))
        
        if AL_select == 'loss':
#             print(losss,loss_v,sid,did)
            if losss < loss_v:
                print(gid,did)
                gid = did
                losss = loss_v               
        else:
#             print(did,acc_v,accs)
            assert AL_select == 'acc'
#             print(accs > acc_v)
            if accs > acc_v:
#                 print(sid,did)
                gid = did
                accs = acc_v
        del loss_v, acc_v, did
   
    worst_dl = dldic[gid]
    del dldic
    if return_loader:
        return worst_dl
    else:
        return gid

def test_model(model, test_loader, criterion, device, problem_type = 'binary'):
    model.to(device).eval()
    losses = AverageVarMeter()
    accs = AverageVarMeter()

    if problem_type == 'binary':
        acc_fn = accuracy_b
    else:
        acc_fn = accuracy
        
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
        
    return losses.avg.detach().cpu(), accs.avg.detach().cpu()
