import argparse
import logging
import os, pickle, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.test_utils import obtain_AL_ckpts, split_AL_loaders
from utils.test_utils import load_AL_config, load_AL_dataloader, load_test_model
from utils.test_utils import test_groupwise, test_model, test_group_perf
from utils.test_utils import save_logs, load_logs

from sklearn.linear_model import LogisticRegression as LogR
import numpy as np

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Test FairAL trained models")
parser.add_argument(
    "--save-dir",
    default="../results/simulation/220306_0139",
    type=str,
    metavar="PATH",
    help="path to result directory for saved model",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",
    help="gpu device information",
)

def dic2logdic(log_gs, log_g, tag = 'train'):
    if len(log_gs[tag].keys())==0:
        for zi in log_g.keys():
            log_gs[tag][zi] = []
            
    for zi in log_g.keys():
        
        log_gs[tag][zi].append(log_g[zi])
    return log_gs

def _test_logs(state_list, save_dir, args, device, dl_loaders, dl_cum_loaders, test_loader, gids):
    loss_gs = {}
    acc_gs = {}
    loss_log = {}
    acc_log = {}

    loss_gs['train'] = {}
    loss_gs['test'] = {}
    acc_gs['train'] = {}
    acc_gs['test'] = {}

    loss_log['train'] = []
    loss_log['test'] = []
    acc_log['train'] = []
    acc_log['test'] = []

    loss_log['train_w'] = []
    loss_log['test_w'] = []
    acc_log['train_w'] = []
    acc_log['test_w'] = []
    

    for it in range(len(state_list)):
        st = args.problem_type+args.model_type +"_"+ str(it)+ ".pt"
        print("="*50)
        clf,clf_criterion = load_test_model(save_dir, st, args)
        loss, acc = test_model(clf, dl_cum_loaders[it],clf_criterion, device, args.problem_type)
        acc_log['train'].append(acc.item())
        loss_log['train'].append(loss.item())

        loss, acc2 = test_model(clf, test_loader,clf_criterion, device, args.problem_type)
        acc_log['test'].append(acc2.item())
        loss_log['test'].append(loss.item())

        lrclf = LogR().fit(dl_cum_loaders[it].dataset.tensors[0].numpy(),dl_cum_loaders[it].dataset.tensors[1].numpy())
        acc3 = lrclf.score(dl_cum_loaders[it].dataset.tensors[0].numpy(),dl_cum_loaders[it].dataset.tensors[1].numpy())
        acc4 = lrclf.score(test_loader.dataset.tensors[0].numpy(),test_loader.dataset.tensors[1].numpy())
        print("cumulated total training/test acc:", acc,"/",acc2,"/",acc3*100,"/",acc4*100)
        loss_g, acc_g,loss_w, acc_w = test_group_perf(clf, dl_cum_loaders[it],clf_criterion, device, 
                   AL_select = args.AL_select, problem_type = args.problem_type, return_loader=False)
        loss_gs = dic2logdic(loss_gs, loss_g, tag = 'train')
        acc_gs = dic2logdic(acc_gs, acc_g, tag = 'train')
        loss_log['train_w'].append(loss_w)
        acc_log['train_w'].append(acc_w)

        print("test set performance")
        loss_g, acc_g, loss_w, acc_w = test_group_perf(clf, test_loader,clf_criterion, device, 
                   AL_select = args.AL_select, problem_type = args.problem_type, return_loader=False)
        loss_gs = dic2logdic(loss_gs, loss_g, tag = 'test')
        acc_gs = dic2logdic(acc_gs, acc_g, tag = 'test')
        loss_log['test_w'].append(loss_w)
        acc_log['test_w'].append(acc_w)

        print("selected group is", gids[it])
        print('-'*50)
        for i in range(it+1):
            _, acc = test_model(clf, dl_loaders[i],clf_criterion, device, args.problem_type)
            print("data loader",i, "acc:",acc)
            test_groupwise(clf, dl_loaders[i],clf_criterion, device, 
                   AL_select = args.AL_select, problem_type = args.problem_type, return_loader=False)
            print("")
    return loss_gs, acc_gs, loss_log, acc_log

def _run_test_models(args):
    
    device = torch.device(args.device)
    print("saved directory:", args.save_dir)
    
    config_file, state_list, data_files, setting_file = obtain_AL_ckpts(args.save_dir)
    print("ckpts: ",state_list)
    print("data loaderes: ", data_files)
    
    print("%"*50)
    with open(os.path.join(args.save_dir, setting_file),'r') as f:
        content = f.read()
    print(content)
    print("%"*50)        
    
    cfgs = load_AL_config(args.save_dir, config_file)
    dl_loader = load_AL_dataloader(args.save_dir,'final_dataloader.pkl')
    test_loader = load_AL_dataloader(args.save_dir,'test_loader.pkl')
    
    dl_loaders,dl_cum_loaders = split_AL_loaders(dl_loader, cfgs)
    
    with open(os.path.join(args.save_dir, 'selected_group_results.pkl'),'rb') as f:
        gids = pickle.load(f)
    
    loss_gs, acc_gs, loss_log, acc_log = _test_logs(state_list, args.save_dir, cfgs, device, dl_loaders, dl_cum_loaders, test_loader, gids)
    for gk in loss_gs.keys():
        for gik in loss_gs[gk].keys():
            if type(loss_gs[gk][gik]) == list:
                loss_gs[gk][gik] = np.array(loss_gs[gk][gik])
            if type(acc_gs[gk][gik]) == list:
                acc_gs[gk][gik] = np.array(acc_gs[gk][gik])
    for dtk in loss_log.keys():
        if type(loss_log[dtk]) ==list:
            loss_log[dtk] = np.array(loss_log[dtk])
        if type(acc_log[dtk]) == list:
            acc_log[dtk] = np.array(acc_log[dtk])
    
    save_logs(args.save_dir, loss_gs, acc_gs, loss_log, acc_log)

if __name__ == "__main__":
    args = parser.parse_args()
    _run_test_models(args)