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
from utils import LogsCLS, test_binary_model, test_sklearn_model

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


    
def _test_logs(state_list, save_dir, args, device, dl_loaders, dl_cum_loaders, test_loader, gids):
    
    for it in range(len(state_list)):
        st = args.problem_type+args.model_type +"_"+ str(it)+ ".pt"
        print("="*50)
        print("training data")
        clf,clf_criterion = load_test_model(save_dir, st, args)
        res = test_binary_model(clf, dl_cum_loaders[it], device)
        res_te = test_binary_model(clf, test_loader, device)
        
        clf_lr = LogR().fit(dl_cum_loaders[it].dataset.tensors[0].numpy(),dl_cum_loaders[it].dataset.tensors[1].numpy())
        reslr = test_sklearn_model(clf_lr, dl_cum_loaders[it])
        reslr_te = test_sklearn_model(clf_lr, test_loader)
        
        if it==0:
            logs = LogsCLS(crit_list = list(res[0].keys()))
            logs_lr = LogsCLS(crit_list = list(res[0].keys()))
        
        logs.append_result(res,'train')
        logs.append_result(res_te,'test')
        logs_lr.append_result(reslr, 'train')
        logs_lr.append_result(reslr_te, 'test')

    return logs, logs_lr
    
def _run_test_models(args):
    device = torch.device(args.device)
    print("saved directory: ",args.save_dir)
    
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
        
    logs, logs_lr = _test_logs(state_list, args.save_dir, cfgs, device, dl_loaders, dl_cum_loaders, test_loader, gids)
    
    with open(os.path.join(args.save_dir, "clf_logs.cls"), "wb") as f:
        pickle.dump(logs, f)
    with open(os.path.join(args.save_dir, "clf_lr_logs.cls"), "wb") as f:
        pickle.dump(logs_lr, f)

if __name__ == "__main__":
    args = parser.parse_args()
    _run_test_models(args)