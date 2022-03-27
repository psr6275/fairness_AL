import argparse
import logging
import os, pickle, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.test_utils import obtain_AL_ckpts, split_AL_loaders
from utils.test_utils import load_AL_config, load_AL_dataloader, load_test_model
from utils.test_utils import test_groupwise, test_model

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
    
    for it,st in enumerate(state_list):
        st = cfgs.problem_type+cfgs.model_type +"_"+ str(it)+ ".pt"
        print("="*50)
        clf,clf_criterion = load_test_model(cfgs.save_dir, st, cfgs)
        _, acc = test_model(clf, dl_cum_loaders[it],clf_criterion, device, cfgs.problem_type)
        _, acc2 = test_model(clf, test_loader,clf_criterion, device, cfgs.problem_type)
        print("cumulated total training/test acc:", acc,"/",acc2)
        test_groupwise(clf, dl_cum_loaders[it],clf_criterion, device, 
                   AL_select = cfgs.AL_select, problem_type = cfgs.problem_type, return_loader=False)
        print("test set performance")
        test_groupwise(clf, test_loader, clf_criterion, device, 
                   AL_select = cfgs.AL_select, problem_type = cfgs.problem_type, return_loader=False)
        print("selected group is", gids[it])
        print('-'*50)
        for i in range(it+1):
            _, acc = test_model(clf, dl_loaders[i], clf_criterion, device, cfgs.problem_type)
            print("data loader",i, "acc:",acc)
            test_groupwise(clf, dl_loaders[i],clf_criterion, device, 
                   AL_select = cfgs.AL_select, problem_type = cfgs.problem_type, return_loader=False)
            print("")

if __name__ == "__main__":
    args = parser.parse_args()
    _run_test_models(args)