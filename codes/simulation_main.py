import argparse
import logging
import os, pickle, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils.data_utils import load_simulation_data

from utils.data_utils import dataset_description, input_uniqueness, initial_dataloaders
from utils.data_utils import load_data_with_name
from utils.train_utils import train_AL, train_AL_valid,train_AL_valid_trgrad
from utils.data_utils import save_dataloader

from utils import config

parser = argparse.ArgumentParser(description="Train FairAL trained models")
parser.add_argument(
    "--sel-type",
    default="binary_entropy",
    type=str,
    help="selection objective for gradient",
)
parser.add_argument(
    "--model-type",
    default="NN",
    type=str,
    help="model type NN (default) or LR",
)
parser.add_argument(
    "--last-layer",
    default=False,
    type=bool,
    help="compute the gradient only for last layer",
)
parser.add_argument(
    "--AL-batch",
    default=32,
    type=int,
    help="batch size for active learning",
)
parser.add_argument(
    "--repeat",
    type=int,
    default=10,
    help="repeat multiple times for different initializations",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    help="training epochs for each AL iteration",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",
    help="gpu device information",
)


def _run_training_AL(cm_args):
    Xtr,Xte,ytr,yte,Ztr,Zte = load_simulation_data(simulation_params = {'p':100,'q':40, 'r':10, 'b':0, 't':0}, 
                                                   n1=2000, n2=800, svm=False, random_state=55, intercept=False, 
                                                   train_frac = 0.7)

    args_dic = {}
    args_dic['epochs'] = cm_args.epochs
    args_dic['batch_size']=64
    args_dic['AL_batch']=cm_args.AL_batch
    args_dic['init_num'] = 100
    args_dic['tr_num'] = Xtr.shape[0]
    args_dic['AL_iters'] = None # if None, we can conduct AL for whole dataset
    args_dic['AL_select']='acc'
    args_dic['val_ratio'] =0.2 
    args_dic['val_scheduler'] ='linear' ## validation ratio scheduler add!
    args_dic['problem_type'] = 'binary' 
    args_dic['model_type'] =cm_args.model_type
    args_dic['model_args'] ={'n_hidden':32, 'p_dropout':0.0}
    args_dic['dataset'] = 'simulation'
    args_dic['save_model']=True
    args_dic['save_dir'] = None


    args = config.Args(**args_dic)

    args.print_args()
    sel_args = {"param_names":None, "last_layer":cm_args.last_layer, "sel_idxs": None, "normalize":True, "ent_num":200}
    sel_type = cm_args.sel_type # identity, binary_entropy, entropy, random
    args.set_selection_params(sel_type,sel_args)
    
    N1 = args.init_num
    train_loader, select_loader, test_loader = initial_dataloaders(Xtr,ytr,Ztr,Xte,yte,Zte,N1,args)
    
    device = torch.device(cm_args.device)
    
    clf, train_loader_p, select_loader_p, gids = train_AL_valid_trgrad(train_loader, select_loader, device, 
                                                      args, test_loader, True)
    save_dataloader(args.save_dir,test_loader)
    with open(os.path.join(args.save_dir,'train_AL_valid_trgrad.txt'),'w') as f:
        f.write(str(args_dic))
        f.write("\n")
        f.write(sel_type)
        f.write("\n")
        f.write(str(sel_args))    

    with open(os.path.join(args.save_dir,'selected_group_results.pkl'),'wb') as f:
        pickle.dump(gids, f)
    return args.save_dir
if __name__ == '__main__':
    cm_args = parser.parse_args()
    for i in range(cm_args.repeat):
        save_dir = _run_training_AL(cm_args)
        with open(cm_args.model_type+"_"+cm_args.sel_type+".txt","a") as f:
            f.write(save_dir)
            f.write("\n")