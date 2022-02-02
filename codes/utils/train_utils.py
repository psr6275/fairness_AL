import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from livelossplot import PlotLosses

import os

from .data_utils import obtain_newDS_rev, train_valid_split
from .eval_utils import AverageVarMeter, accuracy, accuracy_b
from .binary_utils import construct_model_binary, select_examples_binary
from .test_utils import test_groupwise, test_model

def train_AL(train_loader,select_loader,device,args,test_loader=None,from_scratch=True):
    
    n_features = train_loader.dataset.tensors[0].shape[1]
    if args.problem_type =='binary':
        construct_model = construct_model_binary
        clf_criterion = nn.BCELoss()
        select_examples = select_examples_binary
        
    else:
        raise NotImplementedError("need to be implemented for multi-class classification")

    if from_scratch ==False:
        clf = construct_model(args.model_type,n_feautres,**args.model_args)
            
    liveloss = PlotLosses()

    assert((args.AL_iters-1)*args.AL_batch<select_loader.dataset.tensors[0].shape[0])
    
    if args.save_model:
        args.n_features = n_features
        args_path = os.path.join(args.save_dir, args.sel_fn+"_config.args")
        torch.save(args,args_path)
        print("config argument is saved in ", args_path)
    
    for it in range(args.AL_iters):

        if from_scratch:            
            clf = construct_model(args.model_type, n_features, **args.model_args)
            ## optimizer!
            clf_optimizer = optim.Adam(clf.parameters())
        
        clf.to(device)
        
        ## training the model
        train_model(clf, train_loader, clf_criterion, clf_optimizer, device, 
                    args.epochs, test_loader, liveloss, args.problem_type)
        
        if args.save_model:
            model_path = args.problem_type+args.model_type+"_%d.pt"%(it)
            torch.save(clf.state_dict(),os.path.join(args.save_dir, model_path))
        ## find the worst group and select a batch!
        worst_loader = test_groupwise(clf, train_loader, clf_criterion, device, 
                                      args.AL_select, args.problem_type, return_loader = True)
        
        sidx = select_examples(clf,worst_loader,select_loader,device,args.AL_batch,
                                             args.sel_fn, **args.sel_args)
        
        
        #select loader is not shuffle!
        train_loader, select_loader = obtain_newDS_rev(train_loader, select_loader, 
                                                       sidx, args.batch_size)
    
    if args.save_model:
        data_file = os.path.join(args.save_dir, "final_dataloader.pkl")
        torch.save(train_loader,data_file)
        print("final dataloader is saved in ", data_file)
        
    return clf, train_loader, select_loader

def train_AL_valid(train_loader, select_loader, device, args, test_loader=None,from_scratch=True):
    
    n_features = train_loader.dataset.tensors[0].shape[1]
    
    if args.problem_type =='binary':
        construct_model = construct_model_binary
        clf_criterion = nn.BCELoss()
        select_examples = select_examples_binary
        
    else:
        raise NotImplementedError("need to be implemented for multi-class classification")
    
    if from_scratch ==False:
        clf = construct_model(args.model_type,n_feautres,**args.model_args)
            
    liveloss = PlotLosses()
    
    assert((args.AL_iters-1)*args.AL_batch<select_loader.dataset.tensors[0].shape[0])
    
    tr_num = train_loader.dataset.tensors[0].size(0)
    
    if args.save_model:
        args.n_features = n_features
        args_path = os.path.join(args.save_dir, "valid_config.args")
        torch.save(args,args_path)
        print("config argument is saved in ", args_path)
    
    for it in range(args.AL_iters):
        if from_scratch:
            clf = construct_model(args.model_type, n_features, **args.model_args)
            ## optimizer!
            clf_optimizer = optim.Adam(clf.parameters())
            
        clf.to(device)
        
        ## train_valid split!
        tr_loader, val_loader = train_valid_split(train_loader,tr_num,args.val_ratio,random_seed = it)
        
        ## training the model
        train_model(clf, tr_loader, clf_criterion, clf_optimizer, device, 
                    args.epochs, test_loader, liveloss, args.problem_type)
        
        if args.save_model:
            model_path = args.problem_type+args.model_type+"_%d.pt"%(it)
            torch.save(clf.state_dict(),os.path.join(args.save_dir, model_path))
        
        ## find the worst group and select a batch!
        worst_loader = test_groupwise(clf, val_loader, clf_criterion, device, 
                                      args.AL_select, args.problem_type, return_loader = True)
                 
        sidx = select_examples(clf, worst_loader,select_loader,device,args.AL_batch,
                                             args.sel_fn, **args.sel_args)
#         print(sidx.shape)    
        #select loader is not shuffle!
        train_loader, select_loader = obtain_newDS_rev(train_loader, select_loader, 
                                                       sidx, args.batch_size)
    
    if args.save_model:
        data_file = os.path.join(args.save_dir, "final_dataloader.pkl")
        torch.save(train_loader,data_file)
        print("final dataloader is saved in ", data_file)
    
    return clf, train_loader, select_loader


def train_model(model, train_loader, criterion, optimizer, device, epochs, 
                test_loader = None, liveloss = None, problem_type = 'binary'):
    model.train()
    if liveloss is None:
        liveloss = PlotLosses()
    
    if problem_type == 'binary':
        acc_fn = accuracy_b
    else:
        acc_fn = accuracy
    
    logs = {}
    
    for epoch in range(epochs):
        model.train()
        losses = AverageVarMeter()
        accs = AverageVarMeter()
        
        for batch_idx, ts in enumerate(train_loader):
#             print(device)
            x = ts[0].to(device)
            y = ts[1].to(device)
            optimizer.zero_grad()
            p_y = model(x)
            loss = criterion(p_y, y)
            loss.backward()
            optimizer.step()
            acc = acc_fn(p_y.detach().cpu(),y.detach().cpu())

            losses.update(loss,x.size(0))
            accs.update(acc,x.size(0))
            
            del x,y,ts,loss,acc

        logs['loss'] = losses.avg.detach().cpu()
        logs['acc'] = accs.avg.detach().cpu()
        
        if test_loader is not None:
            logs['val_loss'], logs['val_acc'] = test_model(model, test_loader, criterion, 
                                                           device, problem_type)
        
        liveloss.update(logs)
        liveloss.send()
        
    print('Finished Training')

