import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from livelossplot import PlotLosses


from eval_utils import *
from fairAL_utils import divide_groupsDL, cal_meangrad, select_examples, select_random,select_entexamples,select_binary_entexamples, select_clf_entropy_examples
from load_data import obtain_newDS,train_valid_split

class Args:
    def __init__(self):
        self.epochs = 30
        self.max_epochs = 40
        self.AL_iters = 10 # AL batch 몇 번 뽑는지?
        self.AL_batch = 32 # AL 시에 select 되는 데이터 수
        self.batch_size = 32
        self.AL_select = 'loss'
        
    def print_args(self):
        print("train epochs/batch: {}/{}".format(self.epochs,self.batch_size))
        print("AL iters/batch: {}/{}".format(self.AL_iters,self.AL_batch))
        print("AL selection is based on ", self.AL_select)

def train_AL(train_loader, select_loader, device, args = None, test_loader = None,clf_type='NN',\
             from_scratch = True, sel_method = 'random'):
    if args is None:
        args = Args()

    print("arguments: ", args.print_args())
    n_features = train_loader.dataset.tensors[0].shape[1]
    if from_scratch ==False:
        if clf_type == 'NN':
            clf = Classifier(n_features=n_features)
        else:
            clf = ClassifierLR(n_features=n_features)
        clf_criterion = nn.BCELoss()
        clf_optimizer = optim.Adam(clf.parameters())
    liveloss = PlotLosses()
    assert((args.AL_iters-1)*args.AL_batch<select_loader.dataset.tensors[0].shape[0])
    for it in range(args.AL_iters):
        if from_scratch:
            if clf_type =='NN':
                clf = Classifier(n_features=n_features)
            else:
                clf = ClassifierLR(n_features=n_features)
            clf_criterion = nn.BCELoss()
            clf_optimizer = optim.Adam(clf.parameters())
        clf.to(device)
        
        train_model(clf, train_loader, clf_criterion, clf_optimizer, device, args.epochs, test_loader, liveloss)
        
        if it <args.AL_iters-1:
            if sel_method == 'random':
                ses,sidx = select_random(clf, select_loader,device, args.AL_batch)
            elif sel_method == 'entropy':
                sid, dldic = test_groupwise(clf, train_loader, clf_criterion, device, args)
                grads = cal_meangrad(clf, dldic[sid], clf_criterion, device)
                ses,sidx = select_entexamples(clf, select_loader, grads,device, args)
            elif sel_method == 'binary_entropy':
                sid, dldic = test_groupwise(clf, train_loader, clf_criterion, device, args)
                grads = cal_meangrad(clf, dldic[sid], clf_criterion, device)
                ses,sidx = select_binary_entexamples(clf, select_loader, grads,device, args)
            elif sel_method == 'clf_entropy':
                ses,sidx = select_clf_entropy_examples(clf, select_loader, device, args)
            else:
                sid, dldic = test_groupwise(clf, train_loader, clf_criterion, device, args)
                grads = cal_meangrad(clf, dldic[sid], clf_criterion, device)
                ses,sidx = select_examples(clf, select_loader, clf_criterion, grads,device, args)
            
            
#             print(ses)
            train_loader, select_loader = obtain_newDS(train_loader, select_loader, ses, sidx, args.batch_size)
#         print(train_loader.dataset.tensors[0].shape)
#         print(test_loader.dataset.tensors[0].shape)
#     print(train_loader.dataset.tensors[0].shape)
#     print(select_loader.dataset.tensors[0].shape)
    return clf, train_loader, select_loader    

def train_AL_valid(train_loader, select_loader, device, args = None, test_loader = None, clf_type='NN', from_scratch = True,\
                   sel_method = 'random',val_ratio = 0.2):
    if args is None:
        args = Args()

    print("arguments: ", args.print_args())
    n_features = train_loader.dataset.tensors[0].shape[1]
    if from_scratch ==False:
        if clf_type == 'NN':
            clf = Classifier(n_features=n_features)
        else:
            clf = ClassifierLR(n_features=n_features)
        clf_criterion = nn.BCELoss()
        clf_optimizer = optim.Adam(clf.parameters())
    liveloss = PlotLosses()
    assert((args.AL_iters-1)*args.AL_batch<select_loader.dataset.tensors[0].shape[0])
    tr_num = train_loader.dataset.tensors[0].size(0)
    for it in range(args.AL_iters):
        if from_scratch:
            if clf_type =='NN':
                clf = Classifier(n_features=n_features)
            else:
                clf = ClassifierLR(n_features=n_features)
            clf_criterion = nn.BCELoss()
            clf_optimizer = optim.Adam(clf.parameters())
        clf.to(device)
        #### train_valid_split ####
        tr_loader, val_loader = train_valid_split(train_loader,tr_num,val_ratio,random_seed = it)
        
        train_model(clf, tr_loader, clf_criterion, clf_optimizer, device, args.epochs, test_loader, liveloss)

        if it <args.AL_iters-1:
            if sel_method =='random':
                ses,sidx = select_random(clf, select_loader,device, args.AL_batch)
            elif sel_method == 'entropy':
                sid, dldic = test_groupwise(clf, train_loader, clf_criterion, device, args)
                grads = cal_meangrad(clf, dldic[sid], clf_criterion, device)
                ses,sidx = select_entexamples(clf, select_loader, grads,device, args)
            elif sel_method == 'binary_entropy':
                sid, dldic = test_groupwise(clf, train_loader, clf_criterion, device, args)
                grads = cal_meangrad(clf, dldic[sid], clf_criterion, device)
                ses,sidx = select_binary_entexamples(clf, select_loader, grads,device, args)
            elif sel_method == 'clf_entropy':
                ses,sidx = select_clf_entropy_examples(clf, select_loader, device, args)
            else:
                sid, dldic = test_groupwise(clf, train_loader, clf_criterion, device, args)
                grads = cal_meangrad(clf, dldic[sid], clf_criterion, device)
                ses,sidx = select_examples(clf, select_loader, clf_criterion, grads,device, args)

#             print(ses)
            train_loader, select_loader = obtain_newDS(train_loader, select_loader, ses, sidx, args.batch_size)
#         print(train_loader.dataset.tensors[0].shape)
#         print(test_loader.dataset.tensors[0].shape)
#     print(train_loader.dataset.tensors[0].shape)
#     print(select_loader.dataset.tensors[0].shape)
    return clf, train_loader, select_loader    



def train_model(model, train_loader, criterion, optimizer, device, epochs, test_loader = None, liveloss = None):
    model.train()
    if liveloss is None:
        liveloss = PlotLosses()
#     groups = {'acccuracy': ['acc', 'val_acc'], 'log-loss': ['loss', 'val_loss']}
#     plotlosses = PlotLosses(groups=groups, outputs=outputs)
    logs = {}
    for epoch in range(epochs):
        model.train()
        losses = AverageVarMeter()
        accs = AverageVarMeter()
        for batch_idx, (x,y, _) in enumerate(train_loader):
#             print(device)
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            p_y = model(x)
            loss = criterion(p_y, y)
            loss.backward()
            optimizer.step()
            acc = accuracy_b(p_y.detach().cpu(),y.detach().cpu())
#             acc = accuracy_b(p_y,y)
#             print(loss)
#             print(losses.avg)
            losses.update(loss,x.size(0))
            accs.update(acc,x.size(0))
#             if batch_idx % args.log_interval ==0:
#                 message = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                         epoch, batch_idx * len(y), len(train_loader.dataset),
#                         100. * batch_idx / len(train_loader), loss.item())
#                 print(message)
#         print(losses)
#         print(losses.avg)
#         print(losses.sum)
        logs['loss'] = losses.avg.detach().cpu()
        logs['acc'] = accs.avg.detach().cpu()
        if test_loader is not None:
            logs['val_loss'], logs['val_acc'] = test_model(model, test_loader, criterion, device)
        liveloss.update(logs)
        liveloss.send()
    print('Finished Training')
def test_model(model, test_loader, criterion, device):
    model.eval()
    losses = AverageVarMeter()
    accs = AverageVarMeter()
    for batch_idx, (x,y,z) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)
        p_y = model(x)
        loss = criterion(p_y,y)
        
        acc = accuracy_b(p_y.detach().cpu(), y.detach().cpu())
#         acc = accuracy_b(p_y, y)
        losses.update(loss,x.size(0))
        accs.update(acc,x.size(0))
#         print(losses.avg)
    return losses.avg.detach().cpu(), accs.avg.detach().cpu()

def test_model_noz(model, test_loader, criterion, device):
    model.eval()
    losses = AverageVarMeter()
    accs = AverageVarMeter()
    for batch_idx, (x,y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        p_y = model(x)
        loss = criterion(p_y,y)
        acc = accuracy_b(p_y.detach().cpu(), y.detach().cpu())
#         acc = accuracy_b(p_y, y)
        losses.update(loss,x.size(0))
        accs.update(acc,x.size(0))
    return losses.avg.detach().cpu(), accs.avg.detach().cpu()

def test_groupwise(clf, data_loader, clf_criterion, device,args):
    dlTensors = data_loader.dataset.tensors
    dldic = divide_groupsDL(dlTensors[0],dlTensors[1],dlTensors[2])
    losss = 0
    accs = 100.0
    sid = list(dldic.keys())[0]
    for did in dldic.keys():
        loss_v, acc_v = test_model_noz(clf, dldic[did],clf_criterion, device)
        print("{} : loss {} / acc {}".format(did, loss_v, acc_v))
        if args.AL_select == 'loss':
#             print(losss,loss_v,sid,did)
            if losss < loss_v:
                print(sid,did)
                sid = did
                losss = loss_v               
        else:
#             print(did,acc_v,accs)
            assert args.AL_select == 'acc'
#             print(accs > acc_v)
            if accs > acc_v:
#                 print(sid,did)
                sid = did
                accs = acc_v
    return sid, dldic
class ClassifierLR(nn.Module):

    def __init__(self, n_features, n_hidden=32, p_dropout=0.2):
        super(ClassifierLR, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.network(x))


class Classifier(nn.Module):

    def __init__(self, n_features, n_hidden=32, p_dropout=0.2):
        super(Classifier, self).__init__()
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


class Adversary(nn.Module):

    def __init__(self, n_sensitive, n_hidden=32):
        super(Adversary, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_sensitive),
        )

    def forward(self, x):
        return torch.sigmoid(self.network(x))