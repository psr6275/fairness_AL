import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from livelossplot import PlotLosses
from eval_utils import *
def train_model(model, train_loader, criterion, optimizer, device, args, test_loader = None):
    model.train()
    liveloss = PlotLosses()
#     groups = {'acccuracy': ['acc', 'val_acc'], 'log-loss': ['loss', 'val_loss']}
#     plotlosses = PlotLosses(groups=groups, outputs=outputs)
    logs = {}
    for epoch in range(args.epochs):
        model.train()
        losses = AverageVarMeter()
        accs = AverageVarMeter()
        for batch_idx, (x,y, _) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            p_y = model(x)
            loss = criterion(p_y, y)
            loss.backward()
            optimizer.step()
#             acc = accuracy_b(p_y,y.detach().cpu())
            acc = accuracy_b(p_y,y)
            
            losses.update(loss,x.size(0))
            accs.update(acc,x.size(0))
#             if batch_idx % args.log_interval ==0:
#                 message = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                         epoch, batch_idx * len(y), len(train_loader.dataset),
#                         100. * batch_idx / len(train_loader), loss.item())
#                 print(message)
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
        
#         acc = accuracy_b(p_y, y.detach().cpu())
        acc = accuracy_b(p_y, y)
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
        
        acc = accuracy_b(p_y, y)
        losses.update(loss,x.size(0))
        accs.update(acc,x.size(0))
    return losses.avg.detach().cpu(), accs.avg.detach().cpu()

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