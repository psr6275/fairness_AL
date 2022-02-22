import torch
import torch.nn as nn
from train_utils import test_model

def select_data(trainset, nb_stolen):  #attack용 데이터 중 원하는 개수 추출
    x = trainset.data
    nb_stolen = np.minimum(nb_stolen, x.shape[0])
    rnd_index = np.random.choice(x.shape[0], nb_stolen, replace=False)
    sampler = SubsetRandomSampler(rnd_index)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, sampler=sampler)
    return trainloader

def query_label(x, victim_clf, victim_clf_fake, tau, use_probability=False):   #tau 조건에 따라 net 또는 fakenet 불러옴
    victim_clf.to(device).eval()
    victim_clf_fake.to(device).eval()
    labels_in = victim_clf(x)
    labels_out = victim_clf_fake(x)
    cond_in = torch.max(labels_in, dim=1).values>tau
    labels = (labels_in*cond_in.view(-1,1)+labels_out*(~cond_in.view(-1,1)))
    
    if not use_probability:
        labels = torch.argmax(labels, axis=1)
        #labels = to_categorical(labels, nb_classes)
    
    victim_clf.cpu()
    victim_clf_fake.cpu()
    
    return labels

def train_stmodel(steal_loader, thieved_clf, criterion, use_probability, optimizer, victim_clf, victim_clf_fake, tau, nb_stolen, batch_size, epochs, device, testloader=None, save_dir = "../results", save_model="cifar_stmodel.pth"):
    
    thieved_clf.to(device)
    liveloss_tr = PlotLosses()
    logs_clf = {}
    best_acc = 0.0
    
    save_model = str(tau)+"_"+save_model
    for epoch in range(epochs):
        losses = AverageVarMeter()
        thieved_clf.train()
        for x,y in steal_loader:
            x = x.to(device)
            
            clf.zero_grad()
            
            out = thieved_clf(x)
            fake_label = query_label(x, victim_clf, victim_clf_fake, tau, use_probability)

            loss = criterion(out,fake_label)
            
            loss.backward()
            optimizer.step()
            
            losses.update(loss,x.size(0))
            del out,x,y,loss, fake_label
        logs_clf['loss'] = losses.avg.detach().cpu()
        if test_loader:
            logs_clf['val_loss'],logs_clf['val_acc'] = test_model(thieved_clf, test_loader, criterion, device, best_acc, save_dir, save_model)
            if best_acc<logs_clf['val_acc']:
                best_acc = logs_clf['val_acc']

        liveloss_tr.update(logs_clf)
        liveloss_tr.send()  
    return thieved_clf, logs_clf
    
    
def test_st_model(model,model_fake, test_loader, criterion, device,pred_prob = False):
    
    model.eval()
    model_fake.eval()
    
    for tau in np.arange(0.1, 1.1, 0.1):
        losses = AverageVarMeter()
        accs = AverageVarMeter()
        for batch_idx, (x,y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)

            p_y = query_label(x,model, model_fake,tau,use_probability=True)
            if pred_prob:
                p_y = torch.log(p_y)
            loss = criterion(p_y,y)

            acc = accuracy(p_y.detach().cpu(), y.detach().cpu())

            losses.update(loss,x.size(0))
            accs.update(acc[0],x.size(0))
    #         print(acc)
            del x, y, p_y, acc, loss
        print("Accuracy/Loss for tau {:.1f} : {:.2f}/{:.4f}".format(tau, acc,loss))

    return losses.avg.detach().cpu(), accs.avg.detach().cpu()