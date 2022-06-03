import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

import copy

class AverageVarMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val=0
        self.sum2 = 0
        self.avg=0
        self.sum=0
        self.count=0

    def update(self,val,n=1):
        self.val = val
        self.sum2 += (val**2)*n
        self.sum +=val*n
        self.count +=n
        self.avg = self.sum / self.count
class LogsAVG:
    def __init__(self, logs_list):
        self.logs_list = logs_list
        self.train = {}
        self.test = {}
        self.res_cat = logs_list[0].res_cat
        self.crit_list = logs_list[0].crit_list

        self.initialize()        
        self._get_average()
        self._obtain_worst_perf('train')
        self._obtain_worst_perf('test')

        
    def initialize(self,crit_list=None):
        if crit_list is None:
            crit_list = self.crit_list
        
        for cat in self.res_cat:
            self.train[cat] = {}
            self.test[cat] = {}
            for cl in crit_list:
                self.train[cat][cl] = []
                self.test[cat][cl] = []

    def _get_average(self):                
        for cat in self.res_cat:
            for cl in self.crit_list:
                for lgc in self.logs_list:
                    self.train[cat][cl].append(torch.tensor(lgc.train[cat][cl]).flatten())
                    self.test[cat][cl].append(torch.tensor(lgc.test[cat][cl]).flatten())
#                 print(torch.stack(self.train[cat][cl],dim=0))
                self.train[cat][cl] = torch.mean(torch.stack(self.train[cat][cl],dim=0), dim=0)
                self.test[cat][cl] = torch.mean(torch.stack(self.test[cat][cl],dim=0), dim=0)
                
        
    def _obtain_worst_perf(self, mode = 'train', return_val = False):
        if mode == 'train':
            logs = self.train
        elif mode == 'test':
            logs = self.test
        else:
            print("not implementation error")
        
        logs['w'] = {}
        res_keys = list(logs['all'].keys())
        num_data = len(logs['all'][res_keys[0]])
        
        for cl in res_keys:
            logs['g0'][cl] = torch.tensor(logs['g0'][cl])
            logs['g1'][cl] = torch.tensor(logs['g1'][cl])
            if cl == 'loss':
                logs['w'][cl] = torch.max(logs['g0'][cl],logs['g1'][cl])
            else:
                logs['w'][cl] = torch.min(logs['g0'][cl],logs['g1'][cl])
        if return_val:
            return logs['w']
        else:
            return
        
class LogsCLS:
    def __init__(self,res_cat = ['all','g0','g1'],
                 crit_list = ['acc','loss','weighted','micro','macro','roc_auc']):
        self.crit_list = crit_list
        self.res_cat = res_cat
        self.train = {}
        self.test = {}
        self.initialize()
        
    def initialize(self,crit_list=None):
        if crit_list is None:
            crit_list = self.crit_list
        
        for cat in self.res_cat:
            self.train[cat] = {}
            self.test[cat] = {}
            for cl in crit_list:
                self.train[cat][cl] = []
                self.test[cat][cl] = []
        
    def append_result(self, res_dic_list, mode = 'train'):
        if mode == 'train':
            logs = self.train
        elif mode == 'test':
            logs = self.test
        else:
            print("not implementation error")
            
        for it,res_dic in enumerate(res_dic_list):           
            for rk in res_dic.keys():
                logs[self.res_cat[it]][rk].append(res_dic[rk])
                
    def obtain_worst_perf(self, mode = 'train'):
        if mode == 'train':
            logs = self.train
        elif mode == 'test':
            logs = self.test
        else:
            print("not implementation error")
        
        logs['w'] = {}
        res_keys = list(logs['all'].keys())
        num_data = len(logs['all'][res_keys[0]])
        
        for cl in res_keys:
            logs['g0'][cl] = torch.tensor(logs['g0'][cl])
            logs['g1'][cl] = torch.tensor(logs['g1'][cl])
            if cl == 'loss':
                logs['w'][cl] = torch.maximum(logs['g0'][cl],logs['g1'][cl])
            else:
                logs['w'][cl] = torch.minimum(logs['g0'][cl],logs['g1'][cl])
        return logs['w']

        
        
def binary_scores(output, target, prob=True, thres = 0.5):
    score_dic = {}
    if prob:
        pred = torch.tensor(output>thres, dtype=torch.float32)
    else:
        pred = output
    score_dic['weighted'] = f1_score(target, pred, average='weighted')*100
    score_dic['macro'] = f1_score(target, pred, average='macro')*100
    score_dic['micro'] = f1_score(target, pred, average='micro')*100
    score_dic['accuracy'] = accuracy_b(pred, target).item()
    score_dic['roc_auc'] = roc_auc_score(target,output)*100
    
    print('weighted: ', score_dic['weighted'])
    print('macro: ',score_dic['macro'])
    print('micro: ',score_dic['micro'])            
    print('accuracy: ', score_dic['accuracy'])      
    if prob:
        print("roc auc: ", score_dic['roc_auc'])
    return score_dic
        
def accuracy(output, target, topk=(1,)):
    '''Compute the top1 and top k error'''
    maxk = max(topk)
    batch_size = target.size(0)
    _,pred = output.topk(maxk,1,True,True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul(100.0 / batch_size))
    return res

def accuracy_b(output, target, thr = 0.5):
    '''Compute the top1 and top k error'''
    
    batch_size = target.size(0)
    pred = torch.tensor(output>thr,dtype = torch.float32)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

   
    correct_k = correct.view(-1).float().sum(0)
    res=correct_k.mul(100.0 / batch_size)
    
    return res