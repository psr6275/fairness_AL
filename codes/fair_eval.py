import numpy as np
from fair_eval import *
from load_data import *

def calculate_misclassification(pred, y, xs):
    y = y.flatten()
    mis0 = sum((pred==-y)*(xs==0))
    mis1 = sum((pred==-y)*(xs==1))
    s0 = sum(xs==0)
    s1 = sum(xs==1)
    prule = min((mis0/s0)/(mis1/s1),(mis1/s1)/(mis0/s0))*100
    return prule

def calculate_group_misclassification(pred, y, xs,outputs = False):
    y = y.flatten()
    mis_idx = (pred != y)
    
    if len(xs.shape)==1:
        xs = xs.reshape((-1,1))
    
    mis = []
    miss = []
    misr = []
    for i in range(xs.shape[1]):
        xsv = np.unique(xs[:,i])
        for j in xsv:
            xs_idx = (j == xs[:,i])
            mis_g = mis_idx * xs_idx
            mis.append(sum(mis_g))
            miss.append(sum(xs_idx))
            if sum(mis_g)>0:
                misr.append(sum(mis_g)/sum(xs_idx))

    prule = min(misr)/max(misr)*100
#     mis0 = sum((pred==-y)*(xs==0))
#     mis1 = sum((pred==-y)*(xs==1))
#     s0 = sum(xs==0)
#     s1 = sum(xs==1)
#     prule = min((mis0/s0)/(mis1/s1),(mis1/s1)/(mis0/s0))*100
    if outputs:
        return prue, mis, miss, misr
    else:
        return prule    
    

def calculate_mistreatment(pred, y, xs,cond = 1):
    """
        cond = 1 : False negative rate
        cond = -1: False positive rate
    """
    xs = xs.flatten()
    y = y.flatten()

    if min(y)==0:
        y_ = (y*2)-1
        pred_ = pred*2-1
    else:
        y_ = y
        pred_ = pred
    assert cond in [-1,1]
    
    fr0 = sum((pred_==-cond)*(y_==cond)*(xs==0))
    fr1 = sum((pred_==-cond)*(y_==cond)*(xs==1))
    s0 = sum((y_==cond)*(xs==0))
    s1 = sum((y_==cond)*(xs==1))

    
    prule = min((fr0/s0)/(fr1/s1),(fr1/s1)/(fr0/s0))*100
    
    return prule

def calculate_group_mistreatment(pred, y, xs,cond = 1, outputs = False):
    """
        cond = 1 : False negative rate
        cond = -1: False positive rate
    """
    if len(xs.shape)==1:
        xs = xs.reshape((-1,1))
    y = y.flatten()
    pred = pred.flatten()

#     if min(y)==0:
#         y_ = (y*2)-1
#         pred_ = pred*2-1
#     else:
#         y_ = y
#         pred_ = pred
    assert cond in np.unique(y)
    cond_idx = (y == cond)
    false_idx = (pred != cond)
    
    frn = []
    frs = []
    frr = []
    for i in range(xs.shape[1]):
        xsv = np.unique(xs[:,i])
        for j in xsv:
            g_idx = (xs[:,i]==j)*cond_idx
            frn.append(sum(g_idx))
            frs.append(sum(false_idx*g_idx))
            if sum(g_idx)>0:
                frr.append(sum(false_idx*g_idx)/sum(g_idx))
        
    prule = min(frr)/max(frr)*100
    if outputs:
        return prule, frs, frn, frr
    else:
        return prule

def calculate_impact(pred,y,xs):
    
    y = y.flatten()
    idx_yps0 = (pred==1)*(xs==0)
    idx_yps1 = (pred==1)*(xs==1)
#     idx_yns0 = (pred==-1)*(xs==0)
#     idx_yns1 = (pred==-1)*(xs==1)
    
    s0sum = sum(xs==0)
    s1sum = sum(xs==1)
    
    prule = min((sum(idx_yps1)/s1sum)/(sum(idx_yps0)/s0sum),(sum(idx_yps0)/s0sum)/(sum(idx_yps1)/s1sum))*100
    return prule

def calculate_group_impact(pred,y,xs, cond = 1,outputs = False):
    
    y = y.flatten()
    if len(xs.shape)==1:
        xs = xs.reshape((-1,1))
    pred = pred.flatten()
    assert cond in np.unique(y)
    pred_idx = pred ==cond
    imp = []
    impn = []
    impr = []
    for i in range(xs.shape[1]):
        xsv = np.unique(xs[:,i])
        for j in xsv:
            xs_idx = xs[:,i] == j
            imp.append(sum(xs_idx*pred_idx))
            impn.append(sum(xs_idx))
            if sum(xs_idx):
                impr.append(sum(xs_idx*pred_idx)/sum(xs_idx))
    prule = min(impr)/max(impr)*100
    if outputs:
        return prule, imp, impn, impr
    else:
        return prule

def calculate_prule_clf(pred,y,xs):
    pred = pred.flatten()
    y = y.flatten()
    xs = xs.flatten()
    print("disparate impact: ", calculate_impact(pred,y,xs))
    print("disparate misclassification rate: ", calculate_misclassification(pred,y,xs))
    print("disparate false positive rate:", calculate_mistreatment(pred,y,xs,cond=-1))
    print("disparate false negative rate:", calculate_mistreatment(pred,y,xs,cond=1))

def calculate_group_prule_clf(pred,y,xs):
    pred = pred.flatten()
    y = y.flatten()
    ylh = np.unique(y)
    for cond in ylh:
        print("disparate impact for {}: {}".format(cond, calculate_group_impact(pred,y,xs,cond)))
        print("disparate false rate for {}: {}".format(cond,calculate_group_mistreatment(pred,y,xs,cond)))
    print("disparate misclassification rate: ", calculate_group_misclassification(pred,y,xs))
    
def calculate_odds_clf(pred,y,xs):
    xs = xs.flatten()
    y = y.flatten()
    yls = np.unique(y)
    idx_yps0 = (pred==1)*(xs==0)
    idx_yps1 = (pred==1)*(xs==1)
    
    for yl in yls:
        idx_y = y==yl
        s0sum = sum(idx_y*(xs==0))
        s1sum = sum(idx_y*(xs==1))
        prule = min((sum(idx_yps1*idx_y)/s1sum)/(sum(idx_yps0*idx_y)/s0sum),\
                    (sum(idx_yps0*idx_y)/s0sum)/(sum(idx_yps1*idx_y)/s1sum))*100
        print("equalized opportunity for {} : {}".format(yl,prule))

def calculate_group_odds(pred,y,xs,cond = 1,outputs = False):
    
    y = y.flatten()
    yls = np.unique(y)
    if len(xs.shape)==1:
        xs = xs.reshape((-1,1))
    pred = pred.flatten()
    assert cond in yls
    pred_idx = pred == cond
    y_idx = y==cond
    odd = []
    oddn = []
    oddr = []
    for i in range(xs.shape[1]):
        xsv = np.unique(xs[:,i])
        for j in xsv:
            xs_idx = xs[:,i] ==j
            odd.append(sum(xs_idx*pred_idx*y_idx))
            odd.append(sum(xs_idx*y_idx))
            if sum(xs_idx*y_idx)>0:
                oddr.append(sum(xs_idx*pred_idx*y_idx)/sum(xs_idx*y_idx))
    prule = min(oddr)/max(oddr)*100
    if outputs:
        return prule, odd, oddn, oddr
    else:
        return prule
    
def calculate_group_odds_clf(pred,y,xs):        
    y = y.flatten()
    yls = np.unique(y)
    for cond in yls:
        prule = calculate_group_odds(pred,y,xs,cond = 1,outputs = False)
        print("Equalized Opportunity for {} : {}".format(cond,prule))
        
def calculate_parity_reg(pred,y,xs,thrs = None):
    xs = xs.flatten()
    y = y.flatten()
    if thrs is None:
        thrs = np.mean(y)
    idx_yps0 = (pred>thrs)*(xs==0)
    idx_yps1 = (pred>thrs)*(xs==1)
    s0sum = sum(xs==0)
    s1sum = sum(xs==1)
    
    prule = min((sum(idx_yps1)/s1sum)/(sum(idx_yps0)/s0sum),(sum(idx_yps0)/s0sum)/(sum(idx_yps1)/s1sum))*100
    print("disparate parity for threshold {}: {}".format(thrs, prule))

def calculate_group_loss(loss_fn, pred, y, xs):
    xs =xs.flatten()
    y = y.flatten()
    s0sum = sum(xs==0)
    s1sum = sum(xs==1)
    print("loss function: ", loss_fn.__name__)
    for i in range(2):
        lv = loss_fn(pred[xs==i],y[xs==i])
        print("loss value for group {}: {}".format(i,lv))

def transform_dum2cat(xs,nil = None):
    # xs should start with 0!
    if len(xs.shape)==1:
        xs = xs.reshape((-1,1))
    if type(xs) == torch.Tensor:
        xs = xs.clone().detach().cpu().numpy()
    xsv = np.zeros(xs.shape[0])
    if nil is None:
#         print(xs[:,0])
        nil = [len(np.unique(xs[:,i])) for i in range(xs.shape[1])]
    
    for i in range(xs.shape[1]):
        if i>0:
            ni = np.prod(nil[:i-1])
        else:
            ni = 0
        xsv += xs[:,i]+ni
    return xsv.reshape((-1,1))

def transform_target(pred,thr = 0.5):
    if len(pred.shape)==1:
        pred_ = pred>thr
    elif pred.shape[1] ==1:
        pred_ = pred>thr
    else:
        pred_ = np.argmax(pred)
    return pred_
def calculate_group_acc(pred, y, xs, verbose = True):
    if len(xs.shape)==1:
        xs = xs.reshape((-1,1))
    xsv = transform_dum2cat(xs)
    mis_idx = (pred !=y)
    gacc = []
    for i in range(np.unique(xsv)):
        z_idx = xsv == i
        if sum(z_idx)>0:
            gacc.append(sum(mis_idx*z_idx)/sum(z_idx)*100)
            if verbose:
                print("Acc for group {}: {}".format(i,gacc[-1]))
    return gacc
        
def l2_loss(pred,y):
    return np.mean((pred-y)**2)

eps = np.finfo(float).eps

def bce_loss(pred_,y):
    return -np.mean(y*np.log(pred_+eps)+(1-y)*np.log(1-pred_+eps))

def calculate_overall_accuracy(pred,y):
    pred = pred.flatten()
    return np.sum(pred==y)/len(pred)

