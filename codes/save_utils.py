import pickle
import os


def save_flr(coef, save_dir = '', filename = 'FLR_model' ):
    save_path = os.path.join(save_dir, filename+'.sm')
    with open(save_path,'wb') as f:
        pickle.dump(coef,f)
    print("saved in ",save_path)

def load_flr(save_path):
    with open(save_path,'rb') as f:
        aa = pickle.load(f)
    return aa    

def save_lr(clf, save_dir = '', filename = 'lr_model'):
    res = {}
    res['coef'] = clf.coef_
    res['intercept'] = clf.intercept_
    save_path = os.path.join(save_dir, filename+'.sm')
    with open(save_path,'wb') as f:
        pickle.dump(res,f)
    print("saved in ",save_path)

def load_lr(save_path):
    with open(save_path,'rb') as f:
        aa = pickle.load(f)
    return aa

def save_testdata(Xte, yte,zte,data = 'adult',save_dir = ''):
    res = {}
    res['Xte'] = Xte
    res['yte'] = yte
    res['zte'] = zte
    
    save_path = os.path.join(save_dir,data+'_testset.te')
    save_path2 = os.path.join(save_dir,data+'_testX.te')
    with open(save_path,'wb') as f:
        pickle.dump(res,f)
    print("saved in ", save_path)
    
    with open(save_path2,'wb') as f:
        pickle.dump(res['Xte'],f)
    print("saved in ", save_path2)

def load_testdata(save_path):
    with open(save_path,'rb') as f:
        aa = pickle.load(f)
    return aa['Xte'], aa['yte'], aa['zte']

def load_nparray(save_path):
    with open(save_path,'rb') as f:
        aa = pickle.load(f)
    return aa

def save_prediction(pred,data,save_dir = '', model = 'flr'):
    save_path = os.path.join(save_dir, data +'_'+model+'_pred.pr')
    with open(save_path,'wb') as f:
        pickle.dump(pred,f)
        print("saved in ", save_path)
    
