import os
import pdb

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal  # generating synthetic data

import pickle

from collections import defaultdict
from sklearn import feature_extraction
from sklearn import preprocessing
from random import seed, shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import random

import numpy as np

train_frac = 0.7

def dl2nparray(dataloader):
    arrs = []
    for ts in dataloader.dataset.tensors:
        arrs.append(ts.detach().cpu().numpy()) 
    return tuple(arrs)

def dataset_description(X,y,z,dataname='train'): 

    if len(y.shape)==2:
        y = y.flatten()
    if len(z.shape) ==2:
        z = _transform_dum2cat(z)
        z = z.flatten()
        lenz = len(np.unique(z))
    else:
        lenz = 2
    
    mrat_y = compute_max_ratio(y)
    mrat_z = compute_max_ratio(z)
    
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,sharey=True,figsize=(10,3))
    fig.subplots_adjust(top=0.8)
    fig.suptitle(dataname+" input data shape: "+str(X.shape))
    ax1.set_title("max label %.2f" % mrat_y)
    ax1.hist(y)
    ax2.set_title("max group %.2f" % mrat_z)
    ax2.hist(z)
    ax3.set_title("label* %d +group " % lenz )
    ax3.hist(lenz*y+z)
    plt.show()

def compute_max_ratio(arr):    
    arr = arr.flatten()
    arr_u = np.unique(arr)
    max_rat = 0.0
    arr_n = len(arr)
    for au in arr_u:
        rat = sum(arr==au)/arr_n
        if rat>max_rat:
            max_rat = rat
    return max_rat
    
def input_uniqueness(X1,X2):
    Xt = np.row_stack([X1,X2])
    n1 = Xt.shape[0]
    n2 = np.unique(Xt,axis=0).shape[0]
    print(n1, n2)
    if n1==n2:
        print("satisfy uniqueness")
    else:
        print("redundant examples: ", str(n1-n2))
        
def initial_dataloaders(Xtr,ytr,Ztr,Xte,yte,Zte,N1,args):
    Xtr1, ytr1, Ztr1, Xtr2, ytr2, Ztr2 = split_initial_dataset(Xtr,ytr,Ztr,N_init=N1,random_state=42)
    train_data = NPsDataSet(Xtr1, ytr1, Ztr1)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=False)
    zZtr2 = _transform_dum2cat(Ztr2)
    select_data = NPsDataSet(Xtr2, ytr2, zZtr2)
    select_loader = DataLoader(select_data, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_data = NPsDataSet(Xte, yte, Zte)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, drop_last=False)
    return train_loader,select_loader,test_loader

def obtain_newDS_rev(train_loader, select_loader, sidx, batch_size = 32):
    ds1 = []
    for i, tss in enumerate(train_loader.dataset.tensors):
        ds1.append(np.append(tss,select_loader.dataset.tensors[i][sidx],axis = 0))
    ds2 = []
    for tss in select_loader.dataset.tensors:
        ds2.append(np.delete(tss,sidx.detach().cpu(),axis=0))
    tr_loader = DataLoader(NPsDataSet(ds1[0],ds1[1],ds1[2]),batch_size = batch_size, shuffle=True)
    se_loader = DataLoader(NPsDataSet(ds2[0],ds2[1],ds2[2]),batch_size = batch_size, shuffle=False)
    
    return tr_loader, se_loader

def obtain_newDS(train_loader, select_loader,ses, sidx, batch_size = 32):
    ds1 = []
    for i, tss in enumerate(train_loader.dataset.tensors):
        ds1.append(np.append(tss,ses[i],axis = 0))
    ds2 = []
    for tss in select_loader.dataset.tensors:
        ds2.append(np.delete(tss,sidx.detach().cpu(),axis=0))
    tr_loader = DataLoader(NPsDataSet(ds1[0],ds1[1],ds1[2]),batch_size = batch_size, shuffle=True)
    se_loader = DataLoader(NPsDataSet(ds2[0],ds2[1],ds2[2]),batch_size = batch_size, shuffle=False)
    
    return tr_loader, se_loader

def make_dataloader(x,y,z, batch_size, shuffle):
    return DataLoader(NPsDataSet(x,y,z), batch_size = batch_size, shuffle=shuffle)

def split_initial_dataset(Xtr,ytr,Ztr,N_init=300,random_state=42):
    np.random.seed(random_state)
    n_examples = Xtr.shape[0]
    idx = np.random.permutation(n_examples)
    
    init_idx = idx[:N_init]
    AL_idx = idx[N_init:]
    
    Xinit = Xtr[init_idx]
    yinit = ytr[init_idx]
    Zinit = Ztr[init_idx]
    XAL = Xtr[AL_idx]
    yAL = ytr[AL_idx]
    ZAL = Ztr[AL_idx]
    return Xinit, yinit, Zinit, XAL, yAL, ZAL

class NPsDataSet(TensorDataset):

    def __init__(self, *dataarrays):
        tensors = (torch.tensor(da).clone().detach().float() for da in dataarrays)
        super(NPsDataSet, self).__init__(*tensors)
def train_valid_split(train_loader,train_num,val_ratio = 0.2,random_seed = 7):
    random.seed(random_seed)
    trainDS = train_loader.dataset
    valid_num = int(train_num*val_ratio)
    sp_idx = list(range(train_num))
    random.shuffle(sp_idx)
    sp_idx = sp_idx[:valid_num]
    trds = []
    valds = []
    for tss in train_loader.dataset.tensors:
        trds.append(np.delete(tss,sp_idx,axis=0))
        valds.append(tss[sp_idx])
        
    trloader = DataLoader(NPsDataSet(trds[0],trds[1],trds[2]),batch_size = train_loader.batch_size,shuffle=True)
    valloader = DataLoader(NPsDataSet(valds[0],valds[1],valds[2]),batch_size =train_loader.batch_size,shuffle=False)
    return trloader, valloader
    
def convert_object_type_to_category(df):
    """Converts columns of type object to category."""
    df = pd.concat([df.select_dtypes(include=[], exclude=['object']),
                  df.select_dtypes(['object']).apply(pd.Series.astype, dtype='category')
                  ], axis=1).reindex(df.columns, axis=1)
    return df    
    
def divide_groupsDL(x,y,z, batch_size = 32):
#     print(zs)
    zs = z.flatten()
    zsu = np.unique(zs)
    dataloaders = {}
    for zu in zsu:
        z_idx = zs == zu
        xx = x[z_idx]
        yy = y[z_idx]
        daloader = DataLoader(NPsDataSet(xx,yy),batch_size = 32,shuffle=False)
        dataloaders[zu]= daloader
    return dataloaders
        
def load_singlefold(savepath):
#     filepath = savepath+'_%d.npz'%i
    with open(savepath,'rb') as f:
        Xtr,Xte,ytr,yte,Ztr,Zte = pickle.load(f)
    return Xtr,Xte,ytr,yte,Ztr,Zte

def load_data_with_name(dataname, **data_args):
    
    if dataname == 'adult':
        load_fn = load_adult_data        
    elif dataname == 'bank':
        load_fn = load_bank_data
    elif dataname == 'german':
        load_fn = load_german_data
    elif dataname == 'lsac':
        load_fn = load_lsac_data
    elif dataname == 'compas':
        load_fn = load_compas_data
    else:
        raise NotImplementedError("need to be implemented for other datasets")
    
    Xtr, Xte, ytr, yte, Ztr, Zte = load_fn(**data_args)
    ytr = ytr.astype('float32').reshape((-1,1))
    yte = yte.astype('float32').reshape((-1,1))
    
    return Xtr, Xte, ytr, yte, Ztr, Zte

def load_adult_data(filepath = '../data/adult_proc.csv',svm=False,random_state=42, intercept=False,sensitive_attrs=None):
    df = pd.read_csv(filepath)
    attrs = df.columns
    target_attr = ['income']
    if sensitive_attrs is None:
        sensitive_attrs = ['sex']
    attrs_to_ignore = ['race','sex','marital-status','Unnamed: 0']
    attrs_for_classification = set(attrs) - set(attrs_to_ignore) - set(target_attr)
    X = df[attrs_for_classification].values
    y = df[target_attr].values
    Z = df[sensitive_attrs].values
    
    if svm:
        y = y*2-1
    
    n = X.shape[0]  # Number of examples

    
    # Create train test split
    tr_idx, te_idx = _get_train_test_split(n, train_frac, random_state)
    Xtr, Xte, ytr, yte, Ztr, Zte = _apply_train_test_split(X, y, Z,
                                                           tr_idx, te_idx)
    
    mm = MinMaxScaler()
    Xtr = mm.fit_transform(Xtr)
    Xte = mm.transform(Xte)
    
    if intercept:
        Xtr, Xte = _add_intercept(Xtr, Xte)
    
    return Xtr, Xte, ytr, yte, Ztr, Zte
    
def load_compas_data(filepath = '../data/compas_proc.csv',svm=False,random_state=42, intercept=False, sensitive_attrs = None):
    """
        race_map = {'Black':1.0,'White':0.0,'Other':2.0}
        sex_map = {'Female':1.0,'Male':0.0}
        ccd_map = {'F':1.0,'M':0.0}
        # age_map = {'Greater than 45':2.0, '25 - 45':1.0, 'Less than 25':0.0}
        recid_map = {'Yes':1.0,'No':0.0}
    """
    df = pd.read_csv(filepath)
    attrs = df.columns
    target_attr = ['is_recid']
    if sensitive_attrs is None:
        sensitive_attrs = ['race']
    attrs_to_ignore = ['race','sex']
    attrs_for_classification = set(attrs) - set(attrs_to_ignore) - set(target_attr)
    X = df[attrs_for_classification].values
    y = df[target_attr].values
    Z = df[sensitive_attrs].values
    
    if svm:
        y = y*2-1
    
    n = X.shape[0]  # Number of examples

    
    # Create train test split
    tr_idx, te_idx = _get_train_test_split(n, train_frac, random_state)
    Xtr, Xte, ytr, yte, Ztr, Zte = _apply_train_test_split(X, y, Z,
                                                           tr_idx, te_idx)
    
    mm = MinMaxScaler()
    Xtr = mm.fit_transform(Xtr)
    Xte = mm.transform(Xte)
    
    if intercept:
        Xtr, Xte = _add_intercept(Xtr, Xte)
    
    return Xtr, Xte, ytr, yte, Ztr, Zte

def load_lsac_data(filepath='../data/lsac_proc.csv',svm=False,random_state=42,
                   intercept=False,sensitive_attrs =None):

    """
    race_map = {'Black':1.0,'White':0.0,'Other':2.0}
    sex_map = {'Female':1.0,'Male':0.0}
    bar_map = {'Passed':1.0, 'Failed_or_not_attempted':0.0}
    part_map = {'Yes':1.0,'No':0.0}
    """
    df = pd.read_csv(filepath)
    attrs = df.columns
    target_attr = ['pass_bar']
    if sensitive_attrs is None:
        sensitive_attrs = ['sex']
    attrs_to_ignore = ['race','sex','Unnamed: 0']
    attrs_for_classification = set(attrs) - set(attrs_to_ignore) - set(target_attr)
    X = df[attrs_for_classification].values
    y = df[target_attr].values
    Z = df[sensitive_attrs].values
    
    if svm:
        y = y*2-1
    
    n = X.shape[0]  # Number of examples

    
    # Create train test split
    tr_idx, te_idx = _get_train_test_split(n, train_frac, random_state)
    Xtr, Xte, ytr, yte, Ztr, Zte = _apply_train_test_split(X, y, Z,
                                                           tr_idx, te_idx)
    
    mm = MinMaxScaler()
    Xtr = mm.fit_transform(Xtr)
    Xte = mm.transform(Xte)
    
    if intercept:
        Xtr, Xte = _add_intercept(Xtr, Xte)
    
    return Xtr, Xte, ytr, yte, Ztr, Zte
    
def load_german_data(filepath='../data/german.data-numeric', svm =False,random_state=42, intercept = False):
    """
    Read the german dataset.

    The training test set split is set to 0.8, but the actual training set size
    is the largest power of two smaller than 0.8 * n_examples and the actual
    test set is everything that remains.

    Args:
        filepath: The file path to the data
        random_state: The random seed of the train-test split

    Returns:
        Xtr, Xte, ytr, yte, Ztr, Zte: X are the features, y are the labels and Z
            are the sensitive attributes. 'tr' stands for training and 'te' for
            test set.
    """
    if filepath.endswith('ready.npz') and os.path.exists(filepath):
        return load_npz(filepath)
    df = pd.read_csv(filepath, header=None, delim_whitespace=True)

    # change label to 0/1
    cols = list(df)
    label_idx = len(cols)-1
    if svm:
        df[label_idx] = df[label_idx].map({2: -1, 1: 1})
    else:
        df[label_idx] = df[label_idx].map({2: 0, 1: 1})

    M = df.values
    Z = M[:, 9]
    Z = (Z > 25).astype(float).reshape(-1, 1)
    ix = np.delete(np.arange(24), 9)
    X = M[:, ix]
    y = M[:, -1]

    n = X.shape[0]  # Number of examples

    
    # Create train test split
    tr_idx, te_idx = _get_train_test_split(n, train_frac, random_state)
    Xtr, Xte, ytr, yte, Ztr, Zte = _apply_train_test_split(X, y, Z,
                                                           tr_idx, te_idx)

    # Whiten feature data
    mm = MinMaxScaler()
    Xtr = mm.fit_transform(Xtr)
    Xte = mm.transform(Xte)
#     Xtr, Xte = _whiten_data(Xtr, Xte)

    # Center sensitive data
#     Ztr, Zte = _center_data(Ztr, Zte)
#     Ztr = _center_data(Ztr)

    # Add intercept
    if intercept:
        Xtr, Xte = _add_intercept(Xtr, Xte)

    # Labels are already 0/1
    
        

    return Xtr, Xte, ytr, yte, Ztr, Zte

def load_bank_data(filepath = '../data/bank-full.csv',load_data_size=None,svm=False,random_state=42, intercept = False):

    """
        if load_data_size is set to None (or if no argument is provided), then we load and return the whole data
        if it is a number, say 10000, then we will return randomly selected 10K examples
    """
    bank = pd.read_csv(filepath)
    bank['marital'].loc[bank['marital']!='married']=0
    bank['marital'].loc[bank['marital']=='married']=1
    attrs = bank.columns
#     attrs = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
#        'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
#        'previous', 'poutcome', 'y'] # all attributes
    int_attrs = ['age', 'balance','day','duration','campaign','pdays','previous'] # attributes with integer values -- the rest are categorical
    sensitive_attrs = ['marital'] # the fairness constraints will be used for this feature
    attrs_to_ignore = ['marital','y'] # sex and race are sensitive feature so we will not use them in classification, we will not consider fnlwght for classification since its computed externally and it highly predictive for the class (for details, see documentation of the adult data)
    attrs_for_classification = set(attrs) - set(attrs_to_ignore)
  


    X = []
    y = []
    x_control = {}

    attrs_to_vals = {} # will store the values for each attribute for all users
    for k in attrs:
        if k in sensitive_attrs:
            x_control[k] = []
        elif k in attrs_to_ignore:
            pass
        else:
            attrs_to_vals[k] = []
        
#     print(x_control)
    if svm:
        bank['y'] = bank['y'].map({"no": -1, "yes": 1})
    else:
        bank['y'] = bank['y'].map({"no": 0, "yes": 1})
    y = bank.values[:,-1]
    
    for i in range(len(bank)):
        line = bank.iloc[i].values
#         class_label = line[-1]
#         if class_label in ["no"]:
#             class_label = -1
#         elif class_label in ["yes"]:
#             class_label = +1
#         else:
#             raise Exception("Invalid class label value")

#         y.append(class_label)
#         if i ==0:
#             print(line)
#             print(len(line))
        for j in range(0,len(line)):
            attr_name = attrs[j]
            attr_val = line[j]
                # reducing dimensionality of some very sparse features
#             if attr_name == 'previous':
#                 print(attr_name,": ",line)
#             if i ==0:
#                 print(attr_name, attr_val)
            if attr_name in sensitive_attrs:
                x_control[attr_name].append(attr_val)
            elif attr_name in attrs_to_ignore:
                pass
            else:
                attrs_to_vals[attr_name].append(attr_val)
#     print(bank['previous'])
#     print(attrs_to_vals['previous'])
    def convert_attrs_to_ints(d): # discretize the string attributes
        for attr_name, attr_vals in d.items():
            if attr_name in int_attrs: continue
            uniq_vals = sorted(list(set(attr_vals))) # get unique values

            # compute integer codes for the unique values
            val_dict = {}
            for i in range(0,len(uniq_vals)):
                val_dict[uniq_vals[i]] = i

            # replace the values with their integer encoding
            for i in range(0,len(attr_vals)):
                attr_vals[i] = val_dict[attr_vals[i]]
            d[attr_name] = attr_vals

    
    # convert the discrete values to their integer representations
    convert_attrs_to_ints(x_control)
    convert_attrs_to_ints(attrs_to_vals)


    # if the integer vals are not binary, we need to get one-hot encoding for them
    for attr_name in attrs_for_classification:
        
        attr_vals = attrs_to_vals[attr_name]
#         print(attr_name)
#         print(attr_vals[0])
        if attr_name in int_attrs or attr_name == "native_country": # the way we encoded native country, its binary now so no need to apply one hot encoding on it
#             if attr_vals == []:
#                 print(attr_name)
            X.append(attr_vals)
#             print(attr_name,attr_vals)

        else:
            
            attr_vals, index_dict = get_one_hot_encoding(attr_vals)
#             print(attr_vals[0])
#             print(attr_vals.shape)
            if attr_vals.shape==(45211,):
                attr_vals=attr_vals.reshape(45211,1)
            for inner_col in attr_vals.T:                
                X.append(inner_col)
#                 if inner_col == []:
#                     print(attr_name)
                
    for i,xx in enumerate(X):
        if np.array(xx).shape != (45211,):
            print(i,np.array(xx).shape)
#     print(len(X),len(X[0]))
#     print(X[:3])
    # convert to numpy arrays for easy handline
    X = np.array(X, dtype=float).T
    y = np.array(y, dtype = float)
    for k, v in x_control.items(): x_control[k] = np.array(v, dtype=float)
        
    # shuffle the data
    perm = list(range(0,len(y))) # shuffle the data before creating each fold
    shuffle(perm)
    X = X[perm]
    y = y[perm]
    for k in x_control.keys():
        x_control[k] = x_control[k][perm]

    # see if we need to subsample the data
    if load_data_size is not None:
        print ("Loading only %d examples from the data" % load_data_size)
        X = X[:load_data_size]
        y = y[:load_data_size]
        for k in x_control.keys():
            x_control[k] = x_control[k][:load_data_size]
#     print(x_control)
    n = X.shape[0]
    # Create train test split
    Z = np.expand_dims(x_control[k],axis=-1)
    tr_idx, te_idx = _get_train_test_split(n, train_frac, random_state)
    Xtr, Xte, ytr, yte, Ztr, Zte = _apply_train_test_split(X, y, Z,
                                                           tr_idx, te_idx)

    # Whiten feature data
#     Xtr, Xte = _whiten_data(Xtr, Xte)
    mm = MinMaxScaler()
    Xtr = mm.fit_transform(Xtr)
    Xte = mm.transform(Xte)

    # Center sensitive data
#     Ztr, Zte = _center_data(Ztr, Zte)

    # Add intercept
    if intercept:
        Xtr, Xte = _add_intercept(Xtr, Xte)

    # Labels are already 0/1
    
        

    return Xtr, Xte, ytr, yte, Ztr, Zte
    


def load_adult_data_prev(load_data_size=None,svm = False,random_state=42, intercept = False):

    """
        if load_data_size is set to None (or if no argument is provided), then we load and return the whole data
        if it is a number, say 10000, then we will return randomly selected 10K examples
    """

    attrs = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country'] # all attributes
    int_attrs = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week'] # attributes with integer values -- the rest are categorical
    sensitive_attrs = ['sex'] # the fairness constraints will be used for this feature
    attrs_to_ignore = ['sex', 'race','fnlwgt'] # sex and race are sensitive feature so we will not use them in classification, we will not consider fnlwght for classification since its computed externally and it highly predictive for the class (for details, see documentation of the adult data)
    attrs_for_classification = set(attrs) - set(attrs_to_ignore)

    # adult data comes in two different files, one for training and one for testing, however, we will combine data from both the files
    data_files = ["../data/adult.data", "../data/adult.test"]



    X = []
    y = []
    x_control = {}

    attrs_to_vals = {} # will store the values for each attribute for all users
    for k in attrs:
        if k in sensitive_attrs:
            x_control[k] = []
        elif k in attrs_to_ignore:
            pass
        else:
            attrs_to_vals[k] = []

    for f in data_files:
#         check_data_file(f)

        for line in open(f):
            line = line.strip()
            if line == "": continue # skip empty lines
            line = line.split(", ")
            if len(line) != 15 or "?" in line: # if a line has missing attributes, ignore it
                continue

            class_label = line[-1]
            if class_label in ["<=50K.", "<=50K"]:
                class_label = -1
            elif class_label in [">50K.", ">50K"]:
                class_label = +1
            else:
                raise Exception("Invalid class label value")

            y.append(class_label)


            for i in range(0,len(line)-1):
                attr_name = attrs[i]
                attr_val = line[i]
                # reducing dimensionality of some very sparse features
                if attr_name == "native_country":
                    if attr_val!="United-States":
                        attr_val = "Non-United-Stated"
                #elif attr_name == "race":
                #    if attr_val!="White":
                #        attr_val = "Non-White"
                elif attr_name == "education":
                    if attr_val in ["Preschool", "1st-4th", "5th-6th", "7th-8th"]:
                        attr_val = "prim-middle-school"
                    elif attr_val in ["9th", "10th", "11th", "12th"]:
                        attr_val = "high-school"

                if attr_name in sensitive_attrs:
                    x_control[attr_name].append(attr_val)
                elif attr_name in attrs_to_ignore:
                    pass
                else:
                    attrs_to_vals[attr_name].append(attr_val)

    def convert_attrs_to_ints(d): # discretize the string attributes
        for attr_name, attr_vals in d.items():
            if attr_name in int_attrs: continue
            uniq_vals = sorted(list(set(attr_vals))) # get unique values

            # compute integer codes for the unique values
            val_dict = {}
            for i in range(0,len(uniq_vals)):
                val_dict[uniq_vals[i]] = i

            # replace the values with their integer encoding
            for i in range(0,len(attr_vals)):
                attr_vals[i] = val_dict[attr_vals[i]]
            d[attr_name] = attr_vals

    
    # convert the discrete values to their integer representations
    convert_attrs_to_ints(x_control)
    convert_attrs_to_ints(attrs_to_vals)


    # if the integer vals are not binary, we need to get one-hot encoding for them
    for attr_name in attrs_for_classification:
        attr_vals = attrs_to_vals[attr_name]
        if attr_name in int_attrs or attr_name == "native_country": # the way we encoded native country, its binary now so no need to apply one hot encoding on it
            X.append(attr_vals)

        else:            
            attr_vals, index_dict = get_one_hot_encoding(attr_vals)
            for inner_col in attr_vals.T:                
                X.append(inner_col) 


    # convert to numpy arrays for easy handline
    X = np.array(X, dtype=float).T
    y = np.array(y, dtype = float)
    for k, v in x_control.items(): x_control[k] = np.array(v, dtype=float)
        
    # shuffle the data
    perm = list(range(0,len(y))) # shuffle the data before creating each fold
    shuffle(perm)
    X = X[perm]
    y = y[perm]
    for k in x_control.keys():
        x_control[k] = x_control[k][perm]

    # see if we need to subsample the data
    if load_data_size is not None:
        print ("Loading only %d examples from the data" % load_data_size)
        X = X[:load_data_size]
        y = y[:load_data_size]
        for k in x_control.keys():
            x_control[k] = x_control[k][:load_data_size]
    print(x_control)
    if not svm:
        y = np.array((y+1)/2,dtype=np.uint32)
        print("for others min: ",min(y), "and max: ",max(y))
    else:
        print("for svm min: ",min(y), "and max: ",max(y))
    n = X.shape[0]
#     print(x_control)
    print(n)
    Z = np.zeros((n,len(x_control.keys())))
    for i,k in enumerate(x_control.keys()):
        Z[:,i] = x_control[k]
    print(Z.shape)
    tr_idx, te_idx = _get_train_test_split(n, train_frac, random_state)
    Xtr, Xte, ytr, yte, Ztr, Zte = _apply_train_test_split(X, y, Z,
                                                           tr_idx, te_idx)
    
    # Whiten feature data
#     Xtr, Xte = _whiten_data(Xtr, Xte)
    mm = MinMaxScaler()
    Xtr = mm.fit_transform(Xtr)
    Xte = mm.transform(Xte)

    # Center sensitive data
    Ztr, Zte = _center_data(Ztr, Zte)

    # Add intercept
    if intercept:
        Xtr, Xte = _add_intercept(Xtr, Xte)
    
        

    return Xtr, Xte, ytr, yte, Ztr, Zte

def load_compas_data_prev(COMPAS_INPUT_FILE = "../data/compas.csv", svm = False,random_state=42, intercept = False):

    FEATURES_CLASSIFICATION = ["age_cat", "race", "sex", "priors_count", "c_charge_degree"] #features to be used for classification
    CONT_VARIABLES = ["priors_count"] # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
    CLASS_FEATURE = "two_year_recid" # the decision variable
#     SENSITIVE_ATTRS = ["race", "sex"]
    SENSITIVE_ATTRS = ["sex"]


    # load the data and get some stats
    df = pd.read_csv(COMPAS_INPUT_FILE)
    df = df.dropna(subset=["days_b_screening_arrest"]) # dropping missing vals

    # convert to np array
    data = df.to_dict('list')
    for k in data.keys():
        data[k] = np.array(data[k])


    """ Filtering the data """

    # These filters are the same as propublica (refer to https://github.com/propublica/compas-analysis)
    # If the charge date of a defendants Compas scored crime was not within 30 days from when the person was arrested, we assume that because of data quality reasons, that we do not have the right offense. 
    idx = np.logical_and(data["days_b_screening_arrest"]<=30, data["days_b_screening_arrest"]>=-30)


    # We coded the recidivist flag -- is_recid -- to be -1 if we could not find a compas case at all.
    idx = np.logical_and(idx, data["is_recid"] != -1)

    # In a similar vein, ordinary traffic offenses -- those with a c_charge_degree of 'O' -- will not result in Jail time are removed (only two of them).
    idx = np.logical_and(idx, data["c_charge_degree"] != "O") # F: felony, M: misconduct

    # We filtered the underlying data from Broward county to include only those rows representing people who had either recidivated in two years, or had at least two years outside of a correctional facility.
    idx = np.logical_and(idx, data["score_text"] != "NA")

    # we will only consider blacks and whites for this analysis
    idx = np.logical_and(idx, np.logical_or(data["race"] == "African-American", data["race"] == "Caucasian"))

    # select the examples that satisfy this criteria
    for k in data.keys():
        data[k] = data[k][idx]



    """ Feature normalization and one hot encoding """

    # convert class label 0 to -1
    y = data[CLASS_FEATURE]
    if svm:
        y[y==0] = -1



    print ("\nNumber of people recidivating within two years")
    print (pd.Series(y).value_counts())
    print ("\n")


    X = np.array([]).reshape(len(y), 0) # empty array with num rows same as num examples, will hstack the features to it
    x_control = defaultdict(list)

    feature_names = []
    for attr in FEATURES_CLASSIFICATION:
        vals = data[attr]
        if attr in CONT_VARIABLES:
            vals = [float(v) for v in vals]
            vals = preprocessing.scale(vals) # 0 mean and 1 variance  
            vals = np.reshape(vals, (len(y), -1)) # convert from 1-d arr to a 2-d arr with one col

        else: # for binary categorical variables, the label binarizer uses just one var instead of two
            lb = preprocessing.LabelBinarizer()
            lb.fit(vals)
            vals = lb.transform(vals)

        # add to sensitive features dict
        if attr in SENSITIVE_ATTRS:
            x_control[attr] = vals


        # add to learnable features
        X = np.hstack((X, vals))

        if attr in CONT_VARIABLES: # continuous feature, just append the name
            feature_names.append(attr)
        else: # categorical features
            if vals.shape[1] == 1: # binary features that passed through lib binarizer
                feature_names.append(attr)
            else:
                for k in lb.classes_: # non-binary categorical features, need to add the names for each cat
                    feature_names.append(attr + "_" + str(k))


    # convert the sensitive feature to 1-d array
    x_control = dict(x_control)
    for k in x_control.keys():
        assert(x_control[k].shape[1] == 1) # make sure that the sensitive feature is binary after one hot encoding
        x_control[k] = np.array(x_control[k]).flatten()

    # sys.exit(1)

    """permute the date randomly"""
    perm = list(range(0,X.shape[0]))
    shuffle(perm)
    X = X[perm]
    y = y[perm]
    for k in x_control.keys():
        x_control[k] = x_control[k][perm]
    
    n = X.shape[0]
    Z = np.expand_dims(x_control[k],axis=-1)
    tr_idx, te_idx = _get_train_test_split(n, train_frac, random_state)
    Xtr, Xte, ytr, yte, Ztr, Zte = _apply_train_test_split(X, y, Z,
                                                           tr_idx, te_idx)
    
    # Whiten feature data
#     Xtr, Xte = _whiten_data(Xtr, Xte)
    mm = MinMaxScaler()
    Xtr = mm.fit_transform(Xtr)
    Xte = mm.transform(Xte)

#     # Center sensitive data
#     Ztr, Zte = _center_data(Ztr, Zte)

    # Add intercept
    if intercept:
        Xtr, Xte = _add_intercept(Xtr, Xte)
    
        

    return Xtr, Xte, ytr, yte, Ztr, Zte

# def load_bank_data_prev(load_data_size=None):

#     """
#         if load_data_size is set to None (or if no argument is provided), then we load and return the whole data
#         if it is a number, say 10000, then we will return randomly selected 10K examples
#     """

#     attrs = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
#        'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
#        'previous', 'poutcome'] # all attributes
#     int_attrs = ['age', 'balance','day','duration','campaign','pdays','previous'] # attributes with integer values -- the rest are categorical
#     sensitive_attrs = ['marital'] # the fairness constraints will be used for this feature
#     attrs_to_ignore = ['marital'] # sex and race are sensitive feature so we will not use them in classification, we will not consider fnlwght for classification since its computed externally and it highly predictive for the class (for details, see documentation of the adult data)
#     attrs_for_classification = set(attrs) - set(attrs_to_ignore)

  


#     X = []
#     y = []
#     x_control = {}

#     attrs_to_vals = {} # will store the values for each attribute for all users
#     for k in attrs:
#         if k in sensitive_attrs:
#             x_control[k] = []
#         elif k in attrs_to_ignore:
#             pass
#         else:
#             attrs_to_vals[k] = []
        


#     for i in range(len(bank)):
#         line = bank.iloc[i].values
#         class_label = line[-1]
#         if class_label in ["no"]:
#             class_label = -1
#         elif class_label in ["yes"]:
#             class_label = +1
#         else:
#             raise Exception("Invalid class label value")

#         y.append(class_label)


#         for i in range(0,len(line)-1):
#             attr_name = attrs[i]
#             attr_val = line[i]
#                 # reducing dimensionality of some very sparse features

#             if attr_name in sensitive_attrs:
#                 x_control[attr_name].append(attr_val)
#             elif attr_name in attrs_to_ignore:
#                 pass
#             else:
#                 attrs_to_vals[attr_name].append(attr_val)

#     def convert_attrs_to_ints(d): # discretize the string attributes
#         for attr_name, attr_vals in d.items():
#             if attr_name in int_attrs: continue
#             uniq_vals = sorted(list(set(attr_vals))) # get unique values

#             # compute integer codes for the unique values
#             val_dict = {}
#             for i in range(0,len(uniq_vals)):
#                 val_dict[uniq_vals[i]] = i

#             # replace the values with their integer encoding
#             for i in range(0,len(attr_vals)):
#                 attr_vals[i] = val_dict[attr_vals[i]]
#             d[attr_name] = attr_vals

    
#     # convert the discrete values to their integer representations
#     convert_attrs_to_ints(x_control)
#     convert_attrs_to_ints(attrs_to_vals)


#     # if the integer vals are not binary, we need to get one-hot encoding for them
#     for attr_name in attrs_for_classification:
        
#         attr_vals = attrs_to_vals[attr_name]
#         if attr_name in int_attrs or attr_name == "native_country": # the way we encoded native country, its binary now so no need to apply one hot encoding on it

#             X.append(attr_vals)

#         else:            
#             attr_vals, index_dict = get_one_hot_encoding(attr_vals)

#             if attr_vals.shape==(45211,):
#                 attr_vals=attr_vals.reshape(45211,1)
#             for inner_col in attr_vals.T:                
#                 X.append(inner_col) 

 
#     # convert to numpy arrays for easy handline
#     X = np.array(X, dtype=float).T
#     y = np.array(y, dtype = float)
#     for k, v in x_control.items(): x_control[k] = np.array(v, dtype=float)
        
#     # shuffle the data
#     perm = list(range(0,len(y))) # shuffle the data before creating each fold
#     shuffle(perm)
#     X = X[perm]
#     y = y[perm]
#     for k in x_control.keys():
#         x_control[k] = x_control[k][perm]

#     # see if we need to subsample the data
#     if load_data_size is not None:
#         print ("Loading only %d examples from the data" % load_data_size)
#         X = X[:load_data_size]
#         y = y[:load_data_size]
#         for k in x_control.keys():
#             x_control[k] = x_control[k][:load_data_size]

#     return X, y, x_control

# def load_german_data(load_data_size=None):

#     """
#         if load_data_size is set to None (or if no argument is provided), then we load and return the whole data
#         if it is a number, say 10000, then we will return randomly selected 10K examples
#     """

#     attrs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19] # all attributes
#     int_attrs = [1,4,7,8,10,12,15,17] # attributes with integer values -- the rest are categorical
#     sensitive_attrs = [8] # the fairness constraints will be used for this feature
#     attrs_to_ignore = [8] # sex and race are sensitive feature so we will not use them in classification, we will not consider fnlwght for classification since its computed externally and it highly predictive for the class (for details, see documentation of the adult data)
#     attrs_for_classification = set(attrs) - set(attrs_to_ignore)
  
#     X = []
#     y = []
#     x_control = {}

#     attrs_to_vals = {} # will store the values for each attribute for all users
#     for k in attrs:
#         if k in sensitive_attrs:
#             x_control[k] = []
#         elif k in attrs_to_ignore:
#             pass
#         else:
#             attrs_to_vals[k] = []
        


#     for i in range(len(german)):
#         line = german.iloc[i].values
#         class_label = line[-1]
#         if class_label in [1]:
#             class_label = -1
#         elif class_label in [2]:
#             class_label = +1
#         else:
#             raise Exception("Invalid class label value")

#         y.append(class_label)


#         for i in range(0,len(line)-1):
#             attr_name = attrs[i]
#             attr_val = line[i]
#                 # reducing dimensionality of some very sparse features

#             if attr_name in sensitive_attrs:
#                 x_control[attr_name].append(attr_val)
#             elif attr_name in attrs_to_ignore:
#                 pass
#             else:
#                 attrs_to_vals[attr_name].append(attr_val)

#     def convert_attrs_to_ints(d): # discretize the string attributes
#         for attr_name, attr_vals in d.items():
#             if attr_name in int_attrs: continue
#             uniq_vals = sorted(list(set(attr_vals))) # get unique values

#             # compute integer codes for the unique values
#             val_dict = {}
#             for i in range(0,len(uniq_vals)):
#                 val_dict[uniq_vals[i]] = i

#             # replace the values with their integer encoding
#             for i in range(0,len(attr_vals)):
#                 attr_vals[i] = val_dict[attr_vals[i]]
#             d[attr_name] = attr_vals

    
#     # convert the discrete values to their integer representations
#     convert_attrs_to_ints(x_control)
#     convert_attrs_to_ints(attrs_to_vals)


#     # if the integer vals are not binary, we need to get one-hot encoding for them
#     for attr_name in attrs_for_classification:
        
#         attr_vals = attrs_to_vals[attr_name]
#         if attr_name in int_attrs or attr_name == "native_country": # the way we encoded native country, its binary now so no need to apply one hot encoding on it

#             X.append(attr_vals)

#         else:            
#             attr_vals, index_dict = get_one_hot_encoding(attr_vals)

#             if attr_vals.shape==(1000,):
#                 attr_vals=attr_vals.reshape(1000,1)
#             for inner_col in attr_vals.T:                
#                 X.append(inner_col) 

 
#     # convert to numpy arrays for easy handline
#     X = np.array(X, dtype=float).T
#     y = np.array(y, dtype = float)
#     for k, v in x_control.items(): x_control[k] = np.array(v, dtype=float)
        
#     # shuffle the data
#     perm = list(range(0,len(y))) # shuffle the data before creating each fold
#     shuffle(perm)
#     X = X[perm]
#     y = y[perm]
#     for k in x_control.keys():
#         x_control[k] = x_control[k][perm]

#     # see if we need to subsample the data
#     if load_data_size is not None:
#         print ("Loading only %d examples from the data" % load_data_size)
#         X = X[:load_data_size]
#         y = y[:load_data_size]
#         for k in x_control.keys():
#             x_control[k] = x_control[k][:load_data_size]

#     return X, y, x_control



def get_one_hot_encoding(in_arr):
    """
        input: 1-D arr with int vals -- if not int vals, will raise an error
        output: m (ndarray): one-hot encoded matrix
                d (dict): also returns a dictionary original_val -> column in encoded matrix
    """

    for k in in_arr:
        if str(type(k)) != "<type 'numpy.float64'>" and type(k) != int and type(k) != np.int64:
            print (str(type(k)))
            print ("************* ERROR: Input arr does not have integer types")
            return None
        
    in_arr = np.array(in_arr, dtype=int)
    assert(len(in_arr.shape)==1) # no column, means it was a 1-D arr
    attr_vals_uniq_sorted = sorted(list(set(in_arr)))
    num_uniq_vals = len(attr_vals_uniq_sorted)
    if (num_uniq_vals == 2) and (attr_vals_uniq_sorted[0] == 0 and attr_vals_uniq_sorted[1] == 1):
        return in_arr, None

    
    index_dict = {} # value to the column number
    for i in range(0,len(attr_vals_uniq_sorted)):
        val = attr_vals_uniq_sorted[i]
        index_dict[val] = i

    out_arr = []    
    for i in range(0,len(in_arr)):
        tup = np.zeros(num_uniq_vals)
        val = in_arr[i]
        ind = index_dict[val]
        tup[ind] = 1 # set that value of tuple to 1
        out_arr.append(tup)

    return (np.array(out_arr), index_dict)

def _transform_dum2cat(xs,nil = None):
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

def _add_intercept(train, test):
    """Add a column of all ones for the intercept."""
    return tuple([np.hstack((np.ones((x.shape[0], 1)), x))
                  for x in (train, test)])


def _labels_to_zero_one(*args):
    """Map labels to take values in zero and one."""
    for x in args:
        x[x <= 0.] = 0
        x[x > 0.] = 1


def _labels_to_plus_minus(*args):
    """Map labels to take values in minus one and one."""
    for x in args:
        x[x <= 0.] = -1
        x[x > 0.] = 1


def _shuffle_data(*args, random_state=42):
    """Shuffle data with random permutation."""
    n = args[0].shape[0]
    np.random.seed(random_state)
    perm = np.random.permutation(n)
    return tuple([x[perm, :] if x.ndim > 1 else x[perm] for x in args])


def _center_data(train, test= None):
    """Center the data, i.e. subtract the mean column wise."""
    mean = np.mean(train, 0)
    if test is None:
        return train - mean
    else:
        return train - mean, test - mean


def _whiten_data(train, test):
    """Whiten training and test data with training mean and std dev."""
    mean = np.mean(train, 0)
    std = np.std(train, 0)
    return tuple([(x - mean) / (std + 1e-7) for x in (train, test)])


def _get_train_test_split(n_examples, train_fraction, seed, power_of_two=False):
    """
    Args:
        n_examples: Number of training examples
        train_fraction: Fraction of data to use for training
        seed: Seed for random number generation (reproducability)
        power_of_two: Whether to select the greatest power of two for training
            set size and use the remaining examples for testing.

    Returns:
        training indices, test indices
    """
    np.random.seed(seed)
    idx = np.random.permutation(n_examples)
    pivot = int(n_examples * train_fraction)
    if power_of_two:
        pivot = 2**(len(bin(pivot)) - 3)
    training_idx = idx[:pivot]
    test_idx = idx[pivot:]
    return training_idx, test_idx


def _apply_train_test_split(x, y, z, training_idx, test_idx):
    """
    Apply the train test split to the data.

    Args:
        x: Features
        y: Labels
        z: Sensitive attributes
        training_idx: Training set indices
        test_idx: Test set indices

    Returns:
        Xtr, Xte, ytr, yte, Ztr, Zte: X are the features, y are the labels and Z
            are the sensitive attributes. 'tr' stands for training and 'te' for
            test set.
    """
    Xtr = x[training_idx, :]
    Xte = x[test_idx, :]
    ytr = y[training_idx]
    yte = y[test_idx]
    Ztr = z[training_idx, :]
    Zte = z[test_idx, :]
    return Xtr, Xte, ytr, yte, Ztr, Zte


def _onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        print('not in allowable!')
        pdb.set_trace()
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def load_npz(filepath):
    """Load a preprocessed dataset that has been saved in numpy's npz format."""
    d = np.load(filepath)
    return d['Xtr'], d['Xte'], d['ytr'], d['yte'], d['Ztr'], d['Zte']
