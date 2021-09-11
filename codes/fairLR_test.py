import numpy as np
from save_utils import load_flr, load_nparray,save_prediction

def predict_FairLR(X_te, coef):
    res = np.dot(X_te, coef)
    return np.sign(res)
#     return res

def test_FairLR(save_md, save_te, save_dir = ''):
    print("load FLR model")
    coef = load_flr(save_md)
    
    print("load test data from", save_te)
    X_te = load_nparray(save_te)
    
    print("predict FLR")
    pred = predict_FairLR(X_te, coef)
    
    print("save prediction")
    save_prediction(pred,'adult',save_dir,'flr')

def main(*args):    
    print(args)
    test_FairLR(*args)

if __name__ == '__main__':
    save_md = '../results/FLR_model.sm'
    save_te = '../results/adult_testX.te'
    save_dir = '../results'
    
    main(save_md, save_te,save_dir)
