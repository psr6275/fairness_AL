import numpy as np
import pickle
from save_utils import load_nparray,load_testdata
from fair_eval import *

def evaluate_model(path_pr, path_te):

    pred = load_nparray(path_pr)
    X_te, y_te, xs_te = load_testdata(path_te)
    
    calculate_prule_clf(pred,y_te,xs_te)
    calculate_odds_clf(pred,y_te,xs_te)
    calculate_parity_reg(pred,y_te,xs_te)
    calculate_group_loss(l2_loss,pred,y_te,xs_te)
    

def main(*args):    
	evaluate_model(*args)

if __name__ == '__main__':
    path_pr = '../results/adult_flr_pred.pr'
    path_te = '../results/adult_testset.te'
    
    main(path_pr, path_te)
