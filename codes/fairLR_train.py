import numpy as np
from random import seed
import loss_fn as lf # our implementation of loss funcs
from scipy.optimize import minimize # for loss func minimization
# from multiprocessing import Pool, Process, Queue
# from collections import defaultdict
from copy import deepcopy
import matplotlib.pyplot as plt # for plotting stuff
import sys

from load_data import *
from save_utils import save_flr, save_testdata


SEED = 1122334455
seed(SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)

def get_one_hot_encoding(in_arr):
    """
        input: 1-D arr with int vals -- if not int vals, will raise an error
        output: m (ndarray): one-hot encoded matrix
                d (dict): also returns a dictionary original_val -> column in encoded matrix
    """

    for k in in_arr:
        if str(type(k)) != "<type 'numpy.float64'>" and type(k) != int and type(k) != np.int64:
            print(str(type(k)))
            print("************* ERROR: Input arr does not have integer types")
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

    return np.array(out_arr), index_dict

def test_sensitive_attr_constraint_cov(model, x_arr, y_arr_dist_boundary, x_control, thresh, verbose):

    
    """
    The covariance is computed b/w the sensitive attr val and the distance from the boundary
    If the model is None, we assume that the y_arr_dist_boundary contains the distace from the decision boundary
    If the model is not None, we just compute a dot product or model and x_arr
    for the case of SVM, we pass the distace from bounday becase the intercept in internalized for the class
    and we have compute the distance using the project function

    this function will return -1 if the constraint specified by thresh parameter is not satifsified
    otherwise it will reutrn +1
    if the return value is >=0, then the constraint is satisfied
    """

    


    assert(x_arr.shape[0] == x_control.shape[0])
    if len(x_control.shape) > 1: # make sure we just have one column in the array
        assert(x_control.shape[1] == 1)
    
    arr = []
    if model is None:
        arr = y_arr_dist_boundary # simply the output labels
    else:
        arr = np.dot(model, x_arr.T) # the product with the weight vector -- the sign of this is the output label
    
    arr = np.array(arr, dtype=np.float64)


    cov = np.dot(x_control - np.mean(x_control), arr ) / float(len(x_control))

        
    ans = thresh - abs(cov) # will be <0 if the covariance is greater than thresh -- that is, the condition is not satisfied
    # ans = thresh - cov # will be <0 if the covariance is greater than thresh -- that is, the condition is not satisfied
    if verbose is True:
        print("Covariance is", cov)
        print("Diff is:", ans)
        print()
    return ans

def get_constraint(X, y, xs, sen_cov_thresh):
    constraints = []
    
    xs = xs.astype(np.int64).flatten()
    attr_arr, index_dict = get_one_hot_encoding(xs)
    print(index_dict)
    if index_dict is None: # binary attribute
#         thresh = sensitive_attrs_to_cov_thresh[attr]
        c = ({'type': 'ineq', 'fun': test_sensitive_attr_constraint_cov, 'args':(X, y, attr_arr,sen_cov_thresh, False)})
        constraints.append(c)
#         print(constraints)
    else: # otherwise, its a categorical attribute, so we need to set the cov thresh for each value separately
        for attr_val, ind in index_dict.items():
            attr_name = attr_val                
            t = attr_arr[:,ind]
            c = ({'type': 'ineq', 'fun': test_sensitive_attr_constraint_cov, 'args':(X, y, t ,sen_cov_thresh, False)})
            constraints.append(c)
    return constraints

def train_FairLR(X, y, xs, loss_function, fair_const = 1, acc_const = 0, sep_const = 0, sen_cov_thresh=0, gamma=False):

    """

    Function that trains the model subject to various fairness constraints.
    If no constraints are given, then simply trains an unaltered classifier.
    Example usage in: "synthetic_data_demo/decision_boundary_demo.py"

    ----

    Inputs:

    X: (n) x (d+1) numpy array -- n = number of examples, d = number of features, one feature is the intercept
    y: 1-d numpy array (n entries)
    x_control: dictionary of the type {"s": [...]}, key "s" is the sensitive feature name, and the value is a 1-d list with n elements holding the sensitive feature values
    loss_function: the loss function that we want to optimize -- for now we have implementation of logistic loss, but other functions like hinge loss can also be added
    apply_fairness_constraints: optimize accuracy subject to fairness constraint (0/1 values)
    apply_accuracy_constraint: optimize fairness subject to accuracy constraint (0/1 values)
    sep_constraint: apply the fine grained accuracy constraint
        for details, see Section 3.3 of arxiv.org/abs/1507.05259v3
        For examples on how to apply these constraints, see "synthetic_data_demo/decision_boundary_demo.py"
    Note: both apply_fairness_constraints and apply_accuracy_constraint cannot be 1 at the same time
    sensitive_attrs: ["s1", "s2", ...], list of sensitive features for which to apply fairness constraint, all of these sensitive features should have a corresponding array in x_control
    sensitive_attrs_to_cov_thresh: the covariance threshold that the classifier should achieve (this is only needed when apply_fairness_constraints=1, not needed for the other two constraints)
    gamma: controls the loss in accuracy we are willing to incur when using apply_accuracy_constraint and sep_constraint

    ----

    Outputs:

    w: the learned weight vector for the classifier

    """


    assert((acc_const == 1 and fair_const == 1) == False) # both constraints cannot be applied at the same time

    max_iter = 100000 # maximum number of iterations for the minimization algorithm

    if fair_const == 0:
        constraints = []
    else:
        constraints = get_constraint(X, y, xs, sen_cov_thresh)      

    if acc_const == 0: #its not the reverse problem, just train w with cross cov constraints

        f_args=(X, y)
        w = minimize(fun = loss_function,
            x0 = np.random.rand(X.shape[1],),
            args = f_args,
            method = 'SLSQP',
            options = {"maxiter":max_iter},
            constraints = constraints
            )

    else:

        # train on just the loss function
        w = minimize(fun = loss_function,
            x0 = np.random.rand(X.shape[1],),
            args = (X, y),
            method = 'SLSQP',
            options = {"maxiter":max_iter},
            constraints = []
            )

        old_w = deepcopy(w.x)
        

        def constraint_gamma_all(w, x, y,  initial_loss_arr):
            
            gamma_arr = np.ones_like(y) * gamma # set gamma for everyone
            new_loss = loss_function(w, x, y)
            old_loss = sum(initial_loss_arr)
            return ((1.0 + gamma) * old_loss) - new_loss

        def constraint_protected_people(w,x,y): # dont confuse the protected here with the sensitive feature protected/non-protected values -- protected here means that these points should not be misclassified to negative class
            return np.dot(w, x.T) # if this is positive, the constraint is satisfied
        def constraint_unprotected_people(w,ind,old_loss,x,y):
            
            new_loss = loss_function(w, np.array([x]), np.array(y))
            return ((1.0 + gamma) * old_loss) - new_loss

        constraints = []
        predicted_labels = np.sign(np.dot(w.x, X.T))
        unconstrained_loss_arr = loss_function(w.x, X, y, return_arr=True)

        if sep_const == True: # separate gemma for different people
            for i in range(0, len(predicted_labels)):
                if predicted_labels[i] == 1.0 and xs == 1.0: # for now we are assuming just one sensitive attr for reverse constraint, later, extend the code to take into account multiple sensitive attrs
                    c = ({'type': 'ineq', 'fun': constraint_protected_people, 'args':(X[i], y[i])}) # this constraint makes sure that these people stay in the positive class even in the modified classifier             
                    constraints.append(c)
                else:
                    c = ({'type': 'ineq', 'fun': constraint_unprotected_people, 'args':(i, unconstrained_loss_arr[i], X[i], y[i])})                
                    constraints.append(c)
#             print(c)
        else: # same gamma for everyone
            c = ({'type': 'ineq', 'fun': constraint_gamma_all, 'args':(X,y,unconstrained_loss_arr)})
            constraints.append(c)
#             print(c)

        def cross_cov_abs_optm_func(weight_vec, x_in, x_control_in_arr):
            cross_cov = (x_control_in_arr - np.mean(x_control_in_arr)) * np.dot(weight_vec, x_in.T)
            return float(abs(sum(cross_cov))) / float(x_in.shape[0])


        w = minimize(fun = cross_cov_abs_optm_func,
            x0 = old_w,
            args = (X, xs),
            method = 'SLSQP',
            options = {"maxiter":100000},
            constraints = constraints
            )

    try:
        assert(w.success == True)
    except:
        print("Optimization problem did not converge.. Check the solution returned by the optimizer.")
        print("Returned solution is:")
        print(w)



    return w.x

def train_flr(data = 'adult',save_dir = '', filename = 'FLR_model'):
    print("load data:", data)
    if data == 'adult':
        X_tr, X_te, y_tr, y_te, xs_tr, xs_te = load_adult_data(svm=True,random_state=42,intercept=True)
    elif data == 'bank':
        X_tr, X_te, y_tr, y_te, xs_tr, xs_te = load_bank_data(svm=True,random_state=42,intercept=True)
    elif data == 'compas':
        X_tr, X_te, y_tr, y_te, xs_tr, xs_te = load_compas_data(svm=True,random_state=42,intercept=True)
    elif data == 'german':
        X_tr, X_te, y_tr, y_te, xs_tr, xs_te = load_german_data(svm=True,random_state=42,intercept=True)
        
    fair_const = 1 # for fairness constraints setting
    acc_const = 0
    sep_const = 0

    loss_fn = lf._logistic_loss
    sen_cov_thresh = 0 # threshold for fairness level (0: perfect fairness)
    gamma = None
    
    print("train FLR model")
    coef = train_FairLR(X_tr, y_tr, xs_tr, loss_fn, fair_const, acc_const ,sep_const , sen_cov_thresh, gamma)
    
    print("save model")
    save_flr(coef, save_dir, filename)
    
    print("save testdata")
    save_testdata(X_te,y_te,xs_te,data, save_dir)
    

def main(**args):
	train_flr(**args)


if __name__ == '__main__':
#     data = 
	save_dir = '../results'
#     filename = 
	main(save_dir =save_dir)
