#from tool import *
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

############### -------------------------- VIRTUAL SITE ------------------------##################################
def fast_loocv_MO(data_set, new_values):
    """
        This function takes as input a data_set and a set of new values
        and evaluates the Leave One Out Cross Validation Error as
        the equation reported on the article
    """
    y    = data_set.y_sample
    v    = data_set.w_vector
    
    sigma, H = sigma_h(data_set)
    
    l  = leverage(data_set,flag="vs")
    
    err_loocv = []
    mae       = []
    for c in new_values:
        _err_loocv, _mae = fast_loocv(c, H, y, l, v)
        err_loocv.append(np.asscalar(_err_loocv))
        mae.append(_mae)
    return err_loocv, mae


#@jit(fastmath = True) comportamento strano su np.dot....
def fast_loocv(c, H, y, l, v):
    """
        pancakes
    """
    y_sample_est = np.dot(H,c)
    dev          = np.subtract(y_sample_est, y)
    den          = 1 - l
    loocv_ei     = np.power(np.divide(dev.T, den), 2)
    aux          = np.dot(loocv_ei.T, v)    
    mae          = np.mean(np.abs(dev)) 
    return aux, mae
