import numpy as np 

def remove_rejected(y_pred, y_true, reject_portion, uncertainties):
    """Remove the most uncertain predictions based on rejection portion. 
    From: https://github.com/YSale/label-uq/blob/main/arc_ood/uncertainty.py"""
    if reject_portion == 0:
        return y_pred, y_true
    num = int(y_pred.shape[0] * reject_portion) #No. of samples to reject
    indices = np.argsort(uncertainties)
    y_pred = y_pred[indices]
    y_true = y_true[indices]
    y_pred = y_pred[:-num] #Remove most uncertain samples
    y_true = y_true[:-num]
    return y_pred, y_true