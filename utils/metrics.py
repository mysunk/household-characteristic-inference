
import numpy as np
from sklearn import metrics

def KL(P,Q):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001

    # You may want to instead make copies to avoid changing the np arrays.
    P = P+epsilon
    Q = Q+epsilon

    divergence = np.sum(P*np.log(P/Q))
    return divergence

def evaluate(y_true, y_pred):
    """
    calculate acc, auc, f1 score of y_true and y_pred
    """
    acc_ = (np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)).mean()
    fpr, tpr, thresholds = metrics.roc_curve(np.argmax(y_true, axis=1), y_pred[:,0], pos_label=0)
    auc_ = metrics.auc(fpr, tpr)
    f1_score_ = metrics.f1_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), average='weighted')
    # print('accuracy: {:.3f}, auc: {:.3f}, f1 score: {:.3f}'.format(acc_, auc_, f1_score_))

    return acc_, auc_, f1_score_