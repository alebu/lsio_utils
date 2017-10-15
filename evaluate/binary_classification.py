from sklearn.metrics import roc_curve, auc


def gini(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return 2*auc(fpr, tpr) - 1