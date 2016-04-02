import numpy as np
from scipy.stats import norm
def bns(word, data, category):
    total = len(data)
    tp = np.sum([word in data[i] for i in range(0, len(data)) if category[i] == 1])
    fp = np.sum([word in data[i] for i in range(0, len(data)) if category[i] == 0])
    pos = np.sum(category)
    neg = len(data) - pos
    tpr = 1.0 * tp / pos
    fpr = 1.0 * fp / neg
    print tpr, fpr
    return np.abs(norm.ppf(tpr) - norm.ppf