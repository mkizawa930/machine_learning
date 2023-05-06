import pandas as pd
import numpy as np

def split_sequence(x, y, seqlen):
    """
    length x - 1
    
    Arguments:
        x: ndarray (time, features)

    """
    xs, ys = [], []
    for i in range(x.shape[0]):
        x_, y_ = x[i:i+seqlen, :], y[i+seqlen-1]

        xs.append(x_)
        ys.append(y_)

        if i+seqlen == x.shape[0]:
            break

    return np.array(xs), np.array(ys)
