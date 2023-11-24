import numpy as np


def load_logs_csv(filepath):
    R"""
        read CSV file data into structured ndarray
    """
    ans = np.genfromtxt(filepath, names=True, delimiter=',')
    return ans
