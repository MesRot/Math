import numpy as np


def l1_norm(A):
    row_sum = np.sum(abs(A), axis=1)
    return np.max(row_sum)

def lmax_norm(A):
    col_sum = np.sum(abs(A), axis=0)
    return np.max(col_sum)

A = np.asmatrix([[12, 4, 2],
                [8, -10, 5],
                [4, 3, 12]])

print(l1_norm(A))
print(lmax_norm(A))