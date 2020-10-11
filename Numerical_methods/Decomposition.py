import numpy as np

def cholesky_decomposition(A):
    n = len(A)
    for k in range(n):
        A[k, k] = np.sqrt(A[k, k])
        print(f"A[{k+1}, {k+1}:] {A[k, k]}")
        if k < n:
            for l in range(k+1, n):
                A[l, k] = A[l, k] / A[k, k]
                print(f"A[{l+1}, {k+1}]: {A[l, k]}")
                for j in range(k+1, n):
                    A[l, j] = A[l, j] - A[l, k] * A[j, k]
                    print(f"A[{l+1}, {j+1}]: {A[l, j]}")

    return np.tril(A)
