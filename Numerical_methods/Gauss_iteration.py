import numpy as np
import time


def back_substitution(A, b, lower=False):
    n = len(A)
    x = np.zeros_like(b)
    if A[n - 1, n - 1] == 0:
        raise ValueError
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            b[i] = b[i] - A[i, j] * x[j]
        x[i] = b[i] / A[i, i]
    if lower:
        x = np.flip(x)
    return x


def gauss_elimination(A, b):
    start_time = time.time()
    n = len(A)
    x = np.zeros(n)
    for k in range(n):
        for i in range(k+1, n):
            r = A[i, k] / A[k, k]
            for l in range(k, n):
                A[i, l] -= r * A[k, l]
            b[i] = b[i] - r * b[k]
    x = back_substitution(A, b)
    time_taken = time.time() - start_time
    return x, time_taken


def lu_decomposition_and_solve(A, b):
    n = len(A)
    L = np.identity(n)
    U = A.copy()
    for k in range(n-1):
        for j in range(k+1, n):
            L[j, k] = U[j, k] / U[k, k]
            U[j, k:] -= L[j, k] * U[k, k:]
    Lstar = np.flip(L)
    bstar = np.flip(b)
    z = back_substitution(Lstar, bstar, lower=True)
    x = back_substitution(U, z)
    return x


def main():
    n = 3
    #A = np.random.uniform(low=0.5, high=20, size=(n,n))
    #b = np.random.uniform(low=0.5, high=20, size=n)
    A = np.asmatrix([[3, 4, 5], [6, 9, 2], [2, 5, 6]], dtype=np.float64)
    b = np.asarray([7, 4, 5], dtype=np.float64)
    x = lu_decomposition_and_solve(A, b.copy())
    print(x)
    print("B : ", b)
    x, time = gauss_elimination(A, b)
    print(x)
    #print(f"time taken for {n}x{n} matrix: {time} seconds")


if __name__ == "__main__":
    main()