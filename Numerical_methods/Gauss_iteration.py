import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


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
    n = len(A)
    x = np.zeros(n)
    for k in range(n):
        for i in range(k+1, n):
            r = A[i, k] / A[k, k]
            for l in range(k, n):
                A[i, l] -= r * A[k, l]
            b[i] = b[i] - r * b[k]
    x = back_substitution(A, b)
    return x


def lu_decomposition(A):
    n = len(A)
    L = np.identity(n)
    U = A.copy()
    for k in range(n-1):
        for j in range(k+1, n):
            L[j, k] = U[j, k] / U[k, k]
            U[j, k:] -= L[j, k] * U[k, k:]
            for i in range(n):
                n = len(A)
    return L, U


def solve_lu(L, U, b):
    Lstar = np.flip(L)
    bstar = np.flip(b)
    z = back_substitution(Lstar, bstar, lower=True)
    x = back_substitution(U, z)
    return x


def main():
    data = []
    #df = pd.DataFrame(columns=columns, index=)
    for z in tqdm(range(1, 6)):
        for n in range(3, 200, 10):
            A = np.random.uniform(low=0.5, high=20, size=(n, n))
            start_time = time.time()
            L, U = lu_decomposition(A)
            for i in range(1, z):
                b = np.random.uniform(low=0.5, high=20, size=n)
                solve_lu(L, U, b)
            total_time = time.time() - start_time
            data.append([z, n, total_time, "LU"])

            start_time = time.time()
            for i in range(1, z):
                b = np.random.uniform(low=0.5, high=20, size=n)
                gauss_elimination(A, b)
            total_time = time.time() - start_time
            data.append([z, n, total_time, "Gauss"])

    columns = ["b", "N", "Time", "Method"]
    df = pd.DataFrame(columns=columns, data=data)
    for i in range(1, 6):
        data = df[(df.b == i)]
        gauss_df = data[(data.Method == "Gauss")]
        lu_df = data[(data.Method == "LU")]
        plt.plot(lu_df.N, lu_df.Time, 'b-')
        plt.plot(gauss_df.N, gauss_df.Time, 'r-')
        plt.show()
    print(df.head)
    #plt.plot(matrix_sizes, gauss_times, 'b-')
    #plt.ylabel("Time: ")
    #plt.xlabel("Matrix size: ")
    #plt.show()


    #print(f"time taken for {n}x{n} matrix: {time} seconds")


if __name__ == "__main__":
    main()