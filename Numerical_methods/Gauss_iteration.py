import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import timeit


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
    for k in range(n-1):
        for i in range(k+1, n):
            A[i,k] = A[i,k]/A[k,k]
            for j in range(k+1, n):
                A[i, j] -= A[i, k] * A[k, j]
    L = np.tril(A, k=-1) + np.identity(n)
    U = np.triu(A)

    return L, U


def solve_lu(A, b):
    L, U = lu_decomposition(A.copy())
    Lstar = np.flip(L)
    bstar = np.flip(b)
    z = back_substitution(Lstar, bstar, lower=True)
    x = back_substitution(U, z)
    return x


def time_function(z, method, n):
    A = np.random.uniform(low=0.5, high=10, size=(n, n))
    if method == "Gauss":
        for i in range(z):
            b = np.random.uniform(low=0.5, high=20, size=n)
            gauss_elimination(A, b)
    else:
        L, U = lu_decomposition(A)
        for i in range(z):
            b = np.random.uniform(low=0.5, high=10, size=n)
            solve_lu(L, U, b)


def plot_data(df):
    for i in range(1, 6):
        data = df[(df.b == i)]
        gauss_df = data[(data.Method == "Gauss")]
        lu_df = data[(data.Method == "LU")]
        plt.plot(lu_df.N, lu_df.Time, 'b-')
        plt.plot(gauss_df.N, gauss_df.Time, 'r-')
        plt.suptitle(f"Time to solve {i} equations")
        plt.xlabel("Matrix size NxN:")
        plt.ylabel("Time taken:")
        plt.savefig(fname=f"time_to_solve_{i}")
        plt.show()


def main():
    data = []
    testable = ["Gauss", "LU"]
    for z in tqdm(range(1, 6)):
        for n in range(3, 300, 30):
            for function in testable:
                starttime = timeit.default_timer()
                time_function(z, function, n)
                total_time = timeit.default_timer() - starttime
                data.append([z, n, total_time, function])

    columns = ["b", "N", "Time", "Method"]
    df = pd.DataFrame(columns=columns, data=data)
    plot_data(df)


if __name__ == "__main__":
    main()