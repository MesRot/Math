import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import timeit


def back_substitution(A, b, lower=False):
    '''
    Ratkaistaan yhtälöryhmä kolmio matriisista
    :param A: Matriisi
    :param b: b-vektori
    :param lower: Alustavasti funktio toimii vain alakolmioille, mutta asettamalla parametrin True niin funktio toimii
    yläkolmio matriisille
    :return: Palauttaa ratkaistun x-vektorin
    '''
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
    '''
    Gaussin eliminaatio, jossa lopussa kutsutaan funktiota, joka ratkaisee saadun kolmiomatriisin
    :param A: Matriisi
    :param b: B-vektori
    :return: Palauttaa ratkaistun x-vektorin
    '''
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
    '''
    Hajoittaaa matriisin ala- ja yläkolmiomatriisiin
    :param A: Matriisi
    :return: Palauttaa ala- ja yläkolmiomatriisin
    '''
    n = len(A)
    for k in range(n-1):
        for i in range(k+1, n):
            A[i, k] = A[i, k]/A[k, k]
            for j in range(k+1, n):
                A[i, j] -= A[i, k] * A[k, j]
    L = np.tril(A, k=-1) + np.identity(n)
    U = np.triu(A)

    return L, U


def solve_lu(L, U, b):
    '''
    Funktio, joka ratkaisee x-vektorin ala- ja yläkolmiomatriisista.
    :param L: Yläkolmiomatriisi
    :param U: Alakolmiomatriisi
    :param b: b-vektori
    :return: palauttaa x-vektorin
    '''
    Lstar = np.flip(L)
    bstar = np.flip(b)
    z = back_substitution(Lstar, bstar, lower=True)
    x = back_substitution(U, z)
    return x


def time_function(z, method, n):
    '''
    Apufunktio aikatestaukseen. Ei palauta mitään, mutta testataan kauanko funktion suoritus kestää
    :param z: Kuinka monta eri b-vektoria ratkaistaan
    :param method: Kumpi metodi
    :param n: Matriisin koko
    :return: None
    '''
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
    '''
    Apufunktio data piirtämiseen. Kuvaajan tallentava rivi kommentoitu pois
    :param df: Data panda-dataframe formaatissa.
    :return:
    '''
    for i in range(1, 6):
        data = df[(df.b == i)]
        gauss_df = data[(data.Method == "Gauss")]
        lu_df = data[(data.Method == "LU")]
        plt.plot(lu_df.N, lu_df.Time, 'b-')
        plt.plot(gauss_df.N, gauss_df.Time, 'r-')
        plt.suptitle(f"Time to solve {i} equations")
        plt.xlabel("Matrix size NxN:")
        plt.ylabel("Time taken:")
        #plt.savefig(fname=f"time_to_solve_{i}")
        plt.show()


def main():
    '''
    Pääfunktio, jossa määritettään parametrit ja luodaan dataframe ja kutsutaan plot funktiota
    :return:
    '''
    data = []
    testable = ["Gauss", "LU"]
    for z in tqdm(range(1, 6)):
        for n in range(3, 150, 30):
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