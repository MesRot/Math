import numpy as np

def gauss_elimination(A, b):
    n = len(A)
    for k in range(n-1):
        for i in range(k+1, n):
            r = A[i, k] / A[k, k]
            print(r)
            for l in range(k, n):
                print("l: ", l)
                A[i, l] = A[i, l] - r * A[k, l]
                print(A)
            b[i] = b[i] - r * b[k]
    return A, b


def gauss_seidel(A, b, x0=None, iter=20):
    n = len(A)
    D = np.diag(np.diag(A))
    if x0 is None:
        x = np.ones(n).T
    else:
        x = x0
    L_star = np.tril(A)
    print(L_star)
    U = np.triu(A) - D
    L_star_inv = np.linalg.inv(L_star)
    H = np.dot(-L_star_inv, U)
    C = np.dot(L_star_inv, b.T)
    for i in range(iter):
        x = np.matmul(H, x) + C
        #print(x)
    return x


def jacobi_method(A, b, x0=None, iter=20):
    n = len(A)
    D = np.diag(np.diag(A))
    J = np.sum(np.abs(D), axis=1)
    S = np.sum(np.abs(A), axis=1) - J
    #if np.all(J < S):
    #    print("Matrix is not diagonally dominant")
    #    return None
    if x0 is None:
        x = np.ones(n).T
    else:
        x = x0
    U = np.triu(A) - D
    L = np.tril(A) - D
    Dinv = np.linalg.inv(D)
    H = np.dot(-Dinv, (L + U))
    print(f"X0: {x0}")
    print(f"U: {U}\nL: {L}\nD-1: {Dinv}")
    print(f"H : {H}")
    C = np.dot(Dinv, b.T)
    print(f"C: {C}")
    for i in range(iter):
        x = np.matmul(H, x) + C
        print(f"Iteraatio {i+1} , vektori: {x}")
    return x


def newton_raphson(f, fprime, x=1, eps=0.001):
    while True:
        print(f"Xn arvo: {x}")
        xprev = x  # Tallennetaan arvo ylös että sitä voidaan verrata kohta
        x = x - f(x) / fprime(x)  # Newton Raphson algoritmi
        if abs(xprev - x) < eps:  # Pyörii kunnes Xn ja Xn+1 ovat halutun etäisyyden pääässä toisistaan
            print(f"X:n arvo kun kaksi edellistä arvoa ovat halutulla etäisyydellä toisitaan: {x}")
            break
    return x


def main():
    fder = lambda x : 2 * x * np.exp(x ** 2) - 1  # F:n derivaatta
    fderder = lambda x : (4 * x ** 2 + 2) * np.exp(x ** 2)  # F:n toinen derivaatta
    newton_raphson(fder, fderder, x=0, eps=0.1)




if __name__ == "__main__":
    main()