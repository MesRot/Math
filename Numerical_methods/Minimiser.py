import numpy as np

def golden_ratio_search(f, a, b, iter=0, eps=0.001, maxiter=20):
    '''

    :param f: Funktio
    :param a: Välin toinen päätepiste
    :param b: Välin toinen päätepiste
    :param iter: Tämänhetkinen iteraatio, ei anneta alussa arvoa
    :param eps: Haluttu välin pituus
    :param maxiter: Suurin sallittu iteraatioiden määrä. Emme halua että funktio juoksee ikuisesti jos väli ei suppene
    :return: Halutun pituisen välin keskipisteen
    '''
    alpha = ((1 + 5 ** 0.5) / 2) - 1  # Lasketaan algoritmissä tarvittava alfa
    if iter > maxiter:  # Tarkistetaan kuinka monta kertaa funktio on kutsuttu
        print("Iteraatiot käytetty loppuun")
        return None  # Vältetään loputtomia silmukoita pitämällä maksimi iteraatiot
    print(f"Iter: {iter}, value of a: {a}, value of b:{b}")  #Tulostetaan tämän iteraation tulokset
    if abs(a - b) < eps:  # Tarkistetaanko on välin pituus tarpeeksi pieni
        print(f"Väli tarpeeksi pieni. Välin keskikohta: {(a+b)/2}")  # Tulostetaan löydetty välin keskikohta
        return (a + b) / 2
    lam = a + (1 - alpha) * (b - a)  # Lasketaan lambdan arvo
    myy = a + alpha * (b - a)  # Lasketaan myyn arvo
    print(f"lam[{iter}]: {lam}, myy[{iter}]: {myy}")
    print(f"f(lam[{iter}])= {f(lam)}, f(myy[{iter}]={f(myy)}")
    if f(lam) > f(myy):  #Suoraan algoritmista. Kutsutaan funktiota uusilla arvoilla ja lisätään iteraatioon lisää kierroksia
        golden_ratio_search(f, lam, b, iter+1, eps=eps, maxiter=maxiter)
    else:
        golden_ratio_search(f, a, myy, iter+1, eps=eps, maxiter=maxiter)


def main():
    func = lambda x : np.exp(x ** 2) - x
    golden_ratio_search(f=func, a=0, b=1, eps=0.1, maxiter=15)  # Kutsutaan algoritmia aloitusarvoilla ja halutulla välin pituudella.


if __name__ == "__main__":
    main()


