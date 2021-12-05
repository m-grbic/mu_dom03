import numpy as np
import random
from qpsolvers import *
from .data_utils import *
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
from functools import partial
import seaborn as sn
from tqdm import tqdm


class SVM():

    def __init__(self, C: float = None, kernel: str = 'linear', sigma: float = None):

        self.w = None
        self.b = None
        self.C = C
        if kernel not in ["linear", "gauss"]:
            raise ValueError(f"Vrednost {kernel} nije podrzana!")
        self.kernel = kernel
        self.sigma = sigma
        self.gamma = (- 1 / (2 * sigma ** 2)) if sigma is not None else None
        self.X_train = None
        self.y_train = None

    def reset(self, C: float = None, kernel: str = 'linear', sigma: float = None) -> None:

        self.w = None
        self.b = None
        self.C = C
        if kernel not in ["linear", "gauss"]:
            raise ValueError(f"Vrednost {kernel} nije podrzana!")
        self.kernel = kernel
        self.sigma = sigma
        self.gamma = (- 1 / (2 * sigma ** 2)) if sigma is not None else None
        self.X_train = None
        self.y_train = None

    def rbf(self, X: np.ndarray) -> np.ndarray:
        """ Primena gaussovskog kernela na novim podacima ocekivanih dimenzija p x n """
        # Broj primere za predikciju
        p = X.shape[0]
        # Sirenje dimenzija matrice prediktora za predikciju
        X = np.expand_dims(X, axis=0) # dim: 1 x p x n
        # Prosireni skup za predikciju
        X = np.tile(X, (self.m,1,1)) # dim: m x p x n
        X = np.swapaxes(X, 0, 1) # dim: p x m x n
        # Prosirenje obucavajuceg skupa
        X_train = np.tile(self.X_train, (p,1,1)) # dim: p x m x n
        # Euklidska distanca
        dist = np.sum(np.square(X_train - X), axis=2) # dim: p x m
        # Racunanje kernela za sve primere
        gauss = np.exp( dist * self.gamma ) # p x m
        return gauss

    def fit(self, X: np.ndarray, y: np.ndarray, disp: bool = False):

        # Matrice za resavanje problema kvadratnog programiranja
        self.m = X.shape[0]
        # Funkcija koju treba minimizirati
        if self.kernel == 'linear':
            self.P = np.multiply( (X @ X.T), (y @ y.T) )
        elif self.kernel == 'gauss':
            # Cuvanje matrice prediktora i izlaza
            self.X_train = np.expand_dims(X, axis=0) # dim: 1 x m x n
            self.y_train = y # dim: m x 1
            self.P = np.multiply(self.rbf(X), (y @ y.T) )
        self.q = - np.ones((self.m,1))
        # Ogranicenje nejednakosti
        self.G = - np.eye(self.m)
        self.h = np.zeros((self.m, 1))
        if self.C is not None:
            self.G = np.concatenate((self.G, np.eye(self.m)), axis=0)
            self.h = np.concatenate((self.h, self.C * np.ones((self.m, 1))), axis=0)
        # Ogranicenje jednakosti
        self.A = y.reshape(1,-1)
        self.b = np.zeros((1,1))

        # Racunanje Lagranzovih koeficijenata
        self.alpha = cvxopt_solve_qp(self.P, self.q, self.G, self.h, self.A, self.b)
        # Racunanje parametara
        self.__calculate_params(X, y)
        # Vizualizacija rezultata
        if disp:
            self.visualize(X, y)

    def __calculate_params(self, X: np.ndarray, y: np.ndarray, tol: float = 1e-6):
        """ Racunanje parametara modela """
        # Vektor w
        if self.kernel == 'linear':
            self.w = ( self.alpha.flatten() * y.flatten() ).reshape(1,-1) @ X

        # True na mestima indeksa nosecih vektora
        alpha = deepcopy(self.alpha)
        if self.C is not None:
            alpha[abs(alpha - self.C) < tol] = 0
        sv_index = np.argmax(alpha)

        # Slobodan clan separacione prave
        if self.kernel == 'linear':
            self.b = np.squeeze(1/y[sv_index] - self.w @ X[sv_index].reshape(1,-1).T )
        elif self.kernel == 'gauss':
            # Sirenje dimenzija noseceg vektora
            X_sv = X[sv_index].reshape(1,-1) 
            self.b = np.squeeze(1/y[sv_index] - (self.rbf(X_sv) @ ( np.multiply(self.alpha.reshape(-1,1), self.y_train.reshape(-1,1)) ).flatten()))

    def get_support_vectors_inds(self, tol: float = 1e-6):

        # Anuliranje vrednosti u okolini 0 i C
        alpha = deepcopy(self.alpha)
        # if self.C is not None:
        #     alpha[abs(alpha - self.C) < tol] = 0
        alpha[alpha < tol] = 0

        # Indeksi nosecih vektora
        inds = [i for i in range(self.m) if alpha[i] > 0]

        return inds

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Predikcija na novom skupu podataka """
        return np.sign(self.gm(X))

    def gm(self, X: np.ndarray) -> np.ndarray:
        """ Geometrijska margina """
        if self.kernel == 'linear':
            return ((self.w @ X.T) + self.b).flatten()
        elif self.kernel == 'gauss':
            return (self.rbf(X) @ ( np.multiply(self.alpha.reshape(-1,1), self.y_train.reshape(-1,1)) ) + self.b).flatten()

    def fm(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """ Funkcionalna margina """
        return y.flatten() * self.gm(X)

    def __get_separation_line(self, X: np.ndarray, npoints: int = 200, tol: float = 1e-2) -> np.ndarray:

        if self.kernel == 'linear':
            # Pocetna i krajnja tacka separacione prave
            start, stop = np.min(X[:,0]), np.max(X[:,0])
            # Generisanje vrednosti prvog prediktora
            x1 = np.linspace(start, stop, npoints)
            # Generisanje vrednosti drugog prediktora
            x2 = (- self.w[0,0].flatten() * x1 - self.b) / self.w[0,1]
        elif self.kernel == 'gauss':
            # Pocetna i krajnja tacka prvog i drugog prediktora
            start1, stop1 = np.min(X[:,0]) - .2, np.max(X[:,0]) + .2
            start2, stop2 = np.min(X[:,1]) - .2, np.max(X[:,1]) + .2
            # Generisanje vrednosti prvog i drugog prediktora
            x1 = np.repeat(np.linspace(start1, stop1, npoints), npoints).reshape(-1,1)
            x2 = np.tile(np.linspace(start2, stop2, npoints), npoints).reshape(-1,1)
            X = np.concatenate((x1,x2), axis=1)
            # Racunanje geometrijske margine
            gm = self.gm(X).flatten().tolist()
            # Indeksi separacione prave
            ind_sep = [ind for ind in range(X.shape[0]) if abs(gm[ind]) < tol]
            x1, x2 = X[ind_sep,0], X[ind_sep,1]

        return x1, x2

    def visualize(self, X: np.ndarray, y: np.ndarray):

        # Generisanje separacione linije
        x1, x2 = self.__get_separation_line(X)

        # Indeksi nosecih vektora
        sv_inds = self.get_support_vectors_inds()
        X_sv = X[sv_inds]

        # Razdvajanje klasa
        X1, X2 = X[(y == -1).flatten()], X[(y == +1).flatten()]

        # Prikaz rezultata
        plt.figure(figsize=(12,6))
        plt.scatter(X1[:,0], X1[:,1], s=50, c='b', marker='x', label='y = -1')
        plt.scatter(X2[:,0], X2[:,1], s=50, c='r', marker='o', label='y = +1')
        plt.scatter(X_sv[:,0], X_sv[:,1], s=20, c='y', marker='s', label='noseci vektori')
        plt.plot(x1, x2, 'g-.', label='separaciona prava')
        plt.title('Primeri iz obučavajućeg skupa')
        plt.xlabel('Prediktor $x_1$')
        plt.ylabel('Prediktor $x_2$')
        plt.legend()
        plt.show()

    def reset(self, C: float = None) -> None:
        """ Reinicijalizacija modela """
        self.w = None
        self.b = None
        self.C = C

    def hinge_loss(self, X: np.ndarray, y: np.ndarray):
        """ Funkcija gubitaka """
        # Racunanje geometrijske margine tacaka
        eta = 1 - self.fm(X, y)
        # Hinge loss
        eta[eta < 0] = 0
        return sum(eta)


def cross_validate(X: np.ndarray, y: np.ndarray, c_values: list, nfolds: int = 4,
                    kernel: str = 'linear', sigma_values: float = None, disp: bool = False) -> float:

    def calculate_score(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                        y_test: np.ndarray, kernel: str, c: float, sigma: float = None) -> float:
        # Model
        svm = SVM(C=c, sigma=sigma, kernel=kernel)
        # Obučavanje
        svm.fit(X_train, y_train)
        # Racunanje funkcije gubitaka
        loss_train = svm.hinge_loss(X_train, y_train)
        loss_test = svm.hinge_loss(X_test, y_test)

        return loss_train, loss_test


    # Razbijanje skupa na strukove
    folds = cv_split(X, y, nfolds)
    # Lista svih vrednosti funkcije gubitka
    loss_train = []
    loss_test = []

    if kernel == 'gauss':
        # Velicine listi
        c_len, sigma_len = len(c_values), len(sigma_values)
        # Kreiranje kombinacija hiper-parametara
        c_array, sigma_array = np.array(c_values), np.array(sigma_values)
        c_array = np.repeat(c_array, sigma_len)
        sigma_array = np.tile(sigma_array, c_len)
        # Provera da li su nizovi istih duzina
        assert c_array.size == sigma_array.size
        # Lista hiper-parametara
        hyperparams = [ (c_array[i], sigma_array[i]) for i in range(c_array.size)]


    for i, fold in enumerate(tqdm(folds)):

        # Strukove za obucavanje
        train_folds_inds = list(filter(lambda x: x != i, [ind for ind in range(nfolds)]))
        train_folds = list(map(lambda x: folds[x], train_folds_inds))
        X_train, y_train = merge_folds(train_folds)
        # Struk za testiranje
        X_test, y_test = fold

        # Fiksiranje arugmenata pomocne funkcije
        calculate_score_ = partial(calculate_score, X_train, y_train, X_test, y_test, kernel)
        # Racunanje hinge gubitka na testirajucem skupu
        if kernel == 'linear':
            scores = list(map(lambda x: calculate_score_(x), c_values))
        elif kernel == 'gauss':
            scores = list(map(lambda x: calculate_score_(*x), hyperparams))

        loss_train.append(list(map(lambda x: x[0], scores)))
        loss_test.append(list(map(lambda x: x[1], scores)))

    # Konverzija list -> np.ndarray
    loss_train = np.array(loss_train)
    loss_test = np.array(loss_test)
    # Srednja vrednost greske
    loss_train_mean = np.mean(loss_train, axis=0)
    loss_test_mean = np.mean(loss_test, axis=0)
    # Standardna devijacija greske
    loss_train_std = np.std(loss_train, axis=0)
    loss_test_std = np.std(loss_test, axis=0)


    if disp:

        if kernel == 'linear':

            plt.figure(figsize=(12,6))
            plt.title("Krive funkcije gubitka u zavisnosti od hiper-parametra C")
            plt.xlabel("Vrednost C")
            plt.ylabel("Funkcija gubitaka")
            plt.plot(c_values, loss_train_mean, label="Trening",
                        color="darkorange")
            plt.fill_between(c_values, loss_train_mean - loss_train_std,
                            loss_train_mean + loss_train_std, alpha=0.2,
                            color="darkorange")
            plt.plot(c_values, loss_test_mean, label="Validacija",
                        color="navy")
            plt.fill_between(c_values, loss_test_mean - loss_test_std,
                            loss_test_mean + loss_test_std, alpha=0.2,
                            color="navy")
            plt.legend(loc="best")
            plt.show()

        elif kernel == 'gauss':
            # Predimenzionisanje srednje vrednosti gubitaka na test skupu
            loss_test = loss_test_mean.reshape(c_len, sigma_len)
            c_best, sigma_best = hyperparams[np.argmin(loss_test_mean)]
            c_arr = np.array(c_values)
            s_arr = np.array(sigma_values)
            fig, ax = plt.subplots(figsize=(10,25))
            CS = ax.contour(s_arr, c_arr, loss_test, levels=20)
            ax.clabel(CS, inline=1, fontsize=10)
            ax.scatter(sigma_best, c_best, s=100, c='k', marker='X', label='minimum')
            ax.set_title("Konture funkcije gubitka u zavisnosti od hiper-parametara C i $\sigma$")
            ax.set_xlabel("Vrednost sigma")
            ax.set_ylabel("Vrednost C")
            plt.legend()
            plt.show()


    if kernel == 'linear':
        return c_values[np.argmin(loss_test_mean)]
    elif kernel == 'gauss':
        return hyperparams[np.argmin(loss_test_mean)]