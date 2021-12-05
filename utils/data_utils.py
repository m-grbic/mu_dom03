import numpy as np
import pandas as pd
import os
import random


def read_data(path: str = './data/svmData_ls.csv') -> tuple:
    """ Funkcija za ucitavanje podataka """

    if not os.path.exists(path):
        raise FileExistsError(f"Podaci na putanji {path} ne postoje!")

    # Ucitavanje podataka kao matricu
    data = pd.read_csv(path).to_numpy()

    # Razdvajanje prediktora i vektora ocekivanih izlaza
    X, y = data[:,:-1], data[:,-1].reshape(-1,1)

    return X, y


def train_test_split(X: np.ndarray, y: np.ndarray, train_ratio: int = 0.8, random_state: int = 1234) -> tuple:
    """ Funkcija za podelu podataka na obucavajuci i testirajuci skup """
    # Seme generatora slucajnih brojeva
    random.seed(random_state)

    # Indeksi
    ind = [i for i in range(X.shape[0])]

    # Broj podataka za obucavanje
    n_train = int(len(ind) * train_ratio)

    # Odabiranje podataka za obucavanje
    ind_train = random.sample(ind, n_train)
    ind_test = [i for i in ind if i not in ind_train]

    X_train, y_train = X[ind_train], y[ind_train]
    X_test, y_test = X[ind_test], y[ind_test]

    assert (X_train.shape[0] + X_test.shape[0]) == X.shape[0]
    assert (y_train.shape[0] + y_test.shape[0]) == y.shape[0]
    assert set(ind_train + ind_test) == set(ind)

    return X_train, y_train, X_test, y_test


def cv_split(X: np.ndarray, y: np.ndarray, n_folds: int = 4, random_state: int = 1234) -> list:
    """ Podela podataka u strukove """

    # Seme generatora slucajnih brojeva
    np.random.seed(random_state)

    # Pomocna lista strukova
    folds_joint_list = []
    # Konacan lista strukova
    folds = []
    # Lista indeksa
    ind = [i for i in range(X.shape[0])]

    for y_c in [-1, 1]:
        # Indeksi trenutne klase

        ind_ = list(filter(lambda x: y[x] == y_c, ind))
        # Izdvajanje podatak trenutne klase
        X_, y_ = X[ind_], y[ind_]
        # Spajanje ulazno-izlaznih parova
        Xy = np.concatenate((X_, y_), axis=1)
        # Podela na strukove
        Xy_folds = np.array_split(Xy, n_folds)
        # Dodavanje u listu
        folds_joint_list.append(Xy_folds)

    for i in range(n_folds):

        # Izdvajanje ulazno izlaznih parova obe klase
        Xy1 = folds_joint_list[0][i]
        Xy2 = folds_joint_list[1][i]
        # Spajanje klasa
        Xy = np.concatenate((Xy1, Xy2), axis=0)
        # Mesanje podataka
        np.random.shuffle(Xy)
        # Razdvajanje ulazno-izlaznih parova
        X, y = Xy[:,:-1], Xy[:,-1].reshape(-1,1)
        folds.append((X, y))

    return folds


def merge_folds(folds):
    """ Funkcija za spajanje strukova """

    X = np.concatenate(list(map(lambda x: x[0], folds)), axis=0)
    y = np.concatenate(list(map(lambda x: x[1], folds)), axis=0)

    return X, y





