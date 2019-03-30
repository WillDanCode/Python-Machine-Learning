# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 12:07:26 2018

@author: Wildan

Library yang dibuat untuk kebutuhan Machine Learning
"""

import numpy as np
import pandas as pd

#Ubah dataset categorical ke numeric
def catToNum(dataset):
    ds = dataset.copy()
    for col_name in ds.columns:
        if(ds[col_name].dtype == 'object'):
            ds[col_name] = ds[col_name].astype('category')
            ds[col_name] = ds[col_name].cat.codes
#            ds[col_name] = pd.Categorical.from_array(ds[col_name]).codes
    return ds

#Data dalam bentuk array 1D
def manhattanKNN(data1, data2):
#    train = dataTrain[1:len(dataTrain)]
    array1 = np.copy(data1)
    array2 = np.copy(data2)
    hasil = np.sum(np.abs(array1-array2))
    return hasil

#Data dalam bentuk array 1D
def euclideanKNN(data1, data2):
#    train = dataTrain[1:len(dataTrain)]
    array1 = np.copy(data1)
    array2 = np.copy(data2)
    hasil = np.sqrt(np.sum(np.power(array1-array2, 2)))
    return hasil

#Dataset dalam bentuk numpy array
def KNearestNeighbour(datasetTrain, datasetTest, k=5, types='unweighted', distance='manhattan'):
    h, w = datasetTrain.shape
    train = datasetTrain[0:h, 1:w] #asumsi kolom class ada di depan
    test = datasetTest[0:h, 1:w]

    print('\nTrain All\n', datasetTrain)
    print(datasetTrain.shape)
    print('\nTrain\n', train)
    print(train.shape)
    print('\nTest\n', test)
    print(test.shape)

    #Proses KNN
    #hitung jarak
    himpJarak = [] #kumpulan semua jarak data train dengan data test
    for dataTest in test:
        jarak = []
        for dataTrain in train:
            if (distance == 'euclidean'):
                jarak.append(euclideanKNN(dataTrain, dataTest))
            else:
                jarak.append(manhattanKNN(dataTrain, dataTest))
        himpJarak.append(jarak)
    himpJarak = np.array(himpJarak)
    print('\nHimpJarak\n', himpJarak)
    print(himpJarak.shape)

    #ambil jarak berdasarkan k
    ambilJarak = np.sort(himpJarak)[:, :k] #ambil jarak sebanyak k
    hAmbil, wAmbil = ambilJarak.shape
    print('\nAmbilJarak\n', ambilJarak)
    print(ambilJarak.shape)

    #nilai indeks jarak yang sudah disortir k pada data train
    indeks = []
    for i in range(0,hAmbil):
        temp = []
        for j in range(0,wAmbil):
            idx = list(himpJarak[i]).index(ambilJarak[i][j])
            temp.append(idx)
            himpJarak[i][j] = -1
        indeks.append(temp)
    indeks = np.array(indeks)
    print('\nIndeks\n', indeks)
    print(indeks.shape)

    #mengetahui kelas berdasarkan indeks di atas
    kelas = []
    for rowIndeks in indeks:
        kelas.append(datasetTrain[rowIndeks][:, 0:1].flatten())
    kelas = np.array(kelas)
    print('\nKelas\n', kelas)
    print(kelas.shape)

    #melakukan voting kelas terbanyak
    vote = []
    if (types == 'weighted'):
        #kodenya menyusul...

        vote = np.array(vote)
        print('\nVote\n',vote)
        print(vote.shape)
    else:
        for rowKelas in kelas:
            vote.append(max(list(rowKelas), key=list(rowKelas).count))
        vote = np.array(vote)
        print('\nVote\n',vote)
        print(vote.shape)

    hasil = np.insert(test, 0, vote, axis=1)
    return hasil

# Implementasi untuk perhitungan distance
class Distance():

    def __init__(self, dataTrain, dataTest):
        """
        Inisialisasi data

        :param dataTrain: train data berupa numpy array
        :param dataTest: test data berupa numpy array
        """

        self.train = dataTrain
        self.test = dataTest

    # Manhattan distance
    def manhattan(self):
        hasil = np.sum(np.abs(self.test - self.train))
        return hasil

    # Euclidean distance
    def euclidean(self):
        hasil = np.sqrt(np.sum((self.test - self.train)**2))
        return hasil

    # Minkowski distance
    def minkowski(self, jumlahAtribut):
        hasil = np.power(np.sum(np.abs(self.test - self.train) ** jumlahAtribut), 1/jumlahAtribut)
        return hasil

    # Supremum distance
    def supremum(self):
        hasil = np.max(np.abs(self.test - self.train))
        return hasil

# Implementasi untuk algoritme clustering K-Means
class KMeans:
    pass

# Implementasi untuk perhitungan similarity & dissimilarity
class ProximityMeasure():

    def __init__(self):
        pass

    # Similarity
    def sim(self, dissim):
        """
        Nilai similarity semakin besar, maka akurasi makin tinggi
        Nilai dissimilarity semakin kecil, maka akurasi makin tinggi

        :param dissim: nilai dissimilarity
        :return: nilai kebalikan dissimilarity
        """

        return 1 - dissim

    # Dissimilarity untuk atribut nominal / categorical
    def dissimCat(self, train, test):
        """
        :param train: train data berupa DataFrame pandas
        :param test: test data berupa DataFrame pandas
        :return: nilai dissimilarity
        """

        train = np.array(train)
        test = np.array(test)

        # p: banyaknya atribut yang nominal/categorical dalam dataset
        p = train.shape[1]

        dissim = []
        for dataTest in test:
            temp = []
            for dataTrain in train:
                # m: banyak nilai/state yang sama antara data train dan data test
                m = np.sum(dataTrain == dataTest)
                dsim = (p-m) / p
                temp.append(dsim)
            dissim.append(temp)
        dissim = np.array(dissim)
        return dissim

    # Dissimilarity untuk atribut biner
    def dissimBin(self, train, test):
        """
        :param train: train data berupa DataFrame pandas
        :param test: test data berupa DataFrame pandas
        :return: nilai dissimilarity
        """

        train = np.array(train)
        test = np.array(test)

        dissim = []
        for dataTest in test:
            temp = []
            for dataTrain in train:
                # sama: banyaknya nilai biner yang sama antara data train dan data test
                sama = np.sum(dataTrain == dataTest)
                # beda: banyaknya nilai biner yang beda antara data train dan data test
                beda = np.sum(dataTrain != dataTest)
                dsim = beda / (beda+sama)
                temp.append(dsim)
            dissim.append(temp)
        dissim = np.array(dissim)
        return dissim

    # Dissimilarity untuk atribut numerical (interval, ratio)
    def dissimNum(self, train, test, dist='manhattan', jumlahAtribut=3):
        """
        :param train: train data berupa DataFrame pandas
        :param test: test data berupa DataFrame pandas
        :param dist: pilihan distance
        :param jumlahAtribut: parameter yang harus diisikan ketika dist='minkowski'
        :return: jarak data test terhadap data train
        """

        train = np.array(train)
        test = np.array(test)

        # himpJarak = jarak data test terhadap data train
        himpJarak = []
        for dataTest in test:
            jarak = []
            for dataTrain in train:
                distance = Distance(dataTrain, dataTest)
                if dist == 'euclidean':
                    jarak.append(distance.euclidean())
                elif dist == 'minkowski':
                    jarak.append(distance.minkowski(jumlahAtribut))
                elif dist == 'supremum':
                    jarak.append(distance.supremum())
                else:
                    jarak.append(distance.manhattan())
            himpJarak.append(jarak)
        himpJarak = np.array(himpJarak)
        return himpJarak

    def showProximity(self, dissim):
        """
        Karena parameternya nilai dissimilarity, maka yang diambil nilai terkecil dari dissimilarity
        Semakin kecil dissimilarity, semakin mirip suatu data dengan data lain

        :param dissim: nilai dissimilarity yang akan di cek kedekatannya
        :return:
        """

        counter = 0
        for d in dissim:
            mirip = np.argsort(d)[1]
            print('Data ke-', counter, 'paling mirip dengan data ke-', mirip)
            counter += 1

# Implementasi untuk prediksi menggunakan Regresi
class Regression():

    # Mean Absolute Error Evaluation
    def meanAbsoluteError(self, y_true, y_pred):
        """
        :param y_true: nilai y asli, type: ndarray
        :param y_pred: nilai y prediksi, type: ndarray
        :return: nilai MAE
        """

        if type(y_true) or type(y_pred) is not np.ndarray:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

        error = abs(y_true - y_pred)
        mae = np.mean(error)
        return mae

    # Mean Squared Error Evaluation
    def meanSquaredError(self, y_true, y_pred):
        """
        :param y_true: nilai y asli, type: ndarray
        :param y_pred: nilai y prediksi, type: ndarray
        :return: nilai MSE
        """

        if type(y_true) or type(y_pred) is not np.ndarray:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

        error = (y_true - y_pred)**2
        mse = np.mean(error)
        return mse

    # Root Mean Squared Error Evaluation
    def rootMeanSquaredError(self, y_true, y_pred):
        """
        :param y_true: nilai y asli, type: ndarray
        :param y_pred: nilai y prediksi, type: ndarray
        :return: nilai RMSE
        """

        if type(y_true) or type(y_pred) is not np.ndarray:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

        error = (y_true - y_pred)**2
        rmse = np.sqrt(np.mean(error))
        return rmse

    # Interpolasi Lagrange
    def lagrange(self, featureX, featureY, predictX):
        """
        :param featureX: variable independent, type: ndarray
        :param featureY: variable dependent / variable yang diprediksi, type: ndarray
        :param predictX: nilai prediksi fitur x, type: int
        :return: nilai prediksi fitur y, type: int / ndarray
        """

        # List Lagrange
        l = []
        for i, data in enumerate(featureX):
            temp = np.prod((predictX - featureX[featureX != data])) / np.prod(
                (featureX[i] - featureX[featureX != data]))
            l.append(temp)

        # l = np.array(l)
        p = np.sum(featureY * l)
        return p

    # Regresi Linier
    def linearRegression(self, featureX, featureY, n, predictX):
        """
        :param featureX: variable independent, type: ndarray
        :param featureY: variable dependent / variable yang diprediksi, type: ndarray
        :param n: banyaknya data, type: int
        :param predictX: nilai prediksi fitur x, type: int / ndarray
        :return: nilai prediksi fitur y, type: int / ndarray
        """

        sigmaX = np.sum(featureX)
        sigmaY = np.sum(featureY)
        sigmaXY = np.sum(featureX * featureY)
        sigmaX2 = np.sum(featureX ** 2)

        # y = b0 + b1x
        b1 = (n * sigmaXY - sigmaX * sigmaY) / (n * sigmaX2 - sigmaX ** 2)
        b0 = (sigmaY - b1 * sigmaX) / n
        yPred = b0 + b1 * predictX
        return yPred

    # Regresi Linier Berganda (2 variable independent)
    def multiLinearRegression(self, featureX1, featureX2, featureY, predictX1, predictX2):
        """
        :param featureX1: variable independent, type: ndarray
        :param featureX2: variable independent, type: ndarray
        :param featureY: variable dependent / variable yang diprediksi, type: ndarray
        :param predictX1: nilai prediksi fitur x1, type: int / ndarray
        :param predictX2: nilai prediksi fitur x2, type: int / ndarray
        :return: nilai prediksi fitur y, type: int / ndarray
        """

        sigmaX12 = np.sum(featureX1 ** 2)
        sigmaX22 = np.sum(featureX2 ** 2)
        sigmaX1X2 = np.sum(featureX1 * featureX2)
        sigmaX1Y = np.sum(featureX1 * featureY)
        sigmaX2Y = np.sum(featureX2 * featureY)

        # y = b0 + b1x1 + b2x2
        b1 = (sigmaX22 * sigmaX1Y - sigmaX1X2 * sigmaX2Y) / (sigmaX12 * sigmaX22 - sigmaX1X2 ** 2)
        b2 = (sigmaX12 * sigmaX2Y - sigmaX1X2 * sigmaX1Y) / (sigmaX12 * sigmaX22 - sigmaX1X2 ** 2)
        b0 = np.mean(featureY) - b1 * np.mean(featureX1) - b2 * np.mean(featureX2)

        yPred = b0 + b1 * predictX1 + b2 * predictX2
        return yPred