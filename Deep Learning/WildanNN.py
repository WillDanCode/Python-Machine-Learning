import numpy as np
import matplotlib.pyplot as plt

"""
@author: Wildan

Library yang dibuat untuk kebutuhan Artificial Neural Network
"""

#Implementasi jaringan Hebb
class Hebb(object):

    def __init__(self,size=2):
        """
        Inisialisasi bobot dan bias awal dengan nilai 0

        :param size: int
            Banyaknya input pada jaringan Hebb. Harus sesuai dengan banyaknya parameter (fitur pada data latih)
        """

        self.size = size
        self.bias = 0
        # Panjang list weight harus sama dengan banyaknya neuron input
        self.weight = np.zeros(size)

    def train(self,train_data,train_target):
        """
        Proses pelatihan jaringan Hebb

        :param train_data: matriks angka bipolar {-1, 1}
            Kumpulan data latih
        :param train_target: list angka bipolar {-1, 1}
            Kumpulan target yang sesuai dengan data latih
        :return: None
        """

        # looping untuk setiap pasangan data latih dan target
        for data,target in zip(train_data,train_target):
            # w (baru) = w (lama) + data latih * target
            self.weight = self.weight + data * target
            # b (baru) = b (lama) + target
            self.bias = self.bias + target

    def aktivasi(self,x):
        """
        Fungsi aktivasi step function dengan output bipolar

        :param x: int64
            Nilai yang akan dicari output aktivasinya
        :return: int64
            Bilangan -1 atau 1 sesuai dengan kondisi if
        """

        if x < 0:
            return -1
        else:
            return 1

    def test(self,test_data):
        """
        Mendapatkan output dari satu data uji menggunakan jaringan Hebb dengan bobot dan bias input

        :param test_data: Matriks (list of list) int64
            Data yang akan ditentukan outputnya menggunakan jaringan Hebb
        :return: List of float
            Nilai -1 atau 1 dalam bentuk list
        """

        # total perkalian input dan bobot (weight) dapat dilakukan dengan fungsi dot product pada numpy
        output = np.array([])
        for data in test_data:
            v = np.dot(self.weight,data) + self.bias
            y = self.aktivasi(v)
            output = np.append(output, y)

        return output

    def getWeightBias(self):
        """
        Mendapatkan bobot dan bias jaringan Hebb setelah proses training

        :return: weight,bias
            Nilai bobot dan bias
        """

        return self.weight,self.bias

    def getWeight(self):
        """
        Mendapatkan bobot jaringan Hebb setelah proses training

        :return: weight
            Nilai bobot
        """

        return self.weight

    def getBias(self):
        """
        Mendapatkan bias jaringan Hebb setelah proses training

        :return: bias
            Nilai bias
        """

        return self.bias

    def polaToBipolar(self, pola):
        """
        Mengubah pola-pola huruf menjadi sebuah list bipolar

        :param pola: txt pola
            Pola huruf yang akan digunakan untuk pengenalan karakter
        :return: list of float
            List yang berisi bilangan bipolar hasil replace
        """

        pola = pola.replace("#", "1,")
        pola = pola.replace(".", "-1,")
        pola = pola.replace("\n", '')
        angka = np.fromstring(pola[:-1], dtype=int, sep=',')
        return angka

# Implementasi jaringan Single Layer Perceptron
class SLP(object):

    def __init__(self, sizeInput=2, sizeOutput=1, alpha=1, threshold=0.1):
        """
        Inisialisasi bobot dan bias awal dengan nilai 0

        :param sizeInput: int
            Banyaknya input neuron pada jaringan SLP. Harus sesuai dengan banyaknya parameter (fitur pada data latih)
        :param sizeOutput: int
            Banyaknya output neuron pada jaringan SLP
        :param alpha: float 0 < alpha <= 1
            Nilai learning rate
        :param threshold: float
            Nilai ambang batas
        :return: None
        """

        self.sizeInput = sizeInput
        self.sizeOutput = sizeOutput
        self.alpha = alpha
        self.threshold = threshold
        self.weight = np.zeros((sizeInput, sizeOutput))
        self.bias = np.zeros(sizeOutput)

    def train(self,train_data,train_target):
        """
        Proses pelatihan jaringan SLP

        :param train_data: Matriks (list of list) int64
            Matriks yang berisi list nilai bipolar dari pola-pola huruf training
        :param train_target: Matrix of float
            Matriks yang setiap barisnya mewakili 1 huruf
        :return: None
        """

        # Fungsi vectorize pada numpy digunakan agar fungsi aktivasi mampu menerima input list tanpa looping
        v_aktivasi = np.vectorize(self.aktivasi)
        stop = False
        # epoch = 0
        # iterasi = 0
        while stop is False:
            stop = True
            # epoch += 1
            # print('\nEpoch', epoch)
            # data = 63, target = 7, train_data = (7,63), train_target = (7,7)
            # data = 2, target = 1, train_data = (4,2), train_target = (4,)
            for data,target in zip(train_data,train_target):
                # iterasi += 1
                # print('Iterasi', iterasi)
                v = np.dot(data,self.weight) + self.bias
                y = v_aktivasi(v)
                # print('y:', y)
                # print('Bobot Sebelum:', self.weight)
                # print('Bias Sebelum:', self.bias)

                if type(target) is not np.ndarray:
                    target = [target]

                for i in range(len(data)):
                    for j in range(len(target)):
                        if y[j] != target[j]:
                            stop = False
                            # w (baru) = w (lama) + alpha * target * data latih
                            self.weight[i][j] = self.weight[i][j] + self.alpha * target[j] * data[i]
                            # b (baru) = b (lama) + alpha * target
                            self.bias[j] = self.bias[j] + self.alpha * target[j]

                # print('Bobot Sesudah:', self.weight)
                # print('Bias Sesudah:', self.bias)

    def test(self,test_data):
        """
        Mendapatkan output dari satu data uji menggunakan jaringan SLP dengan bobot dan bias input

        :param test_data: Matriks (list of list) int64
            Matriks yang berisi list nilai bipolar dari pola-pola huruf
        :return: Matrix of float
            Nilai -1 atau 1 dalam bentuk matriks dengan setiap baris mewakili 1 huruf
        """

        output = []
        v_aktivasi = np.vectorize(self.aktivasi)
        for data in test_data:
            v = np.dot(data,self.weight) + self.bias
            y = v_aktivasi(v)
            output.append(y)

        output = np.array(output)
        return output

    def aktivasi(self,x):
        """
        Fungsi aktivasi step function dengan output bipolar

        :param x: int64
            Nilai yang akan dicari output aktivasinya
        :return: int64
            Bilangan -1 atau 1 sesuai dengan kondisi if
        """

        if x > self.threshold:
            return 1
        if x < -self.threshold:
            return -1
        else:
            return 0

    def getWeightBias(self):
        """
        Mendapatkan bobot dan bias jaringan SLP setelah proses training

        :return: weight,bias
            Nilai bobot dan bias
        """

        return self.weight,self.bias

    def getWeight(self):
        """
        Mendapatkan bobot jaringan SLP setelah proses training

        :return: weight
            Nilai bobot
        """

        return self.weight

    def getBias(self):
        """
        Mendapatkan bias jaringan SLP setelah proses training

        :return: bias
            Nilai bias
        """

        return self.bias

    def polaToBipolar(self, pola):
        """
        Mengubah pola-pola huruf menjadi sebuah list bipolar

        :param pola: txt pola
            Pola huruf yang akan digunakan untuk pengenalan karakter
        :return: list of float
            List yang berisi bilangan bipolar hasil replace
        """

        pola = pola.replace("#", "1,")
        pola = pola.replace(".", "-1,")
        pola = pola.replace("\n", '')
        angka = np.fromstring(pola[:-1], dtype=int, sep=',')
        return angka