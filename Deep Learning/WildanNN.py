import numpy as np
import matplotlib.pyplot as plt

"""
@author: Wildan

Library yang dibuat untuk kebutuhan Artificial Neural Network
"""

# Implementasi jaringan Adaline (Delta Rule)
class Adaline(object):

    def __init__(self, sizeInput=2, sizeOutput=1, alpha=0.1, threshold=0.1):
        """
        Inisialisasi bobot dan bias awal dengan nilai acak

        :param sizeInput: int
            Banyaknya input neuron pada jaringan Adaline. Harus sesuai dengan banyaknya parameter (fitur pada data latih)
        :param sizeOutput: int
            Banyaknya output neuron pada jaringan Adaline
        :param alpha: float 0.1 <= sizeInput*alpha <= 1
            Nilai learning rate
        :param threshold: float
            Nilai ambang batas
        :return: None
        """

        self.sizeInput = sizeInput
        self.sizeOutput = sizeOutput
        self.alpha = alpha
        self.threshold = threshold
        self.weight = np.random.random(sizeInput)
        self.bias = np.random.random()
        self.target_is_bipolar = False

    def aktivasi_biner(self,x):
        """
        Fungsi aktivasi step function jika target biner

        :param x: int64
            Nilai yang akan dicari output aktivasinya
        :return: int64
            Bilangan 0 atau 1 sesuai dengan kondisi if
        """

        if x < 0:
            return 0
        else:
            return 1

    def aktivasi_bipolar(self,x):
        """
        Fungsi aktivasi step function jika target bipolar

        :param x: int64
            Nilai yang akan dicari output aktivasinya
        :return: int64
            Bilangan -1 atau 1 sesuai dengan kondisi if
        """

        if x < 0:
            return -1
        else:
            return 1

    def getBias(self):
        """
        Mendapatkan bias jaringan Adaline setelah proses training

        :return: bias
            Nilai bias
        """

        return self.bias

    def getWeight(self):
        """
        Mendapatkan bobot jaringan Adaline setelah proses training

        :return: weight
            Nilai bobot
        """

        return self.weight

    def getWeightBias(self):
        """
        Mendapatkan bobot dan bias jaringan Adaline setelah proses training

        :return: weight,bias
            Nilai bobot dan bias
        """

        return self.weight,self.bias

    def train(self,train_data,train_target):
        """
        Proses pelatihan jaringan Adaline

        :param train_data: matriks angka bipolar {-1, 1}
            Kumpulan data latih
        :param train_target: list angka bipolar {-1, 1} atau angka biner {0, 1}
            Kumpulan target yang sesuai dengan data latih
        :return: None
        """

        max_error = self.threshold

        # Cek apakah target berupa bipolar atau bukan
        if np.array_equiv(np.sort(np.unique(train_target)),[-1,1]) is True:
            self.target_is_bipolar = True
        else:
            self.target_is_bipolar = False

        while max_error >= self.threshold:
            max_error = 0
            for data,target in zip(train_data,train_target):
                y = v = np.dot(self.weight,data) + self.bias
                delta_w = self.alpha * (target-y) * data
                self.weight = self.weight + delta_w
                self.bias = self.bias + self.alpha * (target-y)
                max_error = np.max(np.append(delta_w,max_error))

    def test(self,test_data):
        """
        Mendapatkan output dari satu data uji menggunakan jaringan Adaline dengan bobot dan bias input

        :param test_data: Matriks (list of list) int64
            Data yang akan ditentukan outputnya menggunakan jaringan Adaline
        :return: List of float
            Nilai -1 atau 1 dalam bentuk list
        """

        output = np.array([])
        for data in test_data:
            v = np.dot(self.weight, data) + self.bias
            if self.target_is_bipolar is True:
                y = self.aktivasi_bipolar(v)
            else:
                y = self.aktivasi_biner(v)
            output = np.append(output, y)

        return output

# Implementasi untuk perhitungan distance
class Distance():

    def __init__(self):
        pass

    # Manhattan distance
    def manhattan(self, data1, data2):
        hasil = np.sum(np.abs(data1 - data2))
        return hasil

    # Euclidean distance
    def euclidean(self, data1, data2):
        hasil = np.sqrt(np.sum((data1 - data2)**2))
        return hasil

    # Minkowski distance
    def minkowski(self, data1, data2, jumlahAtribut):
        hasil = np.power(np.sum(np.abs(data1 - data2) ** jumlahAtribut), 1/jumlahAtribut)
        return hasil

    # Supremum distance
    def supremum(self, data1, data2):
        hasil = np.max(np.abs(data1 - data2))
        return hasil

# Implementasi jaringan Hebb
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

    def getBias(self):
        """
        Mendapatkan bias jaringan Hebb setelah proses training

        :return: bias
            Nilai bias
        """

        return self.bias

    def getWeight(self):
        """
        Mendapatkan bobot jaringan Hebb setelah proses training

        :return: weight
            Nilai bobot
        """

        return self.weight

    def getWeightBias(self):
        """
        Mendapatkan bobot dan bias jaringan Hebb setelah proses training

        :return: weight,bias
            Nilai bobot dan bias
        """

        return self.weight,self.bias

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

class Helper():

    def bacaFile(self, pathFile):
        berkas = open(pathFile, 'r')
        isi = berkas.read()
        berkas.close()
        return isi

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

# Implementasi jaringan Learning Vector Quantization
class LVQ(object):

    def __init__(self, sizeInput, sizeOutput, max_epoch, alpha=np.random.random(), threshold=np.random.random(), version='1'):
        """
        Inisialisasi class (constructor)
        :param sizeInput (int): Banyaknya input neuron sesuai dengan banyaknya parameter (fitur pada data latih)
        :param sizeOutput (int): Banyaknya output neuron sesuai dengan banyaknya label (kelas pada data latih)
        :param max_epoch (int): Maksimal epoch yang diizinkan
        :param alpha (float): learning rate
        :param threshold (float): nilai ambang batas
        :param version (string): versi dari jaringan LVQ. Bisa diisi dengan '1', '2', '2.1', '3'
        """

        self.sizeInput = sizeInput
        self.sizeOutput = sizeOutput
        self.max_epoch = max_epoch
        self.alpha = alpha
        self.threshold = threshold
        self.version = version
        self.weight = np.zeros((sizeOutput, sizeInput))

    def getWeight(self):
        """
        Mendapatkan bobot jaringan LVQ setelah proses training

        :return: weight (nilai bobot)
        """

        return self.weight

    def train(self,train_data,train_target):
        """
        Proses pelatihan jaringan LVQ
        :param train_data (numpy array): Matriks yang berisi data latih
        :param train_target (numpy array): Array yang berisi label dari data latih
        :return: bobot dan label dari bobot
        """

        weight_label, label_index = np.unique(train_target, True)
        print(weight_label)
        print(label_index)
        # Inisialisasi bobot
        self.weight = train_data[label_index].astype(np.float)
        # print(self.weight)
        # Hapus data yang digunakan untuk inisialisasi bobot
        train_data = np.delete(train_data, label_index, axis=0)
        train_target = np.delete(train_target, label_index, axis=0)

        epoch = 0
        iterasi = 0
        while epoch <= self.max_epoch:
            epoch += 1
            # print('\nEpoch', epoch)
            for data, target in zip(train_data, train_target):
                iterasi += 1
                # print('Iterasi', iterasi)
                distance = np.sqrt(np.sum((data - self.weight) ** 2, axis=1))
                idx_min = np.argmin(distance)
                idx_sort = np.argsort(distance)
                idx_winner, idx_runnerUp = idx_sort[0], idx_sort[1]
                min_distance = min(distance[idx_winner]/distance[idx_runnerUp], distance[idx_runnerUp]/distance[idx_winner])
                max_distance = max(distance[idx_winner]/distance[idx_runnerUp], distance[idx_runnerUp]/distance[idx_winner])
                # print(distance, idx_sort)

                if self.version == '2':
                    self.threshold = 0.35
                    if (
                        (weight_label[idx_winner] != weight_label[idx_runnerUp]) and
                        (target == weight_label[idx_runnerUp] and
                        (distance[idx_winner]/distance[idx_runnerUp] > 1-self.threshold and
                         distance[idx_runnerUp]/distance[idx_winner] < 1+self.threshold))
                    ):
                        self.weight[idx_winner] = self.weight[idx_winner] - self.alpha * (data - self.weight[idx_winner])
                        self.weight[idx_runnerUp] = self.weight[idx_runnerUp] + self.alpha * (data - self.weight[idx_runnerUp])
                    else:
                        if target == weight_label[idx_min]:
                            self.weight[idx_min] = self.weight[idx_min] + self.alpha * (data - self.weight[idx_min])
                        else:
                            self.weight[idx_min] = self.weight[idx_min] - self.alpha * (data - self.weight[idx_min])

                elif self.version == '2.1':
                    self.threshold = 0.35
                    if (
                        (target == weight_label[idx_winner] or target == weight_label[idx_runnerUp]) and
                        (min_distance > 1-self.threshold and max_distance < 1+self.threshold)
                    ):
                        self.weight[idx_winner] = self.weight[idx_winner] + self.alpha * (data - self.weight[idx_winner])
                        self.weight[idx_runnerUp] = self.weight[idx_runnerUp] - self.alpha * (data - self.weight[idx_runnerUp])
                    else:
                        if target == weight_label[idx_min]:
                            self.weight[idx_min] = self.weight[idx_min] + self.alpha * (data - self.weight[idx_min])
                        else:
                            self.weight[idx_min] = self.weight[idx_min] - self.alpha * (data - self.weight[idx_min])

                elif self.version == '3':
                    self.threshold = 0.2
                    m = np.random.uniform(0.1, 0.5)
                    beta = m * self.alpha
                    if (min_distance > (1-self.threshold) * (1+self.threshold)):
                        if (weight_label[idx_winner] != weight_label[idx_runnerUp]):
                            if (target == weight_label[idx_winner] or target == weight_label[idx_runnerUp]):
                                self.weight[idx_winner] = self.weight[idx_winner] + self.alpha * (data - self.weight[idx_winner])
                                self.weight[idx_runnerUp] = self.weight[idx_runnerUp] - self.alpha * (data - self.weight[idx_runnerUp])
                        else:
                            self.weight[idx_winner] = self.weight[idx_winner] + beta * (data - self.weight[idx_winner])
                            self.weight[idx_runnerUp] = self.weight[idx_runnerUp] + beta * (data - self.weight[idx_runnerUp])
                    else:
                        if target == weight_label[idx_min]:
                            self.weight[idx_min] = self.weight[idx_min] + self.alpha * (data - self.weight[idx_min])
                        else:
                            self.weight[idx_min] = self.weight[idx_min] - self.alpha * (data - self.weight[idx_min])

                else:
                    if target == weight_label[idx_min]:
                        self.weight[idx_min] = self.weight[idx_min] + self.alpha * (data - self.weight[idx_min])
                    else:
                        self.weight[idx_min] = self.weight[idx_min] - self.alpha * (data - self.weight[idx_min])

            self.alpha = self.alpha * (1 - epoch / self.max_epoch)

        weight_class = (self.weight, weight_label)
        return weight_class

    def test(self, test_data, weight_class):
        """
        Proses pengujian jaringan LVQ
        :param test_data (numpy array atau pandas dataframe): Matriks yang berisi data uji
        :param weight_class (tuple): Tuple yang berisi pasangan bobot dan labelnya
        :return: Nilai prediksi label/class
        """

        weight, label = weight_class
        output = []
        for data in test_data:
            distance = np.sqrt(np.sum((data - self.weight) ** 2, axis=1))
            idx_min = np.argmin(distance)
            output.append(label[idx_min])

        return output

# Implementasi jaringan Multi Layer Perceptron / Backpropagation Untuk Klasifikasi
# class MLPClassifier(MLPRegressor):
#     pass

# Untuk membuat layer pada MLP
class Layer:
    
    def __init__(self, neuron, input_dim=None, activation='binary sigmoid', initialize_WeightBias=False):
        """Inisialisasi class (constructor)
        
        Arguments:
            neuron {int} -- jumlah neuron pada layer sekarang
        
        Keyword Arguments:
            input_dim {int} -- jumlah neuron pada layer sebelumnya (default: {None})
            activation {str} -- fungsi aktivasi yang akan digunakan (default: {'binary sigmoid'})
            initialize_WeightBias {bool} -- inisialisasi bobot dan bias (default: {False})
        """

        self.neuron = neuron
        self.input_dim = input_dim
        self.activation_func = activation
        self.weight = np.random.uniform(-0.5, 0.5, (self.input_dim, self.neuron))
        self.bias = np.random.uniform(-0.5, 0.5, self.neuron)

        if initialize_WeightBias:
            self.init_WeightBias()

    def init_WeightBias(self):
        """
        Inisialisasi bobot dan bias menggunakan metode Nguyen-Widrow
        """

        # Scale factor = beta
        scale_factor = 0.7 * (self.neuron ** (1/self.input_dim))
        self.weight = np.random.uniform(-0.5, 0.5, (self.input_dim, self.neuron))
        sqrt_weight = np.sqrt(np.sum(self.weight ** 2))
        self.weight = scale_factor * self.weight / sqrt_weight
        self.bias = np.random.uniform(-scale_factor, scale_factor, self.neuron)

# Implementasi jaringan Multi Layer Perceptron / Backpropagation Untuk Prediksi (Regresi)
class MLPRegressor(object):
    
    def __init__(self, layers=[], max_epoch=100, alpha=np.random.random(), beta=10):
        """Inisialisasi class (constructor)
        
        Keyword Arguments:
            layers {list} -- berisi layer-layer (default: {[]})
            max_epoch {int} -- jumlah maksimum iterasi untuk training (default: {100})
            alpha {float} -- learning rate (default: {np.random.random()})
            beta {int} -- learning rate decayer (default: {10})
        """

        self.layers = layers
        self.max_epoch = max_epoch
        self.alpha = alpha
        self.beta = beta

    def activation(self, x, func='binary sigmoid'):
        """Fungsi aktivasi
        
        Arguments:
            x {float} -- nilai yang akan dicari aktivasinya
        
        Keyword Arguments:
            func {str} -- metode aktivasi (default: {'binary sigmoid'})
        
        Returns:
            float -- hasil aktivasi
        """

        if func == 'binary sigmoid':
            return 1 / (1 + np.exp(-x))
        elif func == 'bipolar sigmoid':
            return (2 / (1 + np.exp(-x))) - 1
        else:
            return
    
    def activation_derivative(self, x, func='binary sigmoid'):
        """Fungsi turunan aktivasi
        
        Arguments:
            x {float} -- nilai yang akan dicari turunan aktivasinya
        
        Keyword Arguments:
            func {str} -- metode turunan aktivasi (default: {'binary sigmoid'})
        
        Returns:
            float -- hasil turunan aktivasi
        """

        activation = self.activation(x, func=func)
        if func == 'binary sigmoid':
            return activation * (1 - activation)
        elif func == 'bipolar sigmoid':
            return 0.5 * (1 + activation) * (1 - activation)
        else:
            return

    def add(self, layer):
        """Untuk menambah layer
        
        Arguments:
            layer {object} -- sebuah objek layer
        """

        self.layers.append(layer)

    def forward(self, data):
        """Untuk melakukan fase forward propagation
        
        Arguments:
            data {list} -- list data
        
        Returns:
            tuple of list -- hasil training dan aktivasinya
        """

        x = data
        activation = np.vectorize(self.activation)
        out_in, out = [], []
        for layer in self.layers:
            n = np.dot(x, layer.weight) + layer.bias
            a = activation(n, layer.activation_func)

            out_in.append(n)
            out.append(a)
            x = a
        
        return (out_in, out)

    def backpropagation(self, data, target, out_in, out, method='sgd'):
        """Untuk melakukan fase backpropagation of error
        Ref : 
        [1] Hagan, M.T., H.B. Demuth, and M.H. Beale, Neural Network Design, Boston, MA: PWS Publishing, 1996.
        
        Arguments:
            data {list} -- list data
            target {int/float} -- target
            out_in {list} -- hasil training feed forward
            out {list} -- aktivasi hasil training feed forward
            method {str} -- metode yang dipakai ({'sgd':'stochastic gradient descent', 'lm':'levenberg-marquardt'})
        
        Returns:
            tuple of list -- delta (selisih) bobot dan bias
        """
        
        p = data
        t = target
        n = out_in
        a = out

        derivative = np.vectorize(self.activation_derivative)
        delta_weight, delta_bias = [], []

        if method == 'lm':
            # Levenberg-Marquardt

            miu = self.alpha
            
            # Menghitung vector of error
            e = (t - a[-1])
            v = np.sum(e**2)

            # Menghitung sensitivity (s)
            s = -derivative(n[-1])
            # v = np.sum(s**2)
            for i, layer in reversed(list(enumerate(self.layers))):
                if i == 0:
                    # Hidden -> Input
                    jacobWeight = np.dot(p[:, None], s[None])
                    jacobBias = s
                    # delta = ((J.T * J + u * I) ** -1) * (J.T * v)
                    deltaWeight = np.dot(np.linalg.inv(np.dot(jacobWeight.T, jacobWeight) + miu * np.identity(jacobWeight.shape[1])), np.dot(jacobWeight.T, v))
                    deltaBias = np.dot(np.linalg.inv(np.dot(jacobBias.T, jacobBias) + miu * np.identity(jacobBias.shape[0])), np.dot(jacobBias.T, v))

                    deltaWeight = deltaWeight.T
                    deltaBias = deltaBias.T

                    delta_weight.append(deltaWeight)
                    delta_bias.append(deltaBias)
                else:
                    # Output -> Hidden
                    jacobWeight = np.dot(a[i-1][:, None], s[None])
                    jacobBias = s
                    # delta = ((J.T * J + u * I) ** -1) * (J.T * v)
                    deltaWeight = np.dot(np.linalg.inv(np.dot(jacobWeight.T, jacobWeight) + miu * np.identity(jacobWeight.shape[1])), np.dot(jacobWeight.T, v))
                    deltaBias =  np.dot(np.linalg.inv(np.dot(jacobBias.T, jacobBias) + miu * np.identity(jacobBias.shape[0])), np.dot(jacobBias.T, v))

                    deltaWeight = deltaWeight.T
                    deltaBias = deltaBias.T
                    
                    delta_weight.append(deltaWeight)
                    delta_bias.append(deltaBias)

                    # Jacobian Matrix = w * derivative(n)
                    s_in = np.dot(s, layer.weight.T)
                    s = s_in * derivative(n[i-1])
                    v = np.sum(s**2)

            delta_weight = delta_weight[::-1]
            delta_bias = delta_bias[::-1]
        else:
            # Stochastic Gradient descent / Steepest descent

            # Menghitung sensitivity (s)
            # error = (target - out[-1]) * derivative(out_in[-1])
            s = -2 * (t - a[-1]) * derivative(n[-1])
            for i, layer in reversed(list(enumerate(self.layers))):
                if i == 0:
                    # Hidden -> Input
                    # Ref : https://stackoverflow.com/questions/39026173/numpy-multiplying-different-shapes
                    # deltaWeight (df/dw) = df/dn * dn/dw
                    deltaWeight = self.alpha * np.dot(p[:, None], s[None])
                    # deltaBias (df/db) = df/dn * dn/db
                    deltaBias = self.alpha * s

                    delta_weight.append(deltaWeight)
                    delta_bias.append(deltaBias)
                else:
                    # Output -> Hidden
                    # deltaWeight (df/dw) = df/dn * dn/dw
                    deltaWeight = self.alpha * np.dot(a[i-1][:, None], s[None])
                    # deltaBias (df/db) = df/dn * dn/db
                    deltaBias = self.alpha * s

                    delta_weight.append(deltaWeight)
                    delta_bias.append(deltaBias)

                    # Jacobian Matrix = w * derivative(n)
                    s_in = np.dot(s, layer.weight.T)
                    s = s_in * derivative(n[i-1])

            delta_weight = delta_weight[::-1]
            delta_bias = delta_bias[::-1]

        return (delta_weight, delta_bias)

    def update(self, delta_weight, delta_bias):
        """Untuk melakukan update bobot dan bias
        
        Arguments:
            delta_weight {list} -- selisih bobot
            delta_bias {list} -- selisih bias
        """
        
        for i, layer in enumerate(self.layers):
            # layer.weight += delta_weight[i]
            # layer.bias += delta_bias[i]
            layer.weight -= delta_weight[i]
            layer.bias -= delta_bias[i]

    def train(self, train_data, train_target, optimizer='sgd'):
        """Proses pelatihan jaringan MLP
        
        Arguments:
            train_data {numpy array} -- matriks data latih
            train_target {numpy array} -- array target latih
        
        Keyword Arguments:
            optimizer {str} -- metode oprimasi training yang digunakan (default: {'sgd'})
        """

        epoch = 0
        iterasi = 0
        while epoch <= self.max_epoch:
            epoch += 1
            # print('\nEpoch ', epoch)
            sse = 0
            for data, target in zip(train_data, train_target):
                iterasi += 1
                print('Iterasi ', iterasi)

                sse_before = sse
                # print('sse before: ', sse_before)

                # Forward Propagation Phase
                out_in, out = self.forward(data)
                e = (target - out[-1])
                sse = np.sum(e**2)
                print('sse: ', sse)

                # Backward Propagation Phase (Error Distribution / Backpropagation of Error)
                delta_weight, delta_bias = self.backpropagation(data, target, out_in, out, method=optimizer)

                # # Weight & Bias Update Phase
                # self.update(delta_weight, delta_bias)

                if sse < sse_before:
                    self.alpha /= self.beta

                    # Weight & Bias Update Phase
                    self.update(delta_weight, delta_bias)
                else:
                    self.alpha *= self.beta

                    # Backward Propagation Phase (Error Distribution / Backpropagation of Error)
                    delta_weight, delta_bias = self.backpropagation(data, target, out_in, out, method=optimizer)

                    # Weight & Bias Update Phase
                    self.update(delta_weight, delta_bias)
                
                print('-' * 40)
            
            print('=' * 40)

    def test(self, test_data):
        """Proses pengujian jaringan MLP
        
        Arguments:
            test_data {numpy array} -- matriks data uji
        
        Returns:
            numpy array -- hasil prediksi
        """

        output = []
        for data in test_data:
            # Forward Propagation Phase
            out_in, out = self.forward(data)
            output.append(out[-1])

        output = np.array(output)
        return output

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

    def getBias(self):
        """
        Mendapatkan bias jaringan SLP setelah proses training

        :return: bias
            Nilai bias
        """

        return self.bias

    def getWeight(self):
        """
        Mendapatkan bobot jaringan SLP setelah proses training

        :return: weight
            Nilai bobot
        """

        return self.weight

    def getWeightBias(self):
        """
        Mendapatkan bobot dan bias jaringan SLP setelah proses training

        :return: weight,bias
            Nilai bobot dan bias
        """

        return self.weight,self.bias

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

# Implementasi jaringan Self Organizing Map
# Note: (yang rectangle dan hexagon masih belum jadi)
class SOM(object):

    def __init__(self, sizeInput, sizeOutput, max_epoch, ordering=2, radius=0, alpha=np.random.random(), beta=np.random.random(), architecture='linear'):
        """
        Inisialisasi class (constructor)
        :param sizeInput (int): Banyaknya input neuron sesuai dengan banyaknya parameter (fitur pada data latih)
        :param sizeOutput (int): Banyaknya output neuron sesuai dengan banyaknya label (kelas pada data latih)
        :param max_epoch (int): Maksimal epoch yang diizinkan
        :param ordering (int) : epoch ordering (epoch yang dilakukan proses ordering phase), berupa kelipatan epoch yang nantinya akan terus mengurangi radius
        :param radius (int) : radius / jarak ketetanggaan
        :param alpha (float): learning rate
        :param threshold (float): nilai ambang batas
        :param architecture (string): arsitektur dari jaringan SOM. Bisa diisi dengan 'linear', 'rectangle', 'hexagon'
        """

        self.sizeInput = sizeInput
        self.sizeOutput = sizeOutput
        self.max_epoch = max_epoch
        self.ordering = ordering
        self.radius = radius
        self.alpha = alpha
        self.beta = beta
        self.architecture = architecture
        self.weight = np.random.random((sizeOutput, sizeInput))

    def getWeight(self):
        """
        Mendapatkan bobot jaringan SOM setelah proses training

        :return: weight (nilai bobot)
        """

        return self.weight

    def train(self,train_data):
        """
        Proses pelatihan jaringan SOM
        :param train_data (numpy array): Matriks yang berisi data latih
        :return: label dari bobot
        """

        epoch = 0
        iterasi = 0
        while epoch <= self.max_epoch:
            epoch += 1
            # print('\nEpoch', epoch)

            if epoch % self.ordering == 0 and self.radius != 0:
                self.radius -= 1

            for data in train_data:
                iterasi += 1
                print('Iterasi', iterasi)
                distance = np.sqrt(np.sum((data - self.weight) ** 2, axis=1))
                # print(data - self.weight)
                idx_min = np.argmin(distance)

                if self.radius > 0:
                    # Update bobot pemenang
                    self.weight[idx_min] = self.weight[idx_min] + self.alpha * (data - self.weight[idx_min])

                    # Update bobot tetangga
                    if self.architecture == 'rectangle':
                        pass
                    elif self.architecture == 'hexagon':
                        pass

                    else:
                        # Tetangga atas (indeks tetangga sebelum indeks pemenang)
                        self.weight[idx_min-self.radius:idx_min] = self.weight[idx_min-self.radius:idx_min] + self.alpha * (data - self.weight[idx_min-self.radius:idx_min])
                        # Tetangga bawah (indeks tetangga setelah indeks pemenang)
                        self.weight[idx_min+1:idx_min+self.radius] = self.weight[idx_min+1:idx_min+self.radius] + self.alpha * (data - self.weight[idx_min+1:idx_min+self.radius])

                elif self.radius == 0:
                    self.weight[idx_min] = self.weight[idx_min] + self.alpha * (data - self.weight[idx_min])

                else:
                    print('Radius harus >= 0')
                    break

            self.alpha *= self.beta


    def test(self, test_data):
        """
        Proses pengujian jaringan SOM
        :param test_data (numpy array atau pandas dataframe): Matriks yang berisi data uji
        :return: Nilai prediksi label/class
        """

        output = []
        for data in test_data:
            distance = np.sqrt(np.sum((data - self.weight) ** 2, axis=1))
            idx_min = np.argmin(distance)
            output.append(idx_min)

        return output