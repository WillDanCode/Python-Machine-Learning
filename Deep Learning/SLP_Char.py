from WildanNN import SLP, Helper
import numpy as np
from random import *

print(15*'=' + ' Single Layer Perceptron Pada Pengenalan Karakter Huruf ' + 15*'=')

helper = Helper()

# Membaca berkas training
a = helper.bacaFile('../Dataset/Pengenalan Karakter/A.txt')
b = helper.bacaFile('../Dataset/Pengenalan Karakter/B.txt')
c = helper.bacaFile('../Dataset/Pengenalan Karakter/C.txt')
d = helper.bacaFile('../Dataset/Pengenalan Karakter/D.txt')
e = helper.bacaFile('../Dataset/Pengenalan Karakter/E.txt')
j = helper.bacaFile('../Dataset/Pengenalan Karakter/J.txt')
k = helper.bacaFile('../Dataset/Pengenalan Karakter/K.txt')

# Mengubah pola huruf menjadi list angka bipolar
bipolar_a = helper.polaToBipolar(a)
bipolar_b = helper.polaToBipolar(b)
bipolar_c = helper.polaToBipolar(c)
bipolar_d = helper.polaToBipolar(d)
bipolar_e = helper.polaToBipolar(e)
bipolar_j = helper.polaToBipolar(j)
bipolar_k = helper.polaToBipolar(k)

# Menentukan data latih dan target
data = [bipolar_a, bipolar_b, bipolar_c, bipolar_d, bipolar_e, bipolar_j, bipolar_k]
target = np.eye(len(data))
target = np.where(target == 0, -1, 1)

# Memetakan target huruf
targetHuruf = {
    tuple(target[0]):'A',
    tuple(target[1]):'B',
    tuple(target[2]):'C',
    tuple(target[3]):'D',
    tuple(target[4]):'E',
    tuple(target[5]):'J',
    tuple(target[6]):'K'
}

# Proses training
# Nilai alpha dan threshold harus pas agar bisa mendeteksi semua huruf
# slp = SLP(sizeInput=len(bipolar_a), sizeOutput=len(target), alpha=round(random(), 1), threshold=round(random(), 1))
slp = SLP(sizeInput=len(bipolar_a), sizeOutput=len(target), alpha=0.1, threshold=0.3)
slp.train(data, target)

weight, bias = slp.getWeightBias()
print('Weight: ', weight)
print('Bias: ', bias)
# print('Alpha: ', slp.alpha)
# print('Threshold: ', slp.threshold)

# =========================== Testing =============================

# Membaca berkas testing
a_test = helper.bacaFile('../Dataset/Pengenalan Karakter/A_test.txt')
b_test = helper.bacaFile('../Dataset/Pengenalan Karakter/B_test.txt')
c_test = helper.bacaFile('../Dataset/Pengenalan Karakter/C_test.txt')
d_test = helper.bacaFile('../Dataset/Pengenalan Karakter/D_test.txt')
e_test = helper.bacaFile('../Dataset/Pengenalan Karakter/E_test.txt')
j_test = helper.bacaFile('../Dataset/Pengenalan Karakter/J_test.txt')
k_test = helper.bacaFile('../Dataset/Pengenalan Karakter/K_test.txt')

# Mengubah pola huruf menjadi list angka bipolar
bipolar_a_test = helper.polaToBipolar(a_test)
bipolar_b_test = helper.polaToBipolar(b_test)
bipolar_c_test = helper.polaToBipolar(c_test)
bipolar_d_test = helper.polaToBipolar(d_test)
bipolar_e_test = helper.polaToBipolar(e_test)
bipolar_j_test = helper.polaToBipolar(j_test)
bipolar_k_test = helper.polaToBipolar(k_test)

# Proses testing
data_test = [bipolar_a_test, bipolar_b_test, bipolar_c_test, bipolar_d_test, bipolar_e_test, bipolar_j_test, bipolar_k_test]
test = slp.test(data_test)
print('Hasil SLP:')
for t in test:
    t = tuple(t)
    print(targetHuruf[t])


"""
Untuk kasus pada training pola huruf ini, nilai alpha dan threshold paling optimal:
Alpha = 0.1
Threshold = 0.3 ; 0.4; 0.5
"""
