from WildanNeuralNetwork import SLP
import numpy as np

print(15*'=' + ' Single Layer Perceptron Pada Pengenalan Karakter Huruf ' + 15*'=')

# Membaca berkas training
berkas = open('../Dataset/Pengenalan Karakter/A.txt', 'r')
a = berkas.read()
berkas.close()

berkas = open('../Dataset/Pengenalan Karakter/B.txt', 'r')
b = berkas.read()
berkas.close()

berkas = open('../Dataset/Pengenalan Karakter/C.txt', 'r')
c = berkas.read()
berkas.close()

berkas = open('../Dataset/Pengenalan Karakter/D.txt', 'r')
d = berkas.read()
berkas.close()

berkas = open('../Dataset/Pengenalan Karakter/E.txt', 'r')
e = berkas.read()
berkas.close()

berkas = open('../Dataset/Pengenalan Karakter/J.txt', 'r')
j = berkas.read()
berkas.close()

berkas = open('../Dataset/Pengenalan Karakter/K.txt', 'r')
k = berkas.read()
berkas.close()

slp = SLP()

# Mengubah pola huruf menjadi list angka bipolar
bipolar_a = slp.polaToBipolar(a)
bipolar_b = slp.polaToBipolar(b)
bipolar_c = slp.polaToBipolar(c)
bipolar_d = slp.polaToBipolar(d)
bipolar_e = slp.polaToBipolar(e)
bipolar_j = slp.polaToBipolar(j)
bipolar_k = slp.polaToBipolar(k)

# Menentukan data latih dan target
data = [bipolar_a, bipolar_b, bipolar_c, bipolar_d, bipolar_e, bipolar_j, bipolar_k]
target = np.eye(7)
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
slp = SLP(len(bipolar_a))
slp.trainChar(data, target)

weight, bias = slp.getWeightBias()
print('Weight: ', weight)
print('Bias: ', bias)

# =========================== Testing =============================

# Membaca berkas testing
berkas = open('../Dataset/Pengenalan Karakter/A_test.txt', 'r')
a_test = berkas.read()
berkas.close()

berkas = open('../Dataset/Pengenalan Karakter/B_test.txt', 'r')
b_test = berkas.read()
berkas.close()

berkas = open('../Dataset/Pengenalan Karakter/C_test.txt', 'r')
c_test = berkas.read()
berkas.close()

berkas = open('../Dataset/Pengenalan Karakter/D_test.txt', 'r')
d_test = berkas.read()
berkas.close()

berkas = open('../Dataset/Pengenalan Karakter/E_test.txt', 'r')
e_test = berkas.read()
berkas.close()

berkas = open('../Dataset/Pengenalan Karakter/J_test.txt', 'r')
j_test = berkas.read()
berkas.close()

berkas = open('../Dataset/Pengenalan Karakter/K_test.txt', 'r')
k_test = berkas.read()
berkas.close()

# Mengubah pola huruf menjadi list angka bipolar
bipolar_a_test = slp.polaToBipolar(a_test)
bipolar_b_test = slp.polaToBipolar(b_test)
bipolar_c_test = slp.polaToBipolar(c_test)
bipolar_d_test = slp.polaToBipolar(d_test)
bipolar_e_test = slp.polaToBipolar(e_test)
bipolar_j_test = slp.polaToBipolar(j_test)
bipolar_k_test = slp.polaToBipolar(k_test)

# Proses testing
data_test = [bipolar_a_test, bipolar_b_test, bipolar_c_test, bipolar_d_test, bipolar_e_test, bipolar_j_test, bipolar_k_test]
test = slp.testChar(data_test)
print('Hasil SLP:')
print(test)
for t in test:
    t = tuple(t)
    print(targetHuruf[t])