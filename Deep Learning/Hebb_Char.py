from WildanNeuralNetwork import Hebb
import numpy as np

print(15*'=' + ' Hebb Learning Pada Pengenalan Karakter Huruf ' + 15*'=')

hebb = Hebb()

# Membaca berkas o.txt
berkas = open('../Dataset/Pengenalan Karakter/O.txt', 'r')
o = berkas.read()
berkas.close()

# Membaca berkas x.txt
berkas = open('../Dataset/Pengenalan Karakter/X.txt', 'r')
x = berkas.read()
berkas.close()

# Mengubah pola huruf menjadi list angka bipolar
bipolar_o = hebb.polaToBipolar(o)
bipolar_x = hebb.polaToBipolar(x)

# print(bipolar_o)
# print(bipolar_x)

# Menentukan data latih dan target
data = [bipolar_o, bipolar_x]
# Key pada target harus -1 atau 1, terserah yang -1 itu o atau x dan yang 1 itu o atau x
target = {-1:'o', 1:'x'}

# Proses training
hebb = Hebb(len(bipolar_o))
hebb.train(data, target.keys())

weight, bias = hebb.getWeightBias()
print('Weight: ', weight)
print('Bias: ', bias)

# =========================== Testing =============================

# Membaca berkas mirip_o.txt
berkas = open('../Dataset/Pengenalan Karakter/O_test.txt', 'r')
mirip_o = berkas.read()
berkas.close()

# Membaca berkas mirip_x.txt
berkas = open('../Dataset/Pengenalan Karakter/X_test.txt', 'r')
mirip_x = berkas.read()
berkas.close()

# Mengubah pola huruf menjadi list angka bipolar
bipolar_mirip_o = hebb.polaToBipolar(mirip_o)
bipolar_mirip_x = hebb.polaToBipolar(mirip_x)

# Proses testing
data_test = [bipolar_mirip_o, bipolar_mirip_x]
test = hebb.test(data_test)
print(test)
print('Hasil Hebb Learning:')
for t in test:
    print(target[t])