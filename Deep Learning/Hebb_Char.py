from WildanNN import Hebb, Helper
import numpy as np

print(15*'=' + ' Hebb Learning Pada Pengenalan Karakter Huruf ' + 15*'=')

helper = Helper()

# Membaca berkas training
o = helper.bacaFile('../Dataset/Pengenalan Karakter/O.txt')
x = helper.bacaFile('../Dataset/Pengenalan Karakter/X.txt')

# Mengubah pola huruf menjadi list angka bipolar
bipolar_o = helper.polaToBipolar(o)
bipolar_x = helper.polaToBipolar(x)

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

# Membaca berkas testing
mirip_o = helper.bacaFile('../Dataset/Pengenalan Karakter/O_test.txt')
mirip_x = helper.bacaFile('../Dataset/Pengenalan Karakter/X_test.txt')

# Mengubah pola huruf menjadi list angka bipolar
bipolar_mirip_o = helper.polaToBipolar(mirip_o)
bipolar_mirip_x = helper.polaToBipolar(mirip_x)

# Proses testing
data_test = [bipolar_mirip_o, bipolar_mirip_x]
test = hebb.test(data_test)
print(test)
print('Hasil Hebb Learning:')
for t in test:
    print(target[t])