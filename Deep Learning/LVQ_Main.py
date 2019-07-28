from WildanNN import LVQ
import numpy as np

print(15*'=' + ' Learning Vector Quantization Pada Fungsi Logika ' + 15*'=')

# # AND
# data = np.array([[1,1],
#                  [1,0],
#                  [0,1],
#                  [0,0]])
# target = np.array([1, -1, -1, -1])
#
# # OR
# data = np.array([[1,1],
#                  [1,0],
#                  [0,1],
#                  [0,0]])
# target = np.array([1, 1, 1, -1])
#
# # XOR
# data = np.array([[1,1],
#                  [1,0],
#                  [0,1],
#                  [0,0]])
# target = np.array([-1, 1, 1, -1])

# Dummy
data = np.array([[1,1,0,0],
                 [0,0,0,1],
                 [0,0,1,1],
                 [1,0,0,0],
                 [0,1,1,0]])
target = np.array([1, 1, 2, 1, 2])

n_input = len(data[0])
n_output = len(np.unique(target))

lvq = LVQ(sizeInput=n_input, sizeOutput=n_output, max_epoch=5, version='1')
bobot_dan_label = lvq.train(data, target)
bobot = lvq.getWeight()
print('Bobot: ', bobot)

#uji dengan menggunakan data latih, bandingkan hasilnya dengan target
test = lvq.test(data, bobot_dan_label)
print('Hasil LVQ: ', test)
print('Target: ', target)