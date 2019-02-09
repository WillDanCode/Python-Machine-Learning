from WildanNeuralNetwork import SLP
import numpy as np

print(15*'=' + ' Single Layer Perceptron Pada Fungsi Logika ' + 15*'=')

# AND
data = np.array([[1,1],
                 [1,0],
                 [0,1],
                 [0,0]])
target = np.array([1, -1, -1, -1])

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

slp = SLP(2)
slp.train(data,target)

bobot,bias = slp.getWeightBias()
print('Bobot: ', bobot)
print('Bias: ', bias)

#uji dengan menggunakan data latih, bandingkan hasilnya dengan target
test = slp.test(data)
print('Hasil SLP: ', test)
print('Target: ', target)
