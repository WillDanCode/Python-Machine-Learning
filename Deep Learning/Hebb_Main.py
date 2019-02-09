from WildanNeuralNetwork import Hebb
import numpy as np

print(15*'=' + ' Hebb Learning Pada Fungsi Logika ' + 15*'=')

hebb = Hebb(2)

# AND
data = np.array([[-1,-1],
                 [-1,1],
                 [1,-1],
                 [1,1]])
target = np.array([-1, -1, -1, 1])

# # OR
# data = np.array([[-1,-1],
#                  [-1,1],
#                  [1,-1],
#                  [1,1]])
# target = np.array([-1, 1, 1, 1])
#
# # XOR
# data = np.array([[-1,-1],
#                  [-1,1],
#                  [1,-1],
#                  [1,1]])
# target = np.array([-1, 1, 1, -1])

hebb.train(data,target)

weight, bias = hebb.getWeightBias()
print('Weight: ', weight)
print('Bias: ', bias)

test = hebb.test(data)
print('Hasil Hebb: ', test)
print('Target: ', target)