from WildanNN import Adaline
import numpy as np

print(15*'=' + ' Delta Rule Pada Fungsi Logika ' + 15*'=')

# AND
data = np.array([[1,1],
                 [1,-1],
                 [-1,1],
                 [-1,-1]])
target = np.array([1, -1, -1, -1])

# # OR
# data = np.array([[1,1],
#                  [1,-1],
#                  [-1,1],
#                  [-1,-1]])
# target = np.array([1, 1, 1, -1])
#
# # XOR
# data = np.array([[1,1],
#                  [1,-1],
#                  [-1,1],
#                  [-1,-1]])
# target = np.array([-1, 1, 1, -1])

adaline = Adaline(2,1)
adaline.train(data,target)
weight, bias = adaline.getWeightBias()
print('Weight:', weight)
print('Bias:', bias)

test = adaline.test(data)
print('Hasil Adaline:', test)
print('Target:', target)