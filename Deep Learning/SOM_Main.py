from WildanNN import SOM
import numpy as np

print(15*'=' + ' Self Organizing Map Pada Fungsi Logika ' + 15*'=')

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
                 [1,0,0,0],
                 [0,0,1,1]])

n_input = len(data[0])
n_output = 2

som = SOM(sizeInput=n_input, sizeOutput=n_output, max_epoch=5, ordering=2, radius=0)
som.train(data)
bobot = som.getWeight()
print('Bobot: ', bobot)

# uji dengan menggunakan dataset
label = som.test(data)
print('Hasil SOM: ', label)