import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from WildanNN import SOM

print(15*'=' + ' Self Organizing Map Pada Dataset Iris ' + 15*'=')

iris = load_iris()
n_input = iris.data.shape[1]
n_output = 3

som = SOM(sizeInput=n_input, sizeOutput=n_output, max_epoch=10, ordering=2, radius=5)
som.train(iris.data)
bobot = som.getWeight()
print('Bobot: ', bobot)

# uji dengan menggunakan dataset, bandingkan hasilnya dengan target
y_true = iris.target
y_pred = som.test(iris.data)
print('Label True: ', y_true)
print('Label Pred: ', y_pred)
print('Accuracy: ', accuracy_score(y_true, y_pred))
coba = np.array([[[1,2,3,4], [1,1,1,1], [0,0,0,0]],
                 [[5,6,7,8], [2,2,2,2], [0,0,0,0]],
                 [[9,1,3,5], [3,3,3,3], [0,0,0,0]]])
# coba = np.array([[[1,2,3,4], [5,6,7,8]],
#                  [[9,0,1,2], [3,4,5,6]]])
# coba = np.zeros((4,3))
print(coba)
print('J1 : ', coba[0,0,:])
print('J2 : ', coba[0,1,:])
print('J3 : ', coba[1,0,:])
print('J4 : ', coba[1,1,:])
# print(coba.reshape((2,2,3)))
# print(coba[1,:,:] & coba[:,1,:])
# print(np.sqrt(4) ** 2 == 4)
# print(np.zeros((2,2,4)))
print(np.binary_repr(5)[1])