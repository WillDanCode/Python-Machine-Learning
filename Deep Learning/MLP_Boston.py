from WildanNN import MLPRegressor
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print(15*'=' + ' Multi Layer Perceptron (Backpropagation) Pada Dataset Boston ' + 15*'=')

boston = load_boston()
data = pd.DataFrame(data=boston.data, columns=boston.feature_names)
data['MEDV'] = boston.target
print(data.head())

x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.3)
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

n_input = x_train.shape[1] # 4
n_hidden = 10
n_output = 1

mlp = MLPRegressor(sizeLayer=(n_input, n_hidden, n_output), max_epoch=2)
mlp.train(x_train, y_train)
bobot = mlp.getWeight()
print('Bobot: ', bobot)

# #uji dengan menggunakan data latih, bandingkan hasilnya dengan target
# y_pred = lvq.test(np.array(x_test), bobot_dan_label)
# print('Label Pred: ', y_pred)
# print('Label True: ', y_test)
# print('Accuracy:', accuracy_score(y_test, y_pred))