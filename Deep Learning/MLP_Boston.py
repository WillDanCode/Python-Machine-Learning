from WildanNN import MLPRegressor
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.metrics import mean_squared_error

print(15*'=' + ' Multi Layer Perceptron (Backpropagation) Pada Dataset Boston ' + 15*'=')

boston = load_boston()
data = pd.DataFrame(data=boston.data, columns=boston.feature_names)
data['MEDV'] = boston.target
print(data.head())

# Normalization
pt = PowerTransformer()
result = pt.fit_transform(data)
data = pd.DataFrame(data=result, columns=data.columns)
print(data.head())

x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.3)
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

n_input = x_train.shape[1] # 13
n_hidden = 20
n_output = 1

mlp = MLPRegressor(sizeLayer=(n_input, n_hidden, n_output), max_epoch=100, initialize_WeightBias=True)
mlp.train(x_train, y_train)
weightHidden, weightOutput = mlp.getWeight()
biasHidden, biasOutput = mlp.getBias()
print('Bobot Hidden: ', weightHidden.shape)
print('Bobot Output: ', weightOutput.shape)
print('Bias Hidden: ', biasHidden.shape)
print('Bias Output: ', biasOutput.shape)

# uji dengan menggunakan data latih, bandingkan hasilnya dengan target
y_pred = mlp.test(x_test)
print('Label Pred: ', y_pred.shape)
print('Label True: ', y_test.shape)
print('MSE:', mean_squared_error(y_test, y_pred))