from WildanNN import MLPRegressor, Layer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.metrics import mean_squared_error
from sklearn import neural_network as nn

print(15*'=' + ' Multi Layer Perceptron (Backpropagation) Pada Dataset Forest Fire ' + 15*'=')

data = pd.read_csv('./Dataset/forestfires.csv')
data.drop(columns=['X', 'Y', 'month', 'day', 'rain'], inplace=True)
print(data.head())

x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2)

# Normalization
# Ref : https://stackoverflow.com/questions/38058774/scikit-learn-how-to-scale-back-the-y-predicted-result
mms = MinMaxScaler()
scalerx = mms.fit(x_train)
scalery = mms.fit(y_train[:, None])
x_train = scalerx.transform(x_train)
x_test = scalerx.transform(x_test)
y_train = scalery.transform(y_train[:, None])
y_test = scalery.transform(y_test[:, None])

n_input = x_train.shape[1] # 7
n_hidden1 = 10
n_hidden2 = 7
n_output = 1

hidden1 = Layer(n_hidden1, input_dim=n_input)
hidden2 = Layer(n_hidden2, input_dim=n_hidden1)
output = Layer(n_output, input_dim=n_hidden2)
layers = [hidden1, hidden2, output]

mlp = MLPRegressor(layers=layers, max_epoch=100, alpha=0.01)
mlp.train(x_train, y_train, optimizer='lm')
# weightHidden, weightOutput = mlp.getWeight()
# biasHidden, biasOutput = mlp.getBias()
# print('Bobot Hidden: ', weightHidden.shape)
# print('Bobot Output: ', weightOutput.shape)
# print('Bias Hidden: ', biasHidden.shape)
# print('Bias Output: ', biasOutput.shape)

# uji dengan menggunakan data latih, bandingkan hasilnya dengan target
y_pred = mlp.test(x_test)
y_pred = scalery.inverse_transform(y_pred)
y_test = scalery.inverse_transform(y_test)
print('Label Pred: ', y_pred.shape)
print('Label True: ', y_test.shape)
print('MSE:', mean_squared_error(y_test, y_pred))

# Benchmark
# mlpr = nn.MLPRegressor(hidden_layer_sizes=(n_hidden,1), alpha=0.1, max_iter=1000)
# mlpr.fit(x_train, y_train.ravel())
# pred = mlpr.predict(x_test)
# pred = scalery.inverse_transform(pred[:, None])
# print('MSE Sklearn:', mean_squared_error(y_test, pred))