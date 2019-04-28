from WildanNN import LVQ
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print(15*'=' + ' Learning Vector Quantization Pada Dataset Iris ' + 15*'=')

data = pd.read_csv('../Dataset/iris.csv')
del data['Id']
# print(data.head())

x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.3)
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

n_input = x_train.shape[1]
n_output = len(np.unique(y_train))

lvq = LVQ(sizeInput=n_input, sizeOutput=n_output, max_epoch=5, version='1')
bobot_dan_label = lvq.train(x_train, y_train)
bobot = lvq.getWeight()
print('Bobot: ', bobot)

#uji dengan menggunakan data latih, bandingkan hasilnya dengan target
y_pred = lvq.test(np.array(x_test), bobot_dan_label)
# print('Label Pred: ', y_pred)
# print('Label True: ', y_test)

print('Accuracy:', accuracy_score(y_test, y_pred))