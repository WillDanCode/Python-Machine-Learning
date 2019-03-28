import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def kmeans():
    pass

data = pd.read_csv('dataKmeans.txt')
x = np.array(data['Fitur x'])
y = np.array(data['Fitur y'])
label = np.array(data['Kelompok'])

print(data)
print(data['Kelompok'].value_counts()) #hitung jumlah masing2 kelompok
print(data[data['Kelompok'] == 1].sum())

# plt.scatter(x,y,c=label)
# plt.show()