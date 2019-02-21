# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 11:03:08 2018

@author: MSI
"""

import numpy as np
import pandas as pd
import WildanML as ml

#Read Dataset
ds = pd.read_csv('../Dataset/mushrooms.csv')

#Convert categorical data to numerical data
dsNum = ml.catToNum(ds)

#Pisah data
data = np.array(dsNum)
h, w = data.shape
train = data[0:1000, 0:w]
test = data[1000:1100, 0:w]

print('=== Perhitungan KNN-Manhattan ====')
knnMan = ml.KNearestNeighbour(train, test, 11)
print('\nHasil\n', knnMan)
print(knnMan.shape)
print('\nBanyak jamur tidak beracun : ', list(knnMan[:, 0:1]).count(0))
print('Banyak jamur beracun       : ', list(knnMan[:, 0:1]).count(1))
print('\nAsli\n', test)
print(test.shape)
print('\nBanyak jamur tidak beracun : ', list(test[:, 0:1]).count(0))
print('Banyak jamur beracun       : ', list(test[:, 0:1]).count(1))
akurasi = 0
akurasi = np.where(knnMan[:,0:1] == test[:, 0:1], akurasi+1, akurasi+0)
akurasi = np.array(akurasi.flatten())
hasil = np.sum(akurasi)
print('Tingkat akurasi: ', hasil, '/', len(akurasi))

print('\n=== Perhitungan KNN-Euclidean ====')
knnEuc = ml.KNearestNeighbour(train, test, 11, distance='euclidean')
print('\nHasil\n', knnEuc)
print(knnEuc.shape)
print('\nBanyak jamur tidak beracun : ', list(knnEuc[:, 0:1]).count(0))
print('Banyak jamur beracun       : ', list(knnEuc[:, 0:1]).count(1))
print('\nAsli\n', test)
print(test.shape)
print('\nBanyak jamur tidak beracun : ', list(test[:, 0:1]).count(0))
print('Banyak jamur beracun       : ', list(test[:, 0:1]).count(1))
akurasi = 0
akurasi = np.where(knnEuc[:,0:1] == test[:, 0:1], akurasi+1, akurasi+0)
akurasi = np.array(akurasi.flatten())
hasil = np.sum(akurasi)
print('Tingkat akurasi: ', hasil, '/', len(akurasi))

#print(ds.head())
#print('-----------------------------------------------------------------------')
#print(ds.loc[1000:1100].head())
#print('-----------------------------------------------------------------------')
#print(dsNum.head())
#print('-----------------------------------------------------------------------')
#print(dsNum.loc[1000:1100].head())
#print('-----------------------------------------------------------------------')
#print(data)
#print('-----------------------------------------------------------------------')
#print(data[1000:1100, 0:w])
#print('-----------------------------------------------------------------------')
    
#========== Mushroom Classificaton ==========
#1 = Poisonous (p)
#0 = Edible/Eatable (e)