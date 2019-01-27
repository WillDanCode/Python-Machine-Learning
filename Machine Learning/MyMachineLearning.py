# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 12:07:26 2018

@author: MSI
"""

import numpy as np
import pandas as pd

#Ubah dataset categorical ke numeric
def catToNum(dataset):
    ds = dataset.copy()
    for col_name in ds.columns:
        if(ds[col_name].dtype == 'object'):
            ds[col_name] = ds[col_name].astype('category')
            ds[col_name] = ds[col_name].cat.codes
#            ds[col_name] = pd.Categorical.from_array(ds[col_name]).codes
    return ds

#Data dalam bentuk array 1D
def manhattanKNN(data1, data2):
#    train = dataTrain[1:len(dataTrain)]
    array1 = np.copy(data1)
    array2 = np.copy(data2)
    hasil = np.sum(np.abs(array1-array2))
    return hasil

#Data dalam bentuk array 1D
def euclideanKNN(data1, data2):
#    train = dataTrain[1:len(dataTrain)]
    array1 = np.copy(data1)
    array2 = np.copy(data2)
    hasil = np.sqrt(np.sum(np.power(array1-array2, 2)))
    return hasil

#Dataset dalam bentuk numpy array
def KNearestNeighbour(datasetTrain, datasetTest, k=5, types='unweighted', distance='manhattan'):
    h, w = datasetTrain.shape
    train = datasetTrain[0:h, 1:w] #asumsi kolom class ada di depan
    test = datasetTest[0:h, 1:w]
    
    print('\nTrain All\n', datasetTrain)
    print(datasetTrain.shape)
    print('\nTrain\n', train)
    print(train.shape)
    print('\nTest\n', test)
    print(test.shape)
    
    #Proses KNN
    #hitung jarak
    himpJarak = [] #kumpulan semua jarak data train dengan data test
    for dataTest in test:
        jarak = []
        for dataTrain in train:
            if (distance == 'euclidean'):
                jarak.append(euclideanKNN(dataTrain, dataTest))
            else:
                jarak.append(manhattanKNN(dataTrain, dataTest))
        himpJarak.append(jarak)
    himpJarak = np.array(himpJarak)
    print('\nHimpJarak\n', himpJarak)
    print(himpJarak.shape)
    
    #ambil jarak berdasarkan k
    ambilJarak = np.sort(himpJarak)[:, :k] #ambil jarak sebanyak k
    hAmbil, wAmbil = ambilJarak.shape
    print('\nAmbilJarak\n', ambilJarak)
    print(ambilJarak.shape)
    
    #nilai indeks jarak yang sudah disortir k pada data train
    indeks = []
    for i in range(0,hAmbil):
        temp = []
        for j in range(0,wAmbil):
            idx = list(himpJarak[i]).index(ambilJarak[i][j])
            temp.append(idx)
            himpJarak[i][j] = -1
        indeks.append(temp)
    indeks = np.array(indeks)
    print('\nIndeks\n', indeks)
    print(indeks.shape)
    
    #mengetahui kelas berdasarkan indeks di atas
    kelas = []
    for rowIndeks in indeks:
        kelas.append(datasetTrain[rowIndeks][:, 0:1].flatten())
    kelas = np.array(kelas)
    print('\nKelas\n', kelas)
    print(kelas.shape)
    
    #melakukan voting kelas terbanyak
    vote = []
    if (types == 'weighted'):
        #kodenya menyusul...
        
        vote = np.array(vote)
        print('\nVote\n',vote)
        print(vote.shape)
    else:
        for rowKelas in kelas:
            vote.append(max(list(rowKelas), key=list(rowKelas).count))
        vote = np.array(vote)
        print('\nVote\n',vote)
        print(vote.shape)
    
    hasil = np.insert(test, 0, vote, axis=1)
    return hasil