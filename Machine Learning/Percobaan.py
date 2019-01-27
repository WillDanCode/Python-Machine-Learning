# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 20:15:08 2018

@author: MSI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import MyMachineLearning as pp

matTrain = np.arange(60).reshape(10,6)
np.random.shuffle(matTrain)
matTrainAll = np.insert(matTrain, [0], [[1],[0],[1],[0],[1],[1],[0],[1],[0],[1]], axis=1)
print('\nMatTrainAll\n', matTrainAll)
print(matTrainAll.shape)
print('\nMatTrain\n', matTrain)
print(matTrain.shape)

matTest = np.arange(18).reshape(3,6)
np.random.shuffle(matTest)
print('\nMatTest\n', matTest)
print(matTest.shape)

hasil = []
for elTest in matTest:
    coba = []
    for elTrain in matTrain:
#        print('ElTrain ', elTrain)
#        print('ElTest ', elTest)
#        print('Kurang ', np.abs(elTrain-elTest))
        coba.append(pp.manhattanKNN(elTrain, elTest))
#        coba.append(pp.euclideanKNN(elTrain, elTest))
    hasil.append(coba)

hasil = np.array(hasil)
urut = np.sort(hasil)
ambil = urut[:, :3]
#indeks = ambil == hasil

print('\nHasil\n', hasil)
print(hasil.shape)
#print('\n', np.amin(hasil, axis=0)) #minimum kolom
#print('\n', np.amin(hasil, axis=1)) #minimum baris
#print('\nUrut\n', urut)
print('\nAmbil\n', ambil)
print(ambil.shape)

#hHasil, wHasil = hasil.shape
hAmbil, wAmbil = ambil.shape
indeks = []
for i in range(0,hAmbil):
    temp = []
    for j in range(0,wAmbil):
        idx = list(hasil[i]).index(ambil[i][j])
#        print(hasil[i], ambil[i][j])
        temp.append(idx)
        hasil[i][idx] = -1
    indeks.append(temp)
indeks = np.array(indeks)
print('\nIndeks\n',indeks)
print(indeks.shape)

kelas = []
for rowIndeks in indeks:
    kelas.append(matTrainAll[rowIndeks][:, 0:1].flatten())
kelas = np.array(kelas)
print('\nKelas\n',kelas)
print(kelas.shape)

vote = []
jum0, jum1 = 0, 0
#weighted
#for i in range(0, len(kelas)):
#    for j in range(0, len(kelas[0])):
#        if(kelas[i, j] == 0):
#            jum0 = jum0 + (1/np.power(ambil[i, j]))

#unweighted
for rowKelas in kelas:
    vote.append(max(list(rowKelas), key=list(rowKelas).count))

vote = np.array(vote)
print('\nVote\n',vote)
print(vote.shape)

matTest = np.insert(matTest, 0, vote, axis=1)
print('\nHasil\n',matTest)
print(matTest.shape)
#dTest = []
#for i in range(0, len(kelas)):
#    if i > 0:
#        matTest = np.delete(matTest, 0, 1)
#    matTest = np.insert(matTest, [0], kelas[i], axis=1)
#    dTest.append(matTest)
#    print('Datatest ke ', i+1, '\n', matTest)
#print('\n', np.array(dTest))

#a = np.array([1,2,3,4,3,5,4,1])
#aa = np.array([[1,2,3,4,3,5,4,1], [1,2,3,4,3,3,3,1]])
#b = np.array([1,3,3,1])
#c = np.array([])
#for elB in b:
#    indeks = list(a).index(elB)
#    c = np.append(c, indeks)
#    a[indeks] = -1
#print(c)
##print(list(aa[0]).index(5))