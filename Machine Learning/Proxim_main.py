from WildanML import ProximityMeasure
import numpy as np
import pandas as pd

print(20*'=' + ' Perhitungan Proximity Measure ' + 20*'=' + '\n')

# Membaca dataset
"""
Terdapat 25 data train & 25 data test
Data train: data ke 1-25
Data test: data ke 1-25
"""
df = pd.read_csv('../Dataset/heart.csv')
train = df.loc[:24, :]
test = df.loc[:24, :]
# print(train.info(), '\n')

# Spesifikasi jenis atribut
numerical = ('age', 'trestbps', 'chol', 'thalach', 'oldpeak')
categorical = ('cp', 'restecg', 'slope', 'ca', 'thal')
binary = ('sex', 'fbs', 'exang')

# Pembagian dataset berdasarkan jenis atribut
trainNum = train.loc[:, numerical]
trainCat = train.loc[:, categorical]
trainBin = train.loc[:, binary]
testNum = test.loc[:, numerical]
testCat = test.loc[:, categorical]
testBin = test.loc[:, binary]
# print(trainBin.head(), '\n')
# print(testBin[:5], '\n')
# print(trainNum.shape[1])

# Perhitungan dissimilarity
pm = ProximityMeasure()
dissimMan = pm.dissimNum(trainNum, testNum)
dissimEuc = pm.dissimNum(trainNum, testNum, 'euclidean')
dissimMink = pm.dissimNum(trainNum, testNum, 'minkowski', trainNum.shape[1])
dissimSup = pm.dissimNum(trainNum, testNum, 'supremum')
dissimCat = pm.dissimCat(trainCat, testCat)
dissimBin = pm.dissimBin(trainBin, testBin)
print('\nDissimilarity Atribut Numerik (Manhattan)\n', dissimMan)
print('\nDissimilarity Atribut Numerik (Euclidean)\n', dissimEuc)
print('\nDissimilarity Atribut Numerik (Minkowski)\n', dissimMink)
print('\nDissimilarity Atribut Numerik (Supremum)\n', dissimSup)
print('\nDissimilarity Atribut Categorical\n', dissimCat)
print('\nDissimilarity Atribut Binary\n', dissimBin)

# Pengujian proximity measure
print('\nProximity Measure Atribut Numerik (Manhattan)\n')
pm.showProximity(dissimMan)
print('\nProximity Measure Atribut Numerik (Euclidean)\n')
pm.showProximity(dissimEuc)
print('\nProximity Measure Atribut Numerik (Minkowski)\n')
pm.showProximity(dissimMink)
print('\nProximity Measure Atribut Numerik (Supremum)\n')
pm.showProximity(dissimSup)
print('\nProximity Measure Atribut Categorical\n')
pm.showProximity(dissimCat)
print('\nProximity Measure Atribut Binary\n')
pm.showProximity(dissimBin)