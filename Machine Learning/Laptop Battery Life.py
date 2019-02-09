import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def regresiLinier(fiturX, fiturY, jumlahData, prediksiX):
    sigmaX = np.sum(fiturX)
    sigmaY = np.sum(fiturY)
    sigmaXY = np.sum(fiturX*fiturY)
    sigmaX2 = np.sum(fiturX**2)

    # y = b0 + b1x
    b1 = (jumlahData*sigmaXY - sigmaX*sigmaY) / (jumlahData*sigmaX2 - sigmaX**2)
    b0 = (sigmaY - b1*sigmaX) / jumlahData
    prediksiY = b0 + b1*prediksiX
    return prediksiY

def lagrange(x, prediksiX):
    l = []
    for i, data in enumerate(x):
        if i == 0:
            temp = np.prod((prediksiX - x[i+1:])) / np.prod((x[i] - x[i+1:]))
        else:
            temp = np.prod((prediksiX - np.append(x[i+1:], x[:i]))) / np.prod((x[i] - np.append(x[i+1:], x[:i])))
        l.append(temp)
    # l = np.prod(prediksiX - x) / np.prod(np.append(x[1:], x[0]) - x)
    return l

def interpolasi(x, y, prediksiX):
    p = np.sum(y * lagrange(x, prediksiX))
    return p


if __name__ == "__main__":
    traindata = open("../Dataset/trainingdata.txt", "r")
    fiturX, fiturY = [], []
    jumlahdata = 0
    for data in traindata:
        x, y = data.split(",")
        fiturX.append(float(x))
        fiturY.append(float(y))
        jumlahdata += 1

    fiturX = np.array(fiturX)
    fiturY = np.array(fiturY)
    #
    prediksiY = regresiLinier(fiturX, fiturY, jumlahdata, 1.5)
    print(prediksiY)

    print(interpolasi(fiturX, fiturY, 1.5))

    # prediksiY = interpolasiLagrange(fiturX, fiturY, 1.5)
    # plt.figure(1)
    # plt.plot(fiturX, fiturY, ".")
    # plt.plot(fiturX, regresiLinier(fiturX, fiturY, jumlahdata, fiturX), "r")
    #
    # plt.figure(2)
    # plt.plot(fiturX, fiturY, ".")
    # plt.plot(fiturX, interpolasi(fiturX, fiturY, fiturX), "r")
    # plt.show()