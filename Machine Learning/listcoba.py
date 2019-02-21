import numpy as np

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
    x = np.array([1,4,6])
    y = np.array([1.5709, 1.5727, 1.5751])
    preX = 3.5

    p = interpolasi(x, y, preX)
    print(p)
    print(lagrange(x, preX))
    # print(np.prod(preX - x[0+1:]))
    # print(np.poly1d([1,2,3]))