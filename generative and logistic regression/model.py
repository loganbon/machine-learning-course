import numpy as np
import math
from time import perf_counter

def GenerativeModel(trainX, trainY, testX, testY):
    N = trainY.shape[0]
    N1 = np.sum(trainY)
    N2 = N - N1

    C1 = trainX[[True if i == 1 else False for i in trainY]]
    C2 = trainX[[True if i == 0 else False for i in trainY]]

    mu1, mu2 = np.average(C1, axis=0), np.average(C2, axis=0)

    S = ((trainX - mu1).T.dot(trainX - mu1) + (trainX - mu2).T.dot(trainX - mu2)) / N
    Sinv = np.linalg.inv(S)

    w = Sinv.dot(mu1 - mu2)

    w0 = -.5 * mu1.T.dot(Sinv).dot(mu1) + .5 * mu2.T.dot(Sinv).dot(mu2) + math.log(N1 / N2)

    a = w.dot(testX.T) + w0
    probs = 1 / (1 + np.exp(-a))
    pred = np.where(probs > 0.5, 1, 0)
    
    return np.sum(np.abs(testY - pred)) / testY.shape[0]


def BayesianLogisticRegression(trainX, trainY, testX, testY):

    trainX = np.hstack((np.ones((trainX.shape[0],1)), trainX))
    testX = np.hstack((np.ones((testX.shape[0],1)), testX))

    alpha = 0.1
    w = np.zeros(trainX.shape[1])
    
    y =  1 / (1 + np.exp(- w.dot(trainX.T)))
    R = np.diag(y * (1 - y))
    
    for i in range(100):

        wn = w - np.linalg.inv((alpha * np.identity(trainX.shape[1])) + trainX.T.dot(R).dot(trainX)).dot(trainX.T.dot(y - trainY) + (alpha * w))

        if w.dot(w) != 0:
            if (wn - w).dot(wn - w) / w.dot(w) < 10 ** -3:
                w = wn
                break
        w = wn

        y =  1 / (1 + np.exp(- w.dot(trainX.T)))
        R = np.diag(y * (1 - y))

    y =  1 / (1 + np.exp(- w.dot(testX.T)))
    R = np.diag(y * (1 - y))

    k = lambda x : np.sqrt(1 + (math.pi * x / 8))

    Sn = np.linalg.inv((np.identity(testX.shape[1]) * alpha) + testX.T.dot(R).dot(testX))

    ua = w.T.dot(testX.T)
    vara = testX.dot(Sn).dot(testX.T)

    pred = 1 / (1 + np.exp(-ua / k(np.diag(vara))))

    pred = np.where(pred > 0.5, 1, 0)

    return np.sum(np.absolute(pred - testY)) / testY.shape[0]


def BLRTimed(method, trainX, trainY, testX, testY):

    trainX = np.hstack((np.ones((trainX.shape[0],1)), trainX))
    testX = np.hstack((np.ones((testX.shape[0],1)), testX))
    alpha = 0.1

    w = np.zeros(trainX.shape[1])
    y =  1 / (1 + np.exp(- w.dot(trainX.T)))
    R = np.diag(y * (1 - y))

    t0 = perf_counter()
    times = [0]
    weights = [w]
    error = []
    
    if method == 'newton':
    
        for i in range(100):

            wn = w - np.linalg.inv((alpha * np.identity(trainX.shape[1])) + trainX.T.dot(R).dot(trainX)).dot(trainX.T.dot(y - trainY) + (alpha * w))
            times.append(perf_counter() - t0)
            weights.append(wn)
            if w.dot(w) != 0:
                if (wn - w).dot(wn - w) / w.dot(w) < 10 ** -3:
                    w = wn
                    break
            w = wn
            y =  1 / (1 + np.exp(- w.dot(trainX.T)))
            R = np.diag(y * (1 - y))

    elif method == 'gradient':
        for i in range(1,6001):
            wn = w - (10 ** (-3)) * (trainX.T.dot(y - trainY) + (alpha * w))
            if i % 10 == 0:
                times.append(perf_counter() - t0)
                weights.append(wn)

                if w.dot(w) != 0:
                    if (wn - w).dot(wn - w) / w.dot(w) < 10 ** -3:
                        w = wn
                        break
            w = wn
            y =  1 / (1 + np.exp(- w.dot(trainX.T)))


    for wi in weights:

        y =  1 / (1 + np.exp(- wi.dot(testX.T)))
        R = np.diag(y * (1 - y))

        k = lambda x : np.sqrt(1 + (math.pi * x / 8))

        Sn = np.linalg.inv((np.identity(testX.shape[1]) * alpha) + testX.T.dot(R).dot(testX))

        ua = wi.T.dot(testX.T)
        vara = testX.dot(Sn).dot(testX.T)

        pred = 1 / (1 + np.exp(-ua / k(np.diag(vara))))

        pred = np.where(pred > 0.5, 1, 0)

        error.append(np.sum(np.absolute(pred - testY)) / testY.shape[0])

    return error, times