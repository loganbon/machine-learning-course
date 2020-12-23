import math, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.stattools import adfuller

class TimeSeries:
    
    def __init__(self, data, date_col_name, col_name):
        self.series = data
        self.col_name = col_name

    def summaryPlots(self, lags=10, save_loc=''):
        _, layout = plt.figure(figsize=(14, 14)), (2, 2)
        ts_ax, h_ax = plt.subplot2grid(layout, (0, 0)), plt.subplot2grid(layout, (0, 1))
        ac_ax, pac_ax = plt.subplot2grid(layout, (1, 0)), plt.subplot2grid(layout, (1, 1))
        
        self.series.plot(ax=ts_ax)
        ts_ax.set_title(self.col_name)
        self.series.plot(ax=h_ax, kind='hist', bins=25)
        h_ax.set_title('Histogram')
        plot_acf(self.series, lags=lags, ax=ac_ax)
        plot_pacf(self.series, lags=lags, ax=pac_ax)

        plt.savefig(save_loc)
        
    def adfTest(self):
        result = adfuller(self.series)
        print('ADF Statistic: {}'.format(result[0]))
        print('p-value: {}'.format(result[1]))
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))

    def to_frame(self):
        return pd.DataFrame(self.series)

class AR:
    
    def __init__(self, data):
        self.data = data
        self.w = None

    def fit(self, lags, trainPct, holdoutPct):

        n, m = self.data.shape[0], self.data.shape[1]
        
        split_idx = int(n * (1-holdoutPct))

        
        X = pd.DataFrame(np.ones(n))
        X.set_index(self.data.index, inplace=True)
        for i in range(lags):
            X[i+1] = self.data.shift(i+1)
        X = X.iloc[lags:-lags]

        Y = self.data.iloc[lags:-lags]

        sample = np.random.choice(range(split_idx), size=int(split_idx*trainPct), replace=False)
        
        trainX, testX = X.iloc[sample].to_numpy(), X[split_idx:].to_numpy()
        trainY, testY = Y.iloc[sample].to_numpy(), Y[split_idx:].to_numpy()

        self.w = np.linalg.inv(trainX.T.dot(trainX)).dot(trainX.T.dot(trainY))

        pred = testX.dot(self.w)
        mse = np.sum((pred - testY) ** 2) / len(testY)

        return mse


class BAR:

    def __init__(self, data):
        self.data = data
        self.w = None

    def fit(self, lags, trainPct, holdoutPct, lambdaParam=None):
        alpha, beta = random.randint(1,10), random.randint(1, 10)

        n, _ = self.data.shape[0], self.data.shape[1]
        
        split_idx = int(n * (1-holdoutPct))
        
        X = pd.DataFrame(np.ones(n))
        X.set_index(self.data.index, inplace=True)
        for i in range(lags):
            X[i+1] = self.data.shift(i+1)
        X = X.iloc[lags:-lags]

        Y = self.data.iloc[lags:-lags]

        if trainPct < 1:
            sample = np.random.choice(range(split_idx), size=int(split_idx*trainPct), replace=False)
            trainX, trainY = X.iloc[sample].to_numpy(), Y.iloc[sample].to_numpy()
        else:
            trainX, trainY = X[:split_idx].to_numpy(), Y[:split_idx].to_numpy()
        
        
        testX, testY = X[split_idx:].to_numpy(), Y[split_idx:].to_numpy()

        if lambdaParam == None:
            aErr, bErr = math.inf, math.inf
            while aErr > 10 ** -5 and bErr > 10 ** -5:

                eigVals, _ = np.linalg.eig(beta * (trainX.T.dot(trainX)))

                gamma = np.sum(eigVals / (eigVals + alpha))

                SN = np.linalg.inv((alpha * np.identity(trainX.shape[1])) + (beta * (trainX.T.dot(trainX))))
                mN = beta * (SN.dot(trainX.T).dot(trainY))

                alpha0 = gamma / mN.T.dot(mN)
                beta0 =  ((1 / (len(trainX) - gamma)) * (np.sum((trainY - trainX.dot(mN)) ** 2))) ** -1

                aErr, bErr = abs(alpha0 - alpha), abs(beta0 - beta)
                alpha, beta = alpha0, beta0
        
        lambdaParam = alpha / beta
        w = np.linalg.inv((lambdaParam * np.identity(trainX.shape[1])) + trainX.T.dot(trainX)).dot(trainX.T.dot(trainY))

        mse = np.sum((testX.dot(w) - testY) ** 2) / len(testY)

        return mse