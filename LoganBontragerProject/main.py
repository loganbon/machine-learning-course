import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from model import TimeSeries
from util import setKey, getData
from model import AR, BAR

### File Variables ###
symbols = ['AMZN','F','NFLX','GOOG']
sym = 'AMZN' # series used for univariate analysis
refresh_data = False
years = 8

### Data Refresh ###
if refresh_data:
    setKey('key.txt')
    getData(symbols)

### Data Preperation ###
df = pd.read_csv('data/{}.csv'.format(sym)).head(365 * years)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index(['Date'], drop=True, inplace=True)

series = TimeSeries(df, 'Date', sym)
transformed_series = TimeSeries(pd.Series(np.log(df[sym] / df[sym].shift())).dropna(), 'Date', sym)

### Summary Plot ###
print('\nGenerating Summary Plots...')
autocorr_lags = 50
series.summaryPlots(lags=autocorr_lags, save_loc='output/{}_summary.png'.format(sym))
transformed_series.summaryPlots(lags=autocorr_lags, save_loc='output/{}_trans_summary.png'.format(sym))


### Augmented Dickey Fuller Test ###
print('\nRunning Augmented Dickey Fuller Tests...')
print('(Pre-Transformation)')
series.adfTest()

print('\n(Post-Transformation)')
transformed_series.adfTest()


### Normality Test ###
from statsmodels.graphics.gofplots import qqplot
qqplot(transformed_series.to_frame().to_numpy(), line='s')
plt.savefig('output/{}_qqplot.png'.format(sym))
plt.clf()


print('\nRunning Lag Size Tests...')
berrors = []
aerrors = []
for i in range(30):

    bvarErr, varErr = [], []
    test_pct = 0.03
    lags = range(1,30,3)

    for l in lags:

        model = BAR(transformed_series.to_frame())
        bvar_mse = model.fit(l, 1, test_pct)
        bvarErr.append(bvar_mse)

        model = AR(transformed_series.to_frame())
        var_mse = model.fit(l, 1, test_pct)
        varErr.append(var_mse)

        berrors.append(bvarErr)
        aerrors.append(varErr)

plt.plot(lags, np.sum(np.array(berrors), axis=0)/30, label='BAR')
plt.plot(lags, np.sum(np.array(aerrors), axis=0)/30, label='AR')
plt.legend()
plt.title('Diff MSE vs. Lag Size')
plt.savefig('output/{}_lag_size.png'.format(sym))
plt.clf()

print('\nRunning Training Size Tests...')
berrors = []
aerrors = []

for i in range(30):
    bvarErr, varErr = [], []
    test_pct = 0.03
    trainingSizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    for size in trainingSizes:

        model = BAR(transformed_series.to_frame())
        bvar_mse = model.fit(10, size, test_pct)
        bvarErr.append(bvar_mse)

        model = AR(transformed_series.to_frame())
        var_mse = model.fit(10, size, test_pct)
        varErr.append(var_mse)

    berrors.append(bvarErr)
    aerrors.append(varErr)

plt.plot(trainingSizes, np.sum(np.array(berrors), axis=0)/30, label='BAR')
plt.plot(trainingSizes, np.sum(np.array(aerrors), axis=0)/30, label='AR')
plt.legend()
plt.title('Diff MSE vs. Training Size')
plt.savefig('output/{}_training_size.png'.format(sym))