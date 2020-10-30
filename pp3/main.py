import numpy as np
import matplotlib.pyplot as plt
import model

print('Task 1)')

files = ['A', 'B', 'USPS']
trainSizes = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]

for fileName in files:

    dataX = np.genfromtxt('pp3data/' + fileName + '.csv', delimiter=',')
    dataY = np.genfromtxt('pp3data/labels-' + fileName + '.csv')

    sampleGenError = None
    sampleBayesError = None

    N = dataX.shape[0]
    N1 = int(2*N/3)
    N2 = N - N1

    for i in range(30):

        itrGenError = []
        itrBayesError = []

        for size in trainSizes:

            N1_sub = int(N1 * size)

            mask1 = np.hstack((np.ones(N1), np.zeros(N2)))
            np.random.shuffle(mask1)
            mask1 = mask1.astype(bool)

            mask2 = np.hstack((np.ones(N1_sub), np.zeros(N1 - N1_sub)))
            np.random.shuffle(mask2)
            mask2 = mask2.astype(bool)
            
            trainX, trainY = dataX[mask1][mask2], dataY[mask1][mask2]
            testX, testY = dataX[~mask1], dataY[~mask1]

            genError = model.GenerativeModel(trainX, trainY, testX, testY)
            itrGenError.append(genError)

            bayesError = model.BayesianLogisticRegression(trainX, trainY, testX, testY)
            itrBayesError.append(bayesError)


        if i:
            sampleGenError = np.vstack((sampleGenError, itrGenError))
            sampleBayesError = np.vstack((sampleBayesError, itrBayesError))

        else:
            sampleGenError = itrGenError
            sampleBayesError = itrBayesError


    sampleGenStd = np.std(sampleGenError, axis=0)
    sampleBayesStd = np.std(sampleBayesError, axis=0)

    plt.plot(trainSizes, np.average(sampleGenError, axis=0), color='blue')
    plt.errorbar(trainSizes, np.average(sampleGenError, axis=0), yerr = sampleGenStd, fmt ='o', color='green') 

    plt.ylabel('Error Rate')
    plt.xlabel('Training Percent')
    plt.title('Generative Model: Training Size vs. Error Rate')
    plt.savefig('output/generative-' + fileName)
    plt.clf()

    plt.plot(trainSizes, np.average(sampleBayesError, axis=0), color='blue')
    plt.errorbar(trainSizes, np.average(sampleBayesError, axis=0), yerr = sampleBayesStd, fmt ='o', color='green') 

    plt.ylabel('Error Rate')
    plt.xlabel('Training Percent')
    plt.title('Bayes Model: Training Size vs. Error Rate')
    plt.savefig('output/bayes-' + fileName)
    plt.clf()

print('\nTask 2)\n')