import numpy as np
import matplotlib.pyplot as plt
import random
import math

def LinearRegression(trainX, trainY, testX, testY, trainFraction=1):

    n = int(trainX.shape[0] * trainFraction)
    dataX = trainX[:n,:]
    dataY = trainY[:n]

    w = np.linalg.inv(dataX.T.dot(dataX)).dot(dataX.T.dot(dataY))

    mse = np.sum((testX.dot(w) - testY) ** 2) / n

    return mse

def BayesianLR(trainX, trainY, testX, testY, lambdaParam=None, trainFraction=1):
    alpha, beta = random.randint(1,10), random.randint(1, 10)

    n = int(trainX.shape[0] * trainFraction)
    dataX = trainX[:n,:]
    dataY = trainY[:n]

    if lambdaParam == None:
        aErr, bErr = math.inf, math.inf
        while aErr > 10 ** -5 and bErr > 10 ** -5:

            eigVals, _ = np.linalg.eig(beta * (dataX.T.dot(dataX)))

            gamma = np.sum(eigVals / (eigVals + alpha))

            SN = np.linalg.inv((alpha * np.identity(dataX.shape[1])) + (beta * (dataX.T.dot(dataX))))
            mN = beta * (SN.dot(dataX.T).dot(dataY))

            alpha0 = gamma / mN.T.dot(mN)
            beta0 =  ((1 / (n - gamma)) * (np.sum((dataY - dataX.dot(mN)) ** 2))) ** -1

            aErr, bErr = abs(alpha0 - alpha), abs(beta0 - beta)
            alpha, beta = alpha0, beta0
        lambdaParam = alpha / beta

    w = np.linalg.inv((lambdaParam * np.identity(dataX.shape[1])) + dataX.T.dot(dataX)).dot(dataX.T.dot(dataY))

    mse = np.sum((testX.dot(w) - testY) ** 2) / n

    return alpha, beta, mse

def PolyRegression(trainX, trainY, testX, testY, degree):
    n = trainX.shape[0]
    dataX = np.zeros((degree + 1, n))
    testData = np.zeros((degree + 1, testX.shape[0]))
    for i in range(degree + 1):
        dataX[i] = trainX ** i
        testData[i] = testX ** i
    dataX = dataX.T
    testData = testData.T

    w = np.linalg.inv(dataX.T.dot(dataX)).dot(dataX.T.dot(trainY))

    mse = np.sum((testData.dot(w) - testY) ** 2) / n

    return mse

def BayesianPolyReg(trainX, trainY, testX, testY, degree):
    alpha, beta = random.randint(1,10), random.randint(1, 10)
    
    n = trainX.shape[0]
    dataX = np.zeros((degree + 1, n))
    testData = np.zeros((degree + 1, testX.shape[0]))
    for i in range(degree + 1):
        dataX[i] = trainX ** i
        testData[i] = testX ** i
    dataX = dataX.T
    testData = testData.T

    aErr, bErr = math.inf, math.inf
    while aErr > 10 ** -5 and bErr > 10 ** -5:

        eigVals, _ = np.linalg.eig(beta * (dataX.T.dot(dataX)))

        gamma = np.sum(eigVals / (eigVals + alpha))

        SN = np.linalg.inv((alpha * np.identity(dataX.shape[1])) + (beta * (dataX.T.dot(dataX))))
        print(dataX.shape)
        print(testData.shape)
        mN = beta * (SN.dot(dataX.T.dot(testData)))

        alpha0 = gamma / mN.T.dot(mN)
        beta0 =  ((1 / (n - gamma)) * (np.sum((testData - dataX.dot(mN)) ** 2))) ** -1

        aErr, bErr = abs(alpha0 - alpha), abs(beta0 - beta)
        alpha, beta = alpha0, beta0
    lambdaParam = alpha / beta

    A = (alpha * np.identity(dataX.shape[1])) + (beta * (dataX.T.dot(dataX)))
    EmN = ((beta / 2) * np.sum((trainY - dataX.dot(mN)) ** 2)) + ((alpha / 2) * mN.T.dot(mN))
    N, M = dataX.shape[0], dataX.shape[1]

    logEvid = (M / 2 * math.log(alpha)) + (N / 2 * math.log(beta)) - EmN - (math.log(np.linalg.det(A))) - (N / 2 * math.log(2 * math.pi))

    w = np.linalg.inv((lambdaParam * np.identity(dataX.shape[1])) + dataX.T.dot(dataX)).dot(dataX.T.dot(testY))
    mse = np.sum((testData.dot(w) - testY) ** 2) / n

    return logEvid, mse

'''
### TASK 1
print('Task 1)\n')
fileNames = ['artificial', 'crime']
fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


# data creation
data = []
for i in range(len(fileNames)):
    trainX = np.genfromtxt('pp2data/train-' + fileNames[i] + '.csv', delimiter=',')
    trainY = np.genfromtxt('pp2data/trainR-' + fileNames[i] + '.csv', delimiter=',')
    testX = np.genfromtxt('pp2data/test-' + fileNames[i] + '.csv', delimiter=',')
    testY = np.genfromtxt('pp2data/testR-' + fileNames[i] + '.csv', delimiter=',')
    data.append([trainX, trainY, testX, testY])

print('(i)')
for i in range(len(fileNames)):
    print('File:', fileNames[i])

    for frac in fractions:
        alpha, beta, mse = BayesianLR(data[i][0], data[i][1], data[i][2], data[i][3], trainFraction=frac)
        print('alpha:', alpha, 'beta:', beta, 'lambda:', alpha / beta)
    print()

print('(ii)')

for i in range(len(fileNames)):
    print('File:', fileNames[i])

    linearErrors = []
    bayesErrors = []

    for frac in fractions:
        linearErrors.append(LinearRegression(data[i][0], data[i][1], data[i][2], data[i][3], trainFraction=frac))
        bayesErrors.append(BayesianLR(data[i][0], data[i][1], data[i][2], data[i][3], trainFraction=frac)[2])

    print(linearErrors)
    print(bayesErrors)

    plt.plot(fractions, linearErrors, color='red', label='Linear Regression')
    plt.plot(fractions, bayesErrors, color='blue', label='Bayesian LR')

    plt.xlabel('Training Size')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.title(fileNames[i] + ': Training Size vs. Test Set MSE')
    plt.legend()
    plt.savefig('output/task1ii' + fileNames[i] + '.png')
    plt.clf()
    print()

print('(iii)')
lambdas = [1, 33, 1000]

for i in range(len(fileNames)):
    print('File:', fileNames[i])

    errors = np.zeros((len(lambdas), len(fractions)))
    for j in range(len(lambdas)):
        for k in range(len(fractions)):
            _, _, mse = BayesianLR(data[i][0], data[i][1], data[i][2], data[i][3], lambdaParam=lambdas[j], trainFraction=fractions[k])
            errors[j][k] = mse
        
    colors = ['blue','red','orange']
    for m in range(len(errors)):
        plt.plot(fractions, errors[m], color=colors[m], label='λ = ' + str(lambdas[m]))

    plt.xlabel('Training Size')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.title(fileNames[i] + ': Training Size vs. Test Set MSE')
    plt.legend()
    plt.savefig('output/task1iii' + fileNames[i] + '.png')
    plt.clf()
'''
### TASK 2
print('Task 2)\n')

fileNames = ['f3', 'f5']
degrees = list(range(1, 11))

# data creation
data = []
for i in range(len(fileNames)):
    trainX = np.genfromtxt('pp2data/train-' + fileNames[i] + '.csv', delimiter=',')
    trainY = np.genfromtxt('pp2data/trainR-' + fileNames[i] + '.csv', delimiter=',')
    testX = np.genfromtxt('pp2data/test-' + fileNames[i] + '.csv', delimiter=',')
    testY = np.genfromtxt('pp2data/testR-' + fileNames[i] + '.csv', delimiter=',')
    data.append([trainX, trainY, testX, testY])

for i in range(len(fileNames)):
    print('File:', fileNames[i])

    degreeErrors = [[] for i in range(3)]

    for j in range(len(degrees)):
        degreeErrors[0].append(PolyRegression(data[i][0], data[i][1], data[i][2], data[i][3], degrees[j]))
        logEvidence, mse = BayesianPolyReg(data[i][0], data[i][1], data[i][2], data[i][3], degrees[j])
        degreeErrors[1].append(mse)
        degreeErrors[2].append(logEvidence)

    plt.plot(degrees, degreeErrors[0], color='red', label='Poly Regression')
    plt.plot(degrees, degreeErrors[1], color='blue', label='Bayes Poly MSE')
    plt.plot(degrees, degreeErrors[2], color='orange', label='Bayes Poly Log Evidence')
    plt.xlabel('Degree')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.title(fileNames[i] + ': Degree vs. Test Set MSE')
    plt.legend()
    plt.savefig('output/task2' + fileNames[i] + '.png')
    plt.clf()

    print(degreeErrors)