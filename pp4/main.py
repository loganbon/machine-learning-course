import numpy as np
import matplotlib.pyplot as plt
import csv, math

def BayesianLogisticRegression(trainX, trainY, testX, testY):

    trainX = np.hstack((np.ones((trainX.shape[0],1)), trainX))
    testX = np.hstack((np.ones((testX.shape[0],1)), testX))

    alpha = 0.01
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

def gibbsLDA(filePath, K, D):

    beta = 0.01
    alpha = 5/K
    Nitr = 500

    data = [np.loadtxt(filePath + str(d), dtype=str, delimiter=' ')[:-1] for d in range(1,D+1)]

    V = set()
    [[V.add(w) for w in d] for d in data]
    vocab = {w:i for (i,w) in enumerate(list(V))}
    vocabRev = {i:w for (i,w) in enumerate(list(V))}
    V = len(V)

    Nwords = sum([len(d) for d in data])

    z = np.floor(np.random.rand(Nwords) * K).astype(int)
    w = [vocab[w] for w in np.hstack(data)]
    d = np.hstack([np.full(len(data[i]), i) for i in range(D)])

    pi = np.arange(0, Nwords)
    np.random.shuffle(pi)

    Cd = np.zeros((D,K))
    Ct = np.zeros((K, V))

    for i in range(Nwords):
        Cd[d[i]][z[i]] += 1
        Ct[z[i]][w[i]] += 1

    P = np.zeros(K)

    for i in range(Nitr):
        print(str(int(i/Nitr*100)) + '%', end='\r')
        for j in range(Nwords):
            word = w[pi[j]]
            topic = z[pi[j]]
            doc = d[pi[j]]

            Cd[doc][topic] -= 1
            Ct[topic][word] -= 1

            for k in range(K):
                P[k] = ((alpha + Cd[doc][k]) / ((alpha * K) + np.sum(Cd[doc]))) * ((Ct[k][word] + beta) / ( (V*beta) + np.sum(Ct[k])))

            P = P / np.sum(P)
            topic = np.random.choice(np.arange(K), 1, p=P)[0]
            z[pi[j]] = topic
            Cd[doc][topic] += 1
            Ct[topic][word] += 1

    M = [[(x, vocabRev[j]) for (j, x) in enumerate(row)] for row in Ct]
    
    with open('output/topicwords.csv', 'w') as f:
        wr = csv.writer(f)
        for row in M:
            wr.writerow([x[1] for x in sorted(row, key=lambda tup: tup[0], reverse=True)][:5])

    return z, Cd, Ct

print('Task 1)')
z, Cd, Ct = gibbsLDA('pp4data/20newsgroups/', 20, 200)
np.savetxt('output/Cd.txt', Cd, fmt='%i',delimiter=',')


print('Task 2)')

trainSizes = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]

dataX = np.genfromtxt('output/Cd.txt', delimiter=',')
dataY = np.genfromtxt('pp4data/20newsgroups/index.csv', delimiter=',')[:,1]
K = dataX.shape[0]
alpha = 5/K

data = [np.loadtxt('pp4data/20newsgroups/' + str(d), dtype=str, delimiter=' ')[:-1] for d in range(1,201)]

V = set()
[[V.add(w) for w in d] for d in data]
vocab = {w:i for (i,w) in enumerate(list(V))}

bag = np.zeros((200,len(V)))

for i in range(len(data)):
    for j in range(len(data[i])):
        idx = vocab[data[i][j]]
        bag[i][idx] += 1
    
for d in range(len(dataX)):
    dataX[d] = ((alpha + dataX[d]) / ((alpha * K) + np.sum(dataX[d])))

sampleTopicError = None
sampleBagError = None

N = dataX.shape[0]
N1 = int(2*N/3)
N2 = N - N1

for i in range(30):

    itrTopicError = []
    itrBagError = []

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

        error = BayesianLogisticRegression(trainX, trainY, testX, testY)
        itrTopicError.append(error)

        trainX, testX = bag[mask1][mask2], bag[~mask1]

        error = BayesianLogisticRegression(trainX, trainY, testX, testY)
        itrBagError.append(error)

    if i:
        sampleTopicError = np.vstack((sampleTopicError, itrTopicError))
        sampleBagError = np.vstack((sampleBagError, itrBagError))

    else:
        sampleTopicError = itrTopicError
        sampleBagError = itrBagError


sampleBagStd = np.std(sampleBagError, axis=0)
sampleTopicStd = np.std(sampleTopicError, axis=0)

plt.plot(trainSizes, np.average(sampleTopicError, axis=0), color='blue')
plt.errorbar(trainSizes, np.average(sampleTopicError, axis=0), yerr = sampleTopicStd, fmt ='o', color='green') 

plt.ylabel('Error Rate')
plt.xlabel('Training Percent')
plt.title('Topic Representation: Test Error vs. Training Size')
plt.savefig('output/t2-topic.png')
plt.clf()

plt.plot(trainSizes, np.average(sampleBagError, axis=0), color='blue')
plt.errorbar(trainSizes, np.average(sampleBagError, axis=0), yerr = sampleBagStd, fmt ='o', color='green') 

plt.ylabel('Error Rate')
plt.xlabel('Training Percent')
plt.title('Bag of Words: Test Error vs. Training Size')
plt.savefig('output/t2-bag.png')
plt.clf()