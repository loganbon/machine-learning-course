import numpy as np
import math

class Model:

    def __init__(self, trainFile, testFile):
        self.words = dict()
        self.createDictionary(trainFile, testFile)
        

    def perplexity(self, weights):
        '''
        Takes in array of probability weights and returns the perplexity.
        '''
        logSum = 0
        for w in weights:
            if w == 0:
                logSum += -math.inf
            else:
                logSum += math.log(w)

        return math.e ** (logSum / -len(weights))


    def createDictionary(self, trainFile, testFile):
        '''
        Creates dictionary of key value pairs for words in a file with default value of zero.
        '''
        with open(trainFile) as f:
            self.trainContents = f.read().replace('\n',' ').split(' ')
            self.trainContents = np.asarray([x for x in self.trainContents if x != ''])
            for w in self.trainContents:
                if w not in self.words:
                    self.words[w] = 0

        with open(testFile) as f:
            self.testContents = f.read().replace('\n',' ').split(' ')
            self.testContents = np.asarray([x for x in self.testContents if x != ''])
            for w in self.testContents:
                if w not in self.words:
                    self.words[w] = 0


    def train(self, method = 'ML', trainPartition = 1, alpha = 0, evidence = False):

        trainData = self.trainContents[:int(len(self.trainContents)*trainPartition)]

        trainWords = self.words.copy()

        for w in trainData:
            trainWords[w] += 1

        N = len(trainData)
        K = len(trainWords)

        if method == 'ML':
            trainWords = {k: m / N for k, m in trainWords.items()}
        elif method == 'MAP':
            trainWords = {k: (m + alpha - 1) / (N + (alpha * K) - K) for k, m in trainWords.items()}
        else:
            trainWords = {k: (m + alpha) / (N + (alpha * K)) for k, m in trainWords.items()}

        trainProbs = [trainWords[x] for x in trainData]
        testProbs = [trainWords[x] for x in self.testContents]

        evid = None
        if evidence:
            evid = math.log(math.factorial(int((alpha * K) - 1))) - math.log(math.factorial(int((alpha * K) + N - 1)))
            evid -= math.log(math.factorial(int(alpha - 1))) * K

            for w in trainWords:
                evid += math.log(math.factorial(int(alpha + trainWords[w] - 1)))

        return self.perplexity(trainProbs), self.perplexity(testProbs), evid