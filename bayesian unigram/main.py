import numpy as np
import matplotlib.pyplot as plt
import math

from model import Model

## TASK 1

print('\nTASK 1)')

methods = ['ML', 'MAP', 'Predictive']
partitions = [1/128, 1/64, 1/16, 1/4, 1]
trainFile = 'pp1data/training_data.txt'
testFile = 'pp1data/test_data.txt'

model = Model('pp1data/training_data.txt', 'pp1data/test_data.txt')

labelData = {'ML':'blue', 'MAP':'green', 'Predictive':'orange'}

x_values = ['N/128', 'N/64', 'N/16', 'N/4', 'N']
x_axis = np.arange(0, 5, 1)
plt.xticks(x_axis, x_values)

for m in methods:
    print('Testing', m + '...')
    tempX, tempY = [], []
    for part in partitions:
        if m == 'ML':
            trainPP, testPP, _ = model.train(method=m, trainPartition=part)
        else:
            trainPP, testPP, _ = model.train(method=m, trainPartition=part, alpha=2)
        tempX.append(trainPP)
        tempY.append(testPP)
    print('Test PP', tempY)
    plt.plot(x_values, tempX, color=labelData[m], label=m+' Train')
    plt.plot(x_values, [x if x != math.inf else 9999 for x in tempY], linestyle='--', color=labelData[m], label=m+' Test')

plt.legend(loc='lower right')
plt.xlabel('Training Size')
plt.ylabel('Perplexity')
plt.title('Perplexity vs. Training Size for ML, MAP, and Pred. Estimates')
plt.savefig('output/task1.png')
plt.clf()

## TASK 2

print('\nTASK 2)')
print('Testing predictive for alphas (1 - 10)...')

alphas = np.arange(1, 11)
testPerps = []
evidence = []
for a in alphas:
    _, testPP, evid = model.train(method='Predictive', trainPartition=1/128, alpha=a, evidence=True)
    testPerps.append(testPP)
    evidence.append(evid)

plt.plot(alphas, testPerps, color='red', label='Test Perplexity')
plt.xlabel('Alpha')
plt.ylabel('Perplexity')
plt.title('Test Perplexity vs. Alpha for Pred. Estimate')
plt.savefig('output/task2pp.png')
plt.clf()
plt.plot(alphas, evidence, color='blue', label='Log Evidence')
plt.xlabel('Alpha')
plt.ylabel('Log Evidence')
plt.title('Log Evidence vs. Alpha for Pred. Estimate')
plt.savefig('output/task2evid.png')


print('Test Perplexity:', testPerps)
print('Evidence:', evidence)


## TASK 3


print('\nTASK 3)')
print('Training Predictive Model on pg345...')
auth1Model = Model('pp1data/pg345.txt.clean','pp1data/pg84.txt.clean')
auth2Model = Model('pp1data/pg345.txt.clean','pp1data/pg1188.txt.clean')

print('pg84 Perplexity:', auth1Model.train(method='Predictive', alpha=2)[1])
print('pg1188 Perplexity:', auth2Model.train(method='Predictive', alpha=2)[1])


