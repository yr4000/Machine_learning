from __future__ import division
import numpy as np
import pandas as pd
from Perceptron_and_SGD.hw2 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


#----------------------------------------------------------------------------------------------------



def normalize(x):
    norm = np.linalg.norm(x)
    if norm == 0:
        return
    else:
        x /= norm

#S is for samples
def perceptron(S,labels):
    w = np.zeros((len(S[0]),), dtype=np.int) #initialize W
    for i in range(len(S)):
        prediction = predict(w,S[i])
        if prediction == labels[i]:
            continue
        else:
            w = w+np.multiply(labels[i],S[i]) #Wt+1 = Wt + c*(x)x
    return w

#predicts the label of s in respect to w
#also it normalises s
def predict(w,s):
    normalize(s)
    if (np.linalg.norm(w) == 0):
        prediction = 1
    else:
        prediction = np.dot(s, w) / np.linalg.norm(w)  # (x*Wt)/||Wt||
    #print(prediction)
    prediction = 1 if prediction >= 0  else -1
    return prediction

def calcAcc(w,S,labels,Rmisses = False):
    acc = 0
    misses = []
    for i in range(len(S)):
        prediction = predict(w, S[i])
        if prediction == labels[i]:
            acc += 1
        else:
            misses+=[i]
    if Rmisses:
        return acc/len(labels), misses
    return acc/len(labels)


def a(n):
    #number of samples we are going to check
    Esize = [5,10,50,100,500,1000,5000]
    lenE = len(Esize)
    #here we will save the accuracies we have found. n is the number of runs. in this case n=100
    accuracies = [[0 for j in range(n)] for i in range(len(Esize))]
    for i in range(lenE):
        size = Esize[i]
        CData = [0 for x in range(size)]
        CLabels = [0 for x in range(size)]
        #create random permutations of the sample, n times.
        for j in range(n):
            indexes = np.random.permutation(size)
            for k in range(size):
                CData[k] = train_data[indexes[k]]
                CLabels[k] = train_labels[indexes[k]]
            w = perceptron(CData,CLabels)
            accuracies[i][j] = calcAcc(w,test_data,test_labels)
        accuracies[i].sort()
    data = {'Mean accuracies for 100 runs' : pd.Series([np.mean(accuracies[i]) for i in range(lenE)], index = Esize),
            '5% percentile' : pd.Series([accuracies[i][5] for i in range(lenE)], index = Esize),
            '95% percentile' : pd.Series([accuracies[i][95] for i in range(lenE)], index = Esize)}
    table = pd.DataFrame(data)
    print(table)

def b():
    w = perceptron(train_data,train_labels)
    plt.imshow(reshape(w, (28, 28)), interpolation='nearest')
    plt.savefig("q1b.png")
    plt.close()

def cd():
    w = perceptron(train_data, train_labels)
    #return calcAcc(w,test_data,test_labels,True)
    accuracy, misses = calcAcc(w,test_data,test_labels,True)
    print(accuracy)
    plt.imshow(reshape(test_data[misses[0]], (28, 28)))
    plt.savefig("q1d1.png")
    plt.close()
    plt.imshow(reshape(test_data[misses[1]], (28, 28)))
    plt.savefig("q1d2.png")
    plt.close()

a(100)
b()
cd()

#I want to see the pictures
#plt.imshow(reshape(train_data[40],(28,28)))




