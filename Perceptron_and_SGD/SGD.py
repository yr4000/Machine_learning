from __future__ import division
import numpy as np
from Perceptron_and_SGD.hw2 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from numpy import *
import numpy.random
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing

mnist = fetch_mldata('MNIST original')
data = mnist['data']
labels = mnist['target']

neg, pos = 0,8
train_idx = numpy.random.RandomState(0).permutation(where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
test_idx = numpy.random.RandomState(0).permutation(where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

train_data_unscaled = data[train_idx[:6000], :].astype(float)
train_labels = (labels[train_idx[:6000]] == pos)*2-1

validation_data_unscaled = data[train_idx[6000:], :].astype(float)
validation_labels = (labels[train_idx[6000:]] == pos)*2-1

test_data_unscaled = data[60000+test_idx, :].astype(float)
test_labels = (labels[60000+test_idx] == pos)*2-1

# Preprocessing
train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)

#---------------------------------------------------------------------

def SGD(data,labels,eta0, T, C = 1):
    n, eta = len(data),eta0
    w = [0 for i in range(len(data[0]))]
    for t in range(1,T+1):
        i = np.random.uniform(0,n)
        xi, yi, eta = data[i], labels[i], eta0/t
        if (yi*np.dot(w,xi))<1:
            w = np.dot((1-eta),w)+np.dot(eta*C*yi,xi)
    return w

def predict(w,s):
    if (np.linalg.norm(w) == 0):
        prediction = 1
    else:
        prediction = np.dot(s, w) / np.linalg.norm(w)  # (x*Wt)/||Wt||
    #print(prediction)
    prediction = 1 if prediction >= 0  else -1
    return prediction

def calcAcc(w,VSet,LSet):
    res = 0
    for i in range(len(LSet)):
        if predict(w,VSet[i]) == LSet[i]:
            res += 1
    return res/len(VSet)

def a(etas,index):
    #etas = [10**i for i in range(-10,11)]
    avgEtasAcc = [0 for i in range(len(etas))]
    for i in range(len(etas)):
        w = SGD(train_data,train_labels,etas[i],1000)
        for k in range(10):
            avgEtasAcc[i] += calcAcc(w,validation_data,validation_labels)
        avgEtasAcc[i] /= 10
    #preparing the plot:
    fig = plt.figure()
    fig.suptitle('Average Accuracy as size of Eta0', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel("Eta0 size")
    ax.set_ylabel("Average Accuracy")
    plt.semilogx(etas,avgEtasAcc)
    #plt.plot(etas,avgEtasAcc)
    #plt.show()
    plt.savefig("q3a"+str(index)+".png")
    plt.close()
    return etas[avgEtasAcc.index(max(avgEtasAcc))]

def b(eta0,Cs,index):
    #Cs = [10**i for i in range(-10,11)]
    avgCsAcc = [0 for i in range(len(Cs))]
    for i in range(len(Cs)):
        w = SGD(train_data,train_labels,eta0,1000,Cs[i])
        for k in range(10):
            avgCsAcc[i] += calcAcc(w,validation_data,validation_labels)
        avgCsAcc[i] /= 10
    #preparing the plot:
    fig = plt.figure()
    fig.suptitle('Average Accuracy as size of C according to best Eta0', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel("C size")
    ax.set_ylabel("Average Accuracy")
    plt.semilogx(Cs,avgCsAcc)
    #plt.plot(etas,avgEtasAcc)
    #plt.show()
    plt.savefig("q3b"+str(index)+".png")
    plt.close()
    return Cs[avgCsAcc.index(max(avgCsAcc))]

def cd(eta0,bestC):
    w = SGD(train_data,train_labels,eta0,20000,C = bestC)
    plt.imshow(reshape(w, (28, 28)))
    plt.savefig("q3c.png")
    plt.close()
    print("The best of linear SGD-SVM with eta0 = " + str(eta0) + " ,C= "
          +str(bestC) + " has accuracy of: " + str(calcAcc(w,test_data, test_labels)))

bestEta0 = a([10**i for i in range(-10,11)],1)
bestBestEta0 = a([(bestEta0/10)*i for i in range(1,11)] + [bestEta0*i for i in range(1,11)],2)
bestC = b(bestBestEta0,[10**i for i in range(-10,11)],1)
bestBestC = b(bestBestEta0,[(bestC/10)*i for i in range(1,11)] + [bestC*i for i in range(1,11)],2)
cd(bestBestEta0,bestBestC)
