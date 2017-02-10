from __future__ import division
import numpy as np
import numpy.random
from MulticlassSGD.hw3 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def MultiClassSGD(X,Y,eta0,C = 1,K = 10, T = 5000,printMsg = False,index = 0):
    printMessage("Starting MultiClassSGD "+str(index),printMsg)
    W = [[0 for i in range(len(X[0]))] for x in range(K)] # create K zero vectors
    m = len(X)
    for t in range(1,T+1):
        i = int(np.random.uniform(0,len(X))) ##pick random x
        x , y, eta = X[i], int(Y[i]), eta0/T
        max_index, loss = find_argmax(W,x,y) ##find argmax according to function
        if (loss !=0):
            for j in range(K):
                if(j==max_index):
                    if(np.dot(W[y],x)-np.dot(W[j],x)<=1):
                        W[j] = np.dot(W[j],(1-eta)) - np.dot(x,eta*C)
                    else:
                        W[j] = np.dot(W[j],(1-eta))
                elif(j==y):
                    W[j] = np.dot(W[j],(1-eta)) + np.dot(x,eta*C)
                else:
                    W[j] = np.dot(W[j],(1 - eta))
        else:
            for j in range(K):
                W[j] = np.dot(W[j], (1 - eta))
    printMessage("Finished MultiClassSGD", printMsg)
    return W

def find_argmax(W,x,y):
    max_index = 0
    max_val = -np.Infinity
    n = len(W)
    for j in range(len(W)): ##for each row of w, that is, for each j
        val = np.dot(W[j],x) - np.dot(W[y],x)
        if(j!=y):
            val+=1
        if (val > max_val):
            max_val = val
            max_index = j
    return max_index, max_val

def calcAcc(W,data,labels):
    res = 0
    max_val = -np.Infinity
    prediction = -1
    n = len(data)
    for i in range(n):
        for j in range(len(W)):
            temp = np.dot(W[j],data[i])
            if temp > max_val:
                max_val = temp
                prediction = j
        if(prediction == labels[i]):
            res+=1
        max_val = 0
    return res/n

def printMessage(msg,bool = True):
    if bool: print(msg)

def bestEta(etas,index,printMsg = False):
    printMessage("Starting bestEta, please wait...",True)
    n = len(etas)
    avgEtasAcc = [0 for i in range(n)]
    for i in range(n):
        W = MultiClassSGD(train_data,train_labels,etas[i])
        for k in range(10):
            avgEtasAcc[i] += calcAcc(W,validation_data[:5000],validation_labels[:5000])
        avgEtasAcc[i] /= 10
        printMessage("Accuracy of eta no."+str(i)+" is: "+str(avgEtasAcc[i]),False)
    #preparing the plot:
    fig = plt.figure()
    fig.suptitle('Average Accuracy as size of Eta0', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel("Eta0 size")
    ax.set_ylabel("Average Accuracy")
    #plt.axis([etas[0], etas[n-1], -0.1, 95])
    plt.semilogx(etas,avgEtasAcc)
    #plt.plot(etas,avgEtasAcc)
    #plt.show()
    plt.savefig("q6a"+str(index)+".png")
    plt.close()
    printMessage("Accuracy of best eta0 is: " + str(max(avgEtasAcc)), printMsg)
    printMessage("Finished bestEta",True)
    return etas[avgEtasAcc.index(max(avgEtasAcc))]

def bestC(eta0,Cs,index,printMsg = False):
    printMessage("Starting bestC",True)
    #Cs = [10**i for i in range(-10,11)]
    avgCsAcc = [0 for i in range(len(Cs))]
    for i in range(len(Cs)):
        W = MultiClassSGD(train_data, train_labels, eta0,C = Cs[i])
        for k in range(10):
            avgCsAcc[i] += calcAcc(W, validation_data[:5000], validation_labels[:5000])
        avgCsAcc[i] /= 10
        printMessage("Accuracy of C no." + str(i) + " is: " + str(avgCsAcc[i]), False)
    #preparing the plot:
    fig = plt.figure()
    fig.suptitle('Average Accuracy as size of C according to best Eta0', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel("C size")
    ax.set_ylabel("Average Accuracy")
    plt.semilogx(Cs,avgCsAcc)
    #plt.plot(etas,avgEtasAcc)
    #plt.show()
    plt.savefig("q6a"+str(index)+".png")
    plt.close()
    printMessage("Accuracy of best C is: " + str(max(avgCsAcc)), printMsg)
    printMessage("Finished bestC",True)
    return Cs[avgCsAcc.index(max(avgCsAcc))]

def createLabelPictureFromW(w,eta0,bestC,index):
    plt.imshow(reshape(w, (28, 28)))
    plt.savefig("q6b"+str(index)+".png")
    plt.close()

def q6():
    BEta0 =  bestEta([10**i for i in range(-7,8)],1)
    BBEta0 = bestEta([(BEta0/10)*i for i in range(1,11)] + [BEta0*i for i in range(1,11)],2,printMsg = True)
    BC = bestC(BBEta0,[10**i for i in range(-7,8)],3)
    BBC = bestC(BBEta0,[(BC/10)*i for i in range(1,11)] + [BC*i for i in range(1,11)],4,printMsg = True)
    W = MultiClassSGD(train_data,train_labels,BBEta0,C = BBC)
    printMessage("Starting creating labels picturs",True)
    for i in range(len(W)):
        createLabelPictureFromW(W[i],BBEta0,BBC,i)
    printMessage("Finished creating labels picturs",True)
    print("The best of W with eta0 = " + str(BBEta0) + " ,C= "
          +str(BBC) + " has accuracy of: " + str(calcAcc(W,test_data[:5000], test_labels[:5000])))

    
#q6()
    
    
    
    
    
    
    
    
    
    
    
    