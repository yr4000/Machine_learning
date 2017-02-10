from __future__ import division
import numpy as np
import numpy.random
from MulticlassSGD.hw3 import *
from MulticlassSGD.MultiClassSGD import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def KernelizedMultiClassSGD(X,Y,eta0,Ker,C = 1,K = 10, T = 100,printMsg = False):
    printMessage("Starting KernelizedMultiClassSGD",printMsg)
    m = len(X)
    A = [[0 for u in range(m)] for v in range(K)] # create K zero vectors
    XT = [0 for i in range(T)]
    for t in range(T):
        i = int(np.random.uniform(0,len(X))) ##pick random x
        XT[t] = i
        x , y, eta = X[i], int(Y[i]), eta0/T
        max_index = find_argmax_ker(A,x,y,Ker,X,XT,eta,C,t) ##find argmax according to function
        Wj = 0 #this will be our kernelized Wj
        Wy = 0 #this will be our kernelized Wy
        for u in range(m):
            Wj += A[max_index][u]*eta*C*Ker(X[u],x)
            Wy += A[y][u]*eta*C*Ker(X[u],x)
        if(Wy - Wj <= 1):
            A[max_index][i] -= 1
        A[y][i] += 1
    printMessage("Finished MultiClassSGD", printMsg)
    return A, XT, T

def find_argmax_ker(A,x,y,Ker,X,XT,eta,C,T):
    m = len(A[0])
    max_index = 0
    max_val = -np.Infinity
    for j in range(len(A)): ##for each row of w, that is, for each j
        Wj = 0  # this will be our kernelized Wj
        Wy = 0  # this will be our kernelized Wy
        for u in range(m):
            Wj +=  A[j][u]*eta*C*Ker(X[u], x)
            Wy +=  A[y][u]*eta*C*Ker(X[u], x)
        val = Wj - Wy
        if(j!=y):
            val+=1
        if (val > max_val):
            max_val = val
            max_index = j
    return max_index

def calcKerAcc(A,data,labels,Ker,XT,eta0,C,T):
    res = 0
    n = len(data)
    for i in range(n):
        kerRes = [Ker(data[u], data[i]) for u in range(n)]
        values = np.dot(A,kerRes) #return a vector with each calssifier score .
        x = np.argmax(values) #for debugging
        if(labels[i] == np.argmax(values)): #if the maximum score equals to the real label - we won!
            res+=1
    return res/n


def bestKerEta(etas,index,Ker):
    printMessage("Starting bestKerEta, please wait...",True)
    n = len(etas)
    avgEtasAcc = [0 for i in range(n)]
    for i in range(n):
        A, XT, T = KernelizedMultiClassSGD(train_data[:250],train_labels[:250],etas[i],Ker)
        for k in range(5):
            avgEtasAcc[i] += calcKerAcc(A,validation_data[:250],validation_labels[:250],Ker,XT,etas[i],1,T)
        avgEtasAcc[i] /= 5
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
    plt.savefig("q7a"+str(index)+".png")
    plt.close()
    printMessage("Finished bestKerEta",True)
    return etas[avgEtasAcc.index(max(avgEtasAcc))]

def bestKerC(eta0,Cs,index,Ker):
    printMessage("Starting bestKerC",True)
    #Cs = [10**i for i in range(-10,11)]
    avgCsAcc = [0 for i in range(len(Cs))]
    for i in range(len(Cs)):
        W, XT, T = KernelizedMultiClassSGD(train_data[:250], train_labels[:250], eta0,Ker,C = Cs[i])
        for k in range(5):
            avgCsAcc[i] += calcKerAcc(W, validation_data[:250], validation_labels[:250],Ker,XT,eta0,Cs[i],T)
        avgCsAcc[i] /= 5
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
    plt.savefig("q7a"+str(index)+".png")
    plt.close()
    printMessage("Finished bestKerC",True)
    return Cs[avgCsAcc.index(max(avgCsAcc))]

def linear_Ker(u,v):
    return np.dot(u,v)

BEta0 =  bestKerEta([10**i for i in range(-5,6)],1,linear_Ker)
BBEta0 = bestKerEta([(BEta0/10)*i for i in range(1,11)] + [BEta0*i for i in range(1,11)],2,linear_Ker)
BC = bestKerC(BBEta0,[10**i for i in range(-5,6)],3,linear_Ker)
BBC = bestKerC(BBEta0,[(BC/10)*i for i in range(1,11)] + [BC*i for i in range(1,11)],4,linear_Ker)
#A = KernelizedMultiClassSGD(train_data,train_labels,BBEta0,linear_Ker,BBC)
#print("The best of W with eta0 = " + str(BBEta0) + " ,C= "
#      + str(BBC) + " has accuracy of: " + str(calcKerAcc(A, test_data[:250], test_labels[:250],linear_Ker,[],BBEta0,BBC,100)))