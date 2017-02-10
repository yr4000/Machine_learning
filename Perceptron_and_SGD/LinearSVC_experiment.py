from __future__ import division
from Perceptron_and_SGD.hw2 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC


def calcAccuracy(prediction,reality):
    res = 0
    n = len(reality)
    for i in range(n):
        if prediction[i]==reality[i]:
            res += 1
    return res/n

def a(Cs,index,data,labels,setName):
    #Cs = [10**i for i in range(-10,11)]
    accuracies = [0 for i in range(len(Cs))]
    for i in range(len(Cs)):
        h = LinearSVC(C=Cs[i],loss='hinge',fit_intercept=False)
        h.fit(train_data,train_labels) #here we train h on the training set
        prediction = h.predict(data)
        accuracies[i] = calcAccuracy(prediction,labels) #here we validate on validation set
    #preparing the plot:
    fig = plt.figure()
    fig.suptitle('Accuracy on '+setName+' as size of C', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel("C size")
    ax.set_ylabel("Accuracy on "+setName)
    plt.semilogx(Cs,accuracies)
    #plt.show()
    plt.savefig("q2a"+setName+str(index)+".png")
    plt.close()
    besti = accuracies.index(max(accuracies))
    #C2s = [Cs[besti-1]*i for i in range(1,11)] + [Cs[besti]*i for i in range(1,11)]
    return Cs[besti]

def c(bestC):
    h = LinearSVC(C=bestC, loss='hinge', fit_intercept=False)
    h.fit(train_data, train_labels)
    w = h.coef_[0]
    plt.imshow(reshape(w, (28, 28)))
    plt.savefig("q2c.png")
    plt.close()

def d(bestC):
    h = LinearSVC(C=bestC, loss='hinge', fit_intercept=False)
    h.fit(train_data, train_labels)
    prediction = h.predict(test_data)
    print("The best of linear SVM with C = " + str(bestC) + " has accuracy of: " + str(calcAccuracy(prediction,test_labels)))
    return calcAccuracy(prediction,test_labels)



bestC = a([10**i for i in range(-10,11)],1,train_data,train_labels,"train")
bestBestC = a([(bestC/10)*i for i in range(1,11)] + [bestC*i for i in range(1,11)],2,train_data,train_labels,"train")
bestC = a([10**i for i in range(-10,11)],1,validation_data,validation_labels,"validation")
bestBestC = a([(bestC/10)*i for i in range(1,11)] + [bestC*i for i in range(1,11)],2,validation_data,validation_labels,"validation")
c(bestBestC)
d(bestBestC)
