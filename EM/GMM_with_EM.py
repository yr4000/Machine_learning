from __future__ import division
import numpy as np
import numpy.random
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from scipy.misc import logsumexp
from EM.hw5 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def EM_for_GMM(train_data,num_of_digits = 5,T=10):
    '''
    1) initialize parameters for all the samples (mean, covariance matrix (maybe can saves as a variable because
       of definitions), Czs (probability to belong to digit z) and weights.
    2) write the E-step where the weights are being evaluated
    3) write the M-step where the parameters are being evaluated

    also remember to create different functions for each
     for seif a, make an adjustments for what was asked in the exercise - the right cov-matrix, and what is Q(teta,tetat)?

    '''
    print("Started initializing...")
    N = len(train_data) #number of samples
    d = len(train_data[0]) #dimention of the vector
    likelihood_for_t = np.zeros(T)
    Wts = np.array([np.repeat(1/num_of_digits,num_of_digits) for n in range(N)])#the probabilities of xi to belong to Cz
    Czs = np.repeat(1/num_of_digits,num_of_digits) #the probabilities to belong to distribution Cz
    means = initializeMeans(train_data,num_of_digits) #means of distributions Czs
    covariances = initializeCovs(means) #covariance matrixes of distributions Czs
    print("Finished initializing")

    for t in range(T):
        print("Strating iteration No."+str(t))
        #E-step:
        evaluateWts(Wts,Czs, means, covariances,train_data)

        #M-step:
        maxCzs(Czs,Wts)
        maxMeans(means,Wts,train_data)
        maxCovariances(covariances,means,Wts,train_data)
        likelihood_for_t[t] = likelihood(means,covariances,Czs,train_data)/1000
        s1 = sum(Czs) #TODO these three for debug
        s2 = sum(Wts[0])
        s3 = sum(Wts)
        print("Finished iteration")

    return Czs,means,covariances,likelihood_for_t


#for each sample evaluates its probability to belong to gausian k.
def evaluateWts(Wts,Czs,means, covariances,train_data):
    print("Evaluating Wts...")
    N = len(train_data)
    d = len(train_data[0])
    for i in range(N): #for each sample
        s=0
        for k in range(len(Czs)): #for each gausian
            Wts[i][k] = np.exp(x_belong_to_Cz_prob(means[k],covariances[k],Czs[k],train_data[i]) - \
                  prob_to_get_x_above_all_Czs(means,covariances,Czs,train_data[i]))
        s = sum(Wts[i]) #TODO for debug
    print("Finished evlauation")

#from the scribes: Pr[zi = k | xi, teta-t]
def x_belong_to_Cz_prob(mean,covariance,Czk,x):
    MN = multivariate_normal(mean=mean, cov=covariance)  # creates multivariant_normal distibution over ut, cov-t
    prob = MN.logpdf(x)+np.log(Czk)
    return prob  # pdf = probability density function

#from the scribes: sum(Pr[zi = k | xi, teta-t]) where 1<=k<=K
def prob_to_get_x_above_all_Czs(means,covariances,Czs,x):
    log_probs = np.zeros(len(Czs))
    for k in range(len(Czs)):
        log_probs[k] = x_belong_to_Cz_prob(means[k],covariances[k],Czs[k],x)
    f_prob = logsumexp(log_probs)
    return f_prob

def maxCzs(Czs,Wts):
    print("Evaluating Czs...")
    n = len(Wts)
    s = sum(Wts)
    for k in range(len(Czs)):
        Czs[k] = Nk(Wts,k)/len(Wts)
    print("Finished evlauation")

#returns the sum of probabilities of the samples to belong to gausian k
def Nk(Wts,k):
    sum = 0;
    for i in range(len(Wts)):
        sum += Wts[i][k]
    return sum

def maxMeans(means,Wts,train_data):
    print("Evaluating means...")
    for k in range(len(means)):
        sum = 0
        for i in range(len(train_data)):
            sum += Wts[i][k]*train_data[i]
        means[k] = sum/Nk(Wts,k)
    print("Finished evlauation")

def maxCovariances(covariances,means,Wts,train_data):
    print("Evaluating Covariances...")
    d = len(train_data[0])
    for k in range(len(covariances)):
        sum = 0
        for i in range(len(train_data)):
            sum += Wts[i][k]*np.dot((train_data[i]-means[k]),(train_data[i]-means[k]))
        covariances[k] = sum/(Nk(Wts,k)*d)
    print("Finished evlauation")


def initializeMeans(train_data,num_of_digits,kmeans=False):
    if(kmeans):
        means = KMeans(n_clusters=5,random_state=0).fit(train_data)
        return means.cluster_centers_
    sample_size = len(train_data)//num_of_digits
    means = []
    all_indexes = np.array([i for i in range(len(train_data))])
    for i in range(num_of_digits):
        np.random.shuffle(all_indexes)
        indexes = all_indexes[:sample_size]
        sample = train_data[indexes]
        means.append(np.sum(sample,axis=0)/sample_size)
        all_indexes = all_indexes[sample_size:]
    return np.array(means)

def initializeCovs(means):
    '''
    covM = np.cov(train_data)
    top = covM[0][0]
    avg = np.average(covM)
    '''

    num_of_digits = len(means)
    covs = np.zeros(num_of_digits)
    for i in range(num_of_digits):
        covs[i] = np.cov(means[i])
    return covs

    #return np.repeat(1,len(means))

#Q(teta,teta-t)
def likelihood(means,covariances,Czs,train_data):
    likelihood = 0
    for i in range(len(train_data)):
        likelihood += prob_to_get_x_above_all_Czs(means,covariances,Czs,train_data[i])
    return likelihood

def createLabelPictureFromMean(mean,index):
    plt.imshow(reshape(mean, (28, 28)))
    plt.savefig("MeanLabelNo"+str(index)+".png")
    plt.close()

def create_plot(arrX,arrY,title, x_label,y_label,file_name, semilog = False):
    fig = plt.figure()
    fig.suptitle(title, fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if semilog:
        plt.semilogx(arrX,arrY)
    else:
        plt.plot(arrX,arrY)
    #plt.show()
    plt.savefig(file_name)
    plt.close()

def get_accuracy(data,labels,means,covariances,digits_vals):
    acc = 0
    digits_No = len(means)
    MNs = create_MNs(means,covariances)

    for i in range(len(data)):
        probs = np.zeros(digits_No)
        for k in range(digits_No):
            probs[k] = MNs[k].logpdf(data[i])
        if(digits_vals[np.argmax(probs)] == labels[i]):
            acc += 1
    return acc/len(data)

def get_means_vals(means,covariances,data,labels):
    digits_No = len(means)
    num_of_vecs = 2
    MNs = create_MNs(means,covariances)
    digits = [0,1,3,4,8]
    diffrent_digits = [[None for k in range(num_of_vecs)] for l in range(digits_No)]
    means_vals = np.zeros(digits_No)

    #extracting different vectors
    k,l = 0,0
    for i in range(len(data)):
        if (diffrent_digits[digits.index(labels[i])][l] == None):
            diffrent_digits[digits.index(labels[i])][l] = data[i]
            k += 1
        if(k==digits_No): l+=1
        if l==num_of_vecs: break

    #for each mean evaluate it's best digit
    for i in range(digits_No): #for each row
        probs = np.zeros(digits_No)
        for j in range(num_of_vecs): #for each cell
            for k in range(digits_No):
                probs[k] += np.linalg.norm(means[k] - diffrent_digits[i][j])#MNs[k].logpdf(diffrent_digits[i][j]) #calculate thr sum of probabilites of each mean to be a digit
        means_vals[i] = digits[np.argmin(probs)]
    print("those are the means_vals: " + str(means_vals))

    return means_vals



def create_MNs(means,covariances):
    digits_No = len(means)
    MNs = []
    for k in range(digits_No):
        MNs.append(multivariate_normal(mean=means[k], cov=covariances[k]))
    return MNs



def q4():
    T = 20
    ts = [t for t in range(T)]
    Czs, means, covariances, likelihood = EM_for_GMM(train_data,T=T)
    create_plot(ts,likelihood,"Likelihood as function of t","t","Likelihood","Likelihood_plot.png")
    for i in range(len(means)):
        createLabelPictureFromMean(means[i],i)
    digits_vals = get_means_vals(means,covariances,train_data,train_labels)
    print("The EM accuracy rate is: "+str(get_accuracy(test_data,test_labels,means,covariances,digits_vals)))

q4()
