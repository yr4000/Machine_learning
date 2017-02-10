from __future__ import division
import numpy as np
import numpy.random
from Adaboost.hw4 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def adaboost(my_data,my_labels,H,T):
    print("staring adaboost, please wait...")
    print("initializing....")
    m = len(my_data)
    Dt = np.repeat(np.divide(1,m),m) #creates a numpy array with 1/m in each cell
    best_at_and_ht = [[0, 0, 0] for t in range(T)]  # each row saves [at,i,j] where i is the pixel and j is the h
    num_of_samples = len(my_data)
    num_of_pixels = len(my_data[0])
    num_of_h = len(H)
    print("finished initialize")
    print("Dt[0]: " + str(Dt[0]))

    print("compute best ats and hs...")
    for t in range(T):
        print("started iteration "+str(t+1)+" out of "+str(T))
        epsilon_for_h = [[0 for n in range(num_of_h)] for m in range(num_of_pixels)]  # saves for each hipothesis it's mistake

        print("started computing error...")
        for k in range(num_of_samples): #for each picture
            digit = my_data[k]
            for i in range(num_of_pixels): #for each pixel
                for j in range(num_of_h): #for each hypothesis
                    if (H[j](digit[i]) != my_labels[k]):
                        epsilon_for_h[i][j] += 1*Dt[k] #update the mistake of hj of pix i
        epsilon_for_h = np.divide(epsilon_for_h,num_of_samples) #calculatin the mistake in percentage
        print("finished computing error")

        et = np.min(epsilon_for_h) #lowest error
        full_index = np.argmin(epsilon_for_h) #this gives the index of the lowest error in the matrix, which we need to modify
        pix_index = int(full_index/num_of_h) #returns the pixels index
        h_index = full_index % num_of_h #the h index
        at = 0.5*np.log((1-et)/et)

        print("et = " + str(et) + " h_index = " + str(h_index) + " pix_index = " + str(pix_index) + " at = " + str(at))

        best_at_and_ht[t][0] ,best_at_and_ht[t][1] ,best_at_and_ht[t][2] = at, pix_index, h_index
        Zt = 2*(et*(1-et))**0.5

        print("started update_Dt")
        Dt = update_Dt(Dt,Zt,at,H[h_index],pix_index,my_data,my_labels)
        print("finished update_Dt")

    return best_at_and_ht

def update_Dt(Dt,Zt,at,best_h,pix_index, my_data,my_labels):
    i = 0
    for d in np.nditer(Dt, op_flags=['readwrite']):
        d[...] = (d * np.exp(-at * best_h(my_data[i][pix_index]) * my_labels[i])) / Zt
        i += 1
    return Dt


def h(best_at_and_ht,H,x,T):
    res = 0
    for t in range(T):
        at, pix_index, h_index = best_at_and_ht[t][0], best_at_and_ht[t][1], best_at_and_ht[t][2]
        res += at*H[h_index](x[pix_index])
    return res


def calc_error(best_at_and_ht,H, my_data, my_labels,T):
    error = 0
    for i in range(len(my_data)):
        if sign(h(best_at_and_ht,H,my_data[i],T)) != my_labels[i]:
            error += 1
    return error/len(my_data)

def loss(best_at_and_ht,H,my_data,my_labels,T):
    result = 0
    for i in range(len(my_data)):
        inner_sum = h(best_at_and_ht,H,my_data[i],T)
        result += np.exp(-my_labels[i]*inner_sum)
    return result/len(my_data)

#----------------------

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


def a():
    T = 60
    ts = [t for t in range(T)] #for plot
    H = [(lambda x: 1 if x <= -120 else -1),
         (lambda x: 1 if x <= -80 else -1),
         (lambda x: 1 if x <= -40 else -1),
         (lambda x: 1 if x <= 0 else -1),
         (lambda x: 1 if x <= 40 else -1),
         (lambda x: 1 if x <= 80 else -1),
         (lambda x: 1 if x <= 120 else -1),
         (lambda x: -1 if x <= -120 else 1),
         (lambda x: -1 if x <= -80 else 1),
         (lambda x: -1 if x <= -40 else 1),
         (lambda x: -1 if x <= 40 else 1),
         (lambda x: -1 if x <= 80 else 1),
         (lambda x: -1 if x <= 120 else 1)
         ]


    best_at_and_ht = adaboost(train_data,train_labels,H,T)

    print("Calculating train-error, test-error and loss per t...")
    ats = [best_at_and_ht[n][0] for n in range(T)]
    train_errors = [0 for t in range(T)]
    test_errors = [0 for t in range(T)]
    loss_train_per_t = [0 for t in range(T)]
    loss_test_per_t = [0 for t in range(T)]
    for t in range(T):
        train_errors[t] = calc_error(best_at_and_ht,H,train_data,train_labels,t+1)
        test_errors[t] = calc_error(best_at_and_ht, H, test_data, test_labels, t+1)
        loss_train_per_t[t] = loss(best_at_and_ht,H,train_data,train_labels,t+1)
        loss_test_per_t[t] = loss(best_at_and_ht, H, test_data, test_labels, t+1)
    print("Finished calculating errors")

    print("Creating plots...")
    create_plot(ts,train_errors,'Train-errors as function of t',"t","Train-error","q5a_train_error.png")
    create_plot(ts, test_errors, 'Test-errors as function of t', "t", "Test-error", "q5a_test_error.png")
    create_plot(train_errors,ats,'Alpha-t as function of epsilon-t',"Epsilon-t","Alpha-t","q5a_alpha_epsilon.png")
    create_plot(ts,loss_train_per_t,'Train-loss as function of t',"t","Train-loss","q5b_train_loss.png")
    create_plot(ts, loss_test_per_t, 'Test-loss as function of t', "t", "Test-loss", "q5b_test_loss.png")
    print("Finished creating figure")
    print("Finished adaboost experiment")

a()


"""
def create_new_samples(Dt,my_data,my_labels):
    current_sample = []
    current_my_labels = []
    while(len(current_sample) == 0): #TODO: this is a dangerous idea
        i = 0
        for d in np.nditer(Dt):
            choose = np.random.uniform()
            if d > choose:
                current_sample += [my_data[i]]
                current_my_labels += [my_labels[i]]
            i += 1
    return  current_sample,current_my_labels
"""
