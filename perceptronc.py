import sys
import csv
import numpy as np
import scipy
import matplotlib.pyplot as plt


def perceptronc(w_init, X, Y):
    #PERCEPTRONC fit sequential perceptron to data X,Y

    #PERCEPTRONC(w_init, X, Y) finds and returns the Weights that
    #define the discriminator that linearly separates the data (if
    #possible)

    #w_init is the initialization of the weights with the same dimension
    #as each of the X observations, prepended with a 1

    #X and Y are data inputs, X the data and Y the class

    #returns w, the final weights and e, the number of epochs before termination
    #note that a max number of epochs (with each epoch being defined as
    #iterating all the way through the data points once) before the algorithm
    #will consider the data inseparable and automatically terminate to prevent
    #infinite loops. Default is set at 1000.

    #Author: Nick Paras (ngp947)

    e = 1000 #number of allowed epochs
    seq = 0
    m = len(X)
    maxIts = e * m
    k = seq % m
    w = w_init
    errors = [1]*len(X)
    #stop when all points correctly classified
    while (sum(errors) > 0):
        if (classify(w, X[k])*Y[k] <= 0):
            #misclassification, update w
            w[0] = w[0] + Y[k]
            w[1] = w[1] + X[k]*Y[k]
            errors[k] = 1
        else:
            errors[k] = 0
        #increment counters
        seq = seq + 1
        k = seq % m
        if (seq > maxIts):
            print "Maximum Iterations Reached, aborting..."
            break
   

    #return weights, number of epochs
    return (w, float(seq/m))

#Define testing function to calculate classification error for debugging
def classificationError(weights, X, Y):
    ctr = 0.0
    #iterate through points and count misclassifications
    for i in range(len(X)):
        if (classify(weights, X[i])*Y[i] < 0):
            ctr = ctr + 1 
    return ctr / len(X)

#Small wrapper to calculate the result of the classification
def classify(w, x):
    return (w[0] + w[1]*x)



def main():
    rfile = sys.argv[1]
    
    #read in csv file into np.arrays X1, X2, Y1, Y2
    csvfile = open(rfile, 'rb')
    dat = csv.reader(csvfile, delimiter=',')
    X1 = []
    Y1 = []
    X2 = []
    Y2 = []
    for i, row in enumerate(dat):
        if i > 0:
            X1.append(float(row[0]))
            X2.append(float(row[1]))
            Y1.append(float(row[2]))
            Y2.append(float(row[3]))
    X1 = np.array(X1)
    X2 = np.array(X2)
    Y1 = np.array(Y1)
    Y2 = np.array(Y2)
    #Perform transformation indicated in PDF, (x-1.5)^2
    X3 = np.power(X2-1.5,2)
    Y3 = Y2   

 
    #plot X2, Y2
    f, (p1, p2) = plt.subplots(2)
    p1.scatter(X2, Y2)
    p1.set_title('Y2 v X2')
    p1.set_xlabel('x')
    p1.set_ylabel('y')
    #plot X2t, Y2t
    p2.scatter(X3, Y3)
    p2.set_title('Y2t v X2t')
    p2.set_xlabel('(x-1.5)^2')
    p2.set_ylabel('y')
    plt.show()


    #Execute algorithm for (X1,Y1), (x2,Y2)
    w_init = [0.0, 0.0]
    #w1 = perceptronc(w_init, X1, Y1)
    #print w1
    w3 = perceptronc(w_init, X3, Y3)
    print w3




if __name__ == "__main__":
    main()
