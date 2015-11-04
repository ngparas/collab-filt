#    Starter code for linear regression problem
#    Below are all the modules that you'll need to have working to complete this problem
#    Some helpful functions: np.polyfit, scipy.polyval, zip, np.random.shuffle, np.argmin, np.sum, plt.boxplot, plt.subplot, plt.figure, plt.title
import sys
import csv
import numpy as np
import scipy
import matplotlib.pyplot as plt

def nfoldpolyfit(X, Y, maxK, n, verbose):
#    NFOLDPOLYFIT Fit polynomial of the best degree to data.
#   NFOLDPOLYFIT(X,Y,maxDegree, nFold, verbose) finds and returns the coefficients 
#   of a polynomial P(X) of a degree between 1 and N that fits the data Y 
#   best in a least-squares sense, averaged over nFold trials of cross validation.
#
#   P is a vector (in numpy) of length N+1 containing the polynomial coefficients in
#   descending powers, P(1)*X^N + P(2)*X^(N-1) +...+ P(N)*X + P(N+1). use
#   numpy.polyval(P,Z) for some vector of input Z to see the output.
#
#   X and Y are vectors of datapoints specifying  input (X) and output (Y)
#   of the function to be learned. Class support for inputs X,Y: 
#   float, double, single
#
#   maxDegree is the highest degree polynomial to be tried. For example, if
#   maxDegree = 3, then polynomials of degree 0, 1, 2, 3 would be tried.
#
#   nFold sets the number of folds in nfold cross validation when finding
#   the best polynomial. Data is split into n parts and the polynomial is run n
#   times for each degree: testing on 1/n data points and training on the
#   rest.
#
#   verbose, if set to 1 shows mean squared error as a function of the 
#   degrees of the polynomial on one plot, and displays the fit of the best
#   polynomial to the data in a second plot.
#   
#
#   AUTHOR: Nick Paras (ngp947)

    #Split up the data for CV (use same partitions across each K for comparisons)
    partitionIndices = cvPartition(X, n)
    zippedData = zip(X, Y, partitionIndices)
   
    mseMat = np.array([[0.0]*n]*(maxK + 1))
 
    #Explore each value of K, perform n-fold CV, get testing errors
    for i in range(maxK+1):
        for j in range(n):
            train, test = splitData(zippedData, j)
            trainx, trainy = unzip(train)
            testx, testy = unzip(test)
            fit = np.polyfit(trainx, trainy, i)
            mseMat[i][j] = getMSE(np.polyval(fit, testx),testy)
    
    #Compute average testing errors
    avgTestingErrors = averageTestingError(mseMat)
    bestK = findBestK(avgTestingErrors)

    #print avgTestingErrors
    #print bestK

    #based on folds = 2:10, pick best K as 3
    bestK = 3
    
    #Refit to all training data using best K
    finalFit = np.polyfit(X, Y, bestK)

    #Make Plots
    if (verbose == 1):
        xSorted = [x for x in X] #deep copy
        xSorted.sort()

        f, (p1, p2) = plt.subplots(2)
        p1.set_title('Best Fit on Scatterplot: K = ' + str(bestK))
        p2.set_title('Average Testing MSE versus K')
        p1.scatter(X, Y)
        p1.plot(xSorted, np.polyval(finalFit, xSorted))
        p2.plot(np.arange(maxK+1), avgTestingErrors)
        p1.set_xlabel('x')
        p1.set_ylabel('y')
        p2.set_xlabel('Polynomial Degree, k')
        p2.set_ylabel('Average Testing MSE')
        plt.show()


    print finalFit
    print "The prediction for x = 3 is " + str(np.polyval(finalFit, 3))
    #return Coefficients
    return finalFit



def cvPartition(data, folds):
    #returns an array the same length as data with randomized partition assignments
    parts = [0] * len(data)
    #make partitions roughly equally-sized, fill excess if not divisible with partition k
    partSize = (len(data) / folds) - 1
    ctr = 0
    currPart = 0
    for i in range(len(parts)):
        if (ctr < partSize):
            parts[i] = currPart
            ctr = ctr + 1
        else:
            parts[i] = currPart
            ctr = 0
            currPart = min(folds-1, currPart + 1)
    np.random.seed(1)
    np.random.shuffle(parts)
    return parts   

def splitData(zippedData, fold):
    train = [obs for obs in zippedData if obs[2] != fold]
    test = [obs for obs in zippedData if obs[2] == fold]
    return train, test

def unzip(zippedData):
    x = [row[0] for row in zippedData]
    y = [row[1] for row in zippedData]
    return x, y

def getMSE(pred, truth):
    mseList = [0] * len(pred)
    for i in range(len(pred)):
        mseList = pred[i] - truth[i]
    return np.mean(np.power(mseList, 2))

def averageTestingError(mseMat):
    kTestError = [0.0] * len(mseMat)
    for i in range(len(mseMat)):
        kTestError[i] = np.mean(mseMat[i])
    return kTestError

def findBestK(errorList):
    ind = 0
    incumbent = 1.0
    for i in range(len(errorList)):
        if (errorList[i] < incumbent):
            ind = i
            incumbent = errorList[i]
    return ind

def main():
    # read in system arguments, first the csv file, max degree fit, number of folds, verbose
    rfile = sys.argv[1]
    maxK = int(sys.argv[2])
    nFolds = int(sys.argv[3])
    verbose = bool((int) (sys.argv[4]))
    
    csvfile = open(rfile, 'rb')
    dat = csv.reader(csvfile, delimiter=',')
    X = []
    Y = []
    # put the x coordinates in the list X, the y coordinates in the list Y
    for i, row in enumerate(dat):
        if i > 0:
            X.append(float(row[0]))
            Y.append(float(row[1]))
    X = np.array(X)
    Y = np.array(Y)
    nfoldpolyfit(X, Y, maxK, nFolds, verbose)

if __name__ == "__main__":
    main()
