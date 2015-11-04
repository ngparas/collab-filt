# -*- coding: utf-8 -*-

# Starter code for item-based collaborative filtering
# Complete the function item_based_cf below. Do not change its name, arguments and return variables. 
# Do not change main() function, 


# import modules you need here.
import sys
import csv
import numpy as np
import scipy.stats

# Import data
def readData(datafile):
    dct = {}
    with open(datafile, 'rb') as csvfile:
        rdr = csv.reader(csvfile, delimiter='\t')
        for row in rdr:
            dct[int(row[0]), int(row[1])] = int(row[2])

    return dct
#This function creates the either user or item vectors
#can be used for either item-based or user-based by second arg
def getVectors(dctMat, userOrItem):
    ind = 0 #read users
    if (userOrItem != 0):
        ind = 1 #read items

    #split out number of users and items
    users = []
    items = []
    for key in dctMat:
        users.append(key[0])
        items.append(key[1])
    numUsers = len(set(users))
    numItems = len(set(items))
    #create more useful users/items lists
    users = range(1, numUsers + 1)
    items = range(1, numItems + 1)
    #initialize vectors with zeros for the elements and then add ratings    
    if (ind == 0):
        ratingsU = {}
        for u in users:
            ratingsU[u] = [0] * numItems
        for r in dctMat:
            try:
                ratingsU[int(r[0])][int(r[1])-1] = int(dctMat[r])
            except KeyError:
                ratingsU[int(r[0])][int(r[1])-1] = 0
        return ratingsU
    else: 
        ratingsI = {}
        for i in items:
            ratingsI[i] = [0] * numUsers
        for r in dctMat:
            try:
                ratingsI[int(r[1])][int(r[0])-1] = int(dctMat[r])
            except KeyError:
                ratingsI[int(r[1])][int(r[0])-1] = 0
        return ratingsI

#Get most common rating given a list of ratings
def mode(lst):
    cts = {}
    #get counts
    for e in lst:
        try:
            x = cts[e]
            cts[e] = cts[e] + 1
        except KeyError:
            cts[e] = 1
    m = 0 #initialize to 0
    ct = 0
    #find max
    for d in cts:
        if (cts[d] > ct):
            ct = cts[d]
            m = d
    return m

# Calculate the distances between two vectors
def getDist(vec1, vec2, subject, manOrPear):
    #remove the subject elements
    v1 = [vec1[e] for e in range(len(vec1)) if e != (subject -1)]
    v2 = [vec2[e] for e in range(len(vec2)) if e != (subject -1)]

    if (manOrPear == 1): #use manhattan dist
        dist = 0
        for i in range(0, len(v1)):
            dist = dist + abs(v1[i] - v2[i])
        return dist
    else: #pearson dist
        return scipy.stats.pearsonr(v1, v2)[0] * -1
#Define custom sorting function to sort a list of tuples by their second elements
def customSort(a, b):
    if (a[1] > b[1]):
        return 1
    elif (a[1] == b[1]):
        return 0
    else:
        return -1

#Calculate distances from vector in question to all others, sort, return
def findNearest(target, subject, vectors, distType, emptyCond):
    neighbors = []
    targetVector = vectors[target]
    
    #build out list of tuples: [(index, distance), ...]
    if (emptyCond == 0): #don't allow empty ratings
        for v in vectors:
            if (v != target and vectors[v][subject - 1] != 0):
                neighbors.append((v,getDist(targetVector, vectors[v], subject, distType)))

    else: #allow anything
        for v in vectors:
            if (v != target):
                neighbors.append((v,getDist(targetVector, vectors[v], subject, distType)))

    #sort by the distances and return a list of the first elements
    neighbors.sort(customSort) 
    return [n[0] for n in neighbors]

#Find the ratings for the top k neighbors
def getTopKRatings(k, neighbors, vectors, subject):
    #bit of error handling to ensure we stay within bounds
    #for each neighbor, retrieve their rating for the subject
    if (len(neighbors) >= k):
        ratings = [0] * k
        for i in range(0, k):
            ratings[i] = vectors[neighbors[i]][subject - 1]
    elif (len(neighbors) > 0):
        ratings = [0] * len(neighbors)
        for i in range(0, len(neighbors)):
            ratings[i] = vectors[neighbors[i]][subject - 1]
    else:
        ratings = [0]
    return ratings

def item_based_cf(datafile, userid, movieid, distance, k, iFlag):
    """ 
    build item-based collaborative filter that predicts the rating 
    of a user for a movie.
    This function returns the predicted rating and its actual rating.
    
    Parameters
    ----------
    <datafile> - a fully specified path to a file formatted like the MovieLens100K data file u.data 
    <userid> - a userId in the MovieLens100K data
    <movieid> - a movieID in the MovieLens 100K data set
    <distance> - a Boolean. If set to 0, use Pearson’s correlation as the distance measure. If 1, use Manhattan distance.
    <k> - The number of nearest neighbors to consider
    <iFlag> - A Boolean value. If set to 0 for user-based collaborative filtering, 
    only users that have actual (ie non-0) ratings for the movie are considered in your top K. 
    For item-based, use only movies that have actual ratings by the user in your top K. 
    If set to 1, simply use the top K regardless of whether the top K contain actual or filled-in ratings.

    returns
    -------
    trueRating: <userid>'s actual rating for <movieid>
    predictedRating: <userid>'s rating predicted by collaborative filter for <movieid>


    AUTHOR: Nick Paras
    """

    dat = readData(datafile)
    vecs = getVectors(dat, 1)
    nbrs = findNearest(movieid, userid, vecs, distance, iFlag)
    rts = getTopKRatings(k, nbrs, vecs, userid)
    predictedRating = mode(rts)
    try:
        trueRating = dat[userid, movieid]
    except KeyError:
        trueRating = 0
    return trueRating, predictedRating


def main():
    datafile = sys.argv[1]
    userid = int(sys.argv[2])
    movieid = int(sys.argv[3])
    distance = int(sys.argv[4])
    k = int(sys.argv[5])
    i = int(sys.argv[6])


    trueRating, predictedRating = item_based_cf(datafile, userid, movieid, distance, k, i)
    print 'userID:{} movieID:{} trueRating:{} predictedRating:{} distance:{} K:{} I:{}'\
    .format(userid, movieid, trueRating, predictedRating, distance, k, i)




if __name__ == "__main__":
    main()
