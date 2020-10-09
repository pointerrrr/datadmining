#Mike Knoop (5853915)
#Gideon Ogilvie (5936373)
#Raka Schipperheijn (2827506)


import numpy as np
from anytree import AnyNode, RenderTree
import random

#tree_grow
#x = a 2d matrix of n by m, containing n data entries of size m
#y = a vector of classifiers for x of length n
#nmin = amount of entries a node needs for it to be allowed to split
#minleaf = minimal amount of entries a node needs to exist
#nfeat = total number of features to be considered for a split
#returns a tree, where each node contains a
#tuple (correctly classified (int), incorrectly classified(int), entries in this node (float[,]), classifier for entries int[]),
#an int on what column the data was split and a float of the input data, where the data has been split according to the gini index
def tree_grow(x, y, nmin, minleaf, nfeat):
    goodNodes = 0
    for i in y:
        goodNodes += i
    start = AnyNode(tuple=(goodNodes, len(y) - goodNodes, x, y), splitIndex = -1, splitValue = -1)

    toDoNodes = [start]
    while len(toDoNodes) > 0:
        curNode = toDoNodes[0]
        del toDoNodes[0]

        nodeArray = curNode.tuple[2]
        nodeClassifier = curNode.tuple[3]

        if len(nodeArray) < minleaf:
                raise Exception("Dit mag niet voor komen")
        if len(nodeArray) < nmin:
            continue         

        possibleSplits = np.array(random_unique_list(nfeat, len(curNode.tuple[2][0])))
        splitValues = []
        for i in range(len(possibleSplits)):
            splitValue = consider_split(curNode, possibleSplits[i])
            if splitValue == (-1,-1,-1):
                continue
            splitValues.append(splitValue)
        splitValues = sorted(splitValues, key=lambda tup: tup[1])

        #There are no valid splits on this node.
        if len(splitValues) == 0:
            continue

        temp = []
        for i in range(len(splitValues)):
            temp.append(splitValues[i][1])
        length = len(np.unique(np.array(temp))) 
        
        splitValues.reverse()
        splitValues = np.array(splitValues)
        lList = []
        lClassifier = []
        rList = []
        rClassifier = []
        currentSplitValue = -1
        currentSplitIndex = -1
        for i in range(len(splitValues)):
            lList = []
            lClassifier = []
            rList = []
            rClassifier = []
            splitValue = splitValues[i]
            if splitValue[2] == -1 or splitValue[1] == 0.0:
                continue
            currentSplitValue = splitValue[0]
            currentSplitIndex = splitValue[2]
            for i in range(len(nodeArray)):
                #python thinks splitValue[2] is a float, because the other elements of the tuple are, but it's not
                if nodeArray[i][int(splitValue[2])] <= splitValue[0]:
                    lList.append(np.array(nodeArray[i]))
                    lClassifier.append(nodeClassifier[i])
                else:
                    rList.append(nodeArray[i])
                    rClassifier.append(np.array(nodeClassifier[i]))
            #make sure to only split of neither of the child nodes would violate the minleaf constraint
            if len(lList) >= minleaf and len(rList) >= minleaf:
                break
            

        if len(lList) < minleaf or len(rList) < minleaf:
                continue

        if len(lList) > 0:
            ll, rl = get_class_distribution(lClassifier)
            toDoNodes.append(AnyNode(tuple=(ll, rl, np.array(lList), np.array(lClassifier)), parent=curNode))
        if len(rList) > 0:
            lr, rr = get_class_distribution(rClassifier)
            toDoNodes.append(AnyNode(tuple=(lr, rr, np.array(rList), np.array(rClassifier)), parent=curNode))
        if len(lList) != 0 or len(rList) != 0:
            curNode.splitIndex = currentSplitIndex
            curNode.splitValue = currentSplitValue

    return start

#tree_pred
#x = 2d matrix of n by m, with n data entries of length m
#tr = tree to classify data from x with
#returns a vector of length n, with the predicted result for each entry of x
def tree_pred(x, tr):
    predictionList = []
    for i in x:
        predictionList.append(prediction(i, tr))
    return predictionList

#tree_grow_b
#x, y, nmin, minleaf, nfeat are the same as for tree_grow
#m = amount of samples to be drawn
#returns a list of trees as resulting from tree_grow
def tree_grow_b(x, y, nmin, minleaf, nfeat, m):
    result = []

    for i in range(m):
        indices = []
        for i in range(len(x)):
            indices.append(random.randint(0,len(x)-1))
        xsample = []
        ysample = []
        for i in indices:
            xsample.append(x[i])
            ysample.append(y[i])
        result.append(tree_grow(xsample, ysample, nmin, minleaf, nfeat))

    return result

#tree_pred_b
#x = 2d matrix of n by m, with data entries of length m
#tr = list of trees to classify data from x with
#returns a vector of length n, with the predicted result of each entry of x, using the trees of tr
def tree_pred_b(x, tr):
    predLists = []
    for tree in tr:
        predLists.append(tree_pred(x, tree))

    result = []
    for j in range(len(x)):
        total = 0
        totalOne = 0
        for i in range(len(predLists)):        
            total += 1
            totalOne += predLists[i][j]
        if totalOne > (len(predLists) - totalOne):
            result.append(1)
        else:
            result.append(0)
    return result

#get_class_distribution
#classifier = int array containing zeros and/or ones
#returns a (int, int) tuple, with the amount of left as the first item, and amount of zeros as second item
def get_class_distribution(classifier):
    goodNodes = 0
    for i in classifier:
        goodNodes += i
    return (goodNodes, len(classifier) - goodNodes)

#consider_split
#node = node to consider a split on
#splitIndex = column to split on
#returns a tuple (float,float,int) value to split on, impurity reduction, column we split on
def consider_split(node, splitIndex):
    #splitIndex is sometimes seen as float by python, because it is in a tuple with other floats, but it is always an int
    splitIndex = int(splitIndex)
    x = np.array(node.tuple[2])
    y = np.array(node.tuple[3])

    x2 = np.empty((len(x), len(x[0]) + 1))

    for i in range(len(x)):
        for j in range(len(x[0]) + 1):
            if j == len(x[0]):
                x2[i,j] = y[i]
            else:
                x2[i,j] = x[i,j]
    x = x2
    x = np.array(sorted(x, key=lambda z : z[splitIndex]))

    splits = find_splits(x, splitIndex)
    if len(splits) == 0:
        return (-1,-1,-1)
    ln, rn = get_class_distribution(node.tuple[3])
    iT = impurity((ln,rn))
    splitValues = []
    for i in splits:
        #list for left node
        l = x[:i, 0:-1]
        #classifiers for left node
        lc = x[:i,-1]
        #class distribution for left node
        ld, rd = get_class_distribution(lc)
        #impurity of left node
        iL = impurity((ld, rd))
        piL = len(l) / len(x)
        #right node..
        r = x[i:, 0:-1]
        rc = x[i:, -1]
        lr, rr = get_class_distribution(rc)
        iR = impurity((lr, rr))
        piR = len(r) / len(x)
        #..right node
        #add split value of node to list of all splitvalues in node
        splitValues.append(iT - ((piL * iL) + (piR * iR)))
    max = (-1, 0)
    for i in range(len(splitValues)):
        if splitValues[i] > max[1]:
            max = (i, splitValues[i])
    if max[0] == -1:
        return (-1,-1,-1)
    result = ((x[splits[max[0]]][splitIndex]+ x[splits[max[0]] - 1][splitIndex] ) / 2, max[1], splitIndex)
    return result


#find_splits
#x = 2d array of n by m, of which we want to find the possible valid splits
#splitIndex = column to split on
#returns all valid splits inside node
def find_splits(x, splitIndex):
    splits = []
    prev = x[0]
    for i in range(1,len(x)):
        if x[i][splitIndex] != prev[splitIndex]:
            splits.append(i)
            prev = x[i]

    return splits

#impurity
#node = node to calculate impurity of
#returns the impurity of a node
def impurity(node):
    return (node[0] / (node[0] + node[1])) * (node[1] / (node[0] + node[1]))

#random_unique_list
#length = length of resulting list
#upperbound = upperbound of the values in resulting list (needs to be >= to length)
#return a list of unique random integer between 0 and "upperbound" of "length"
def random_unique_list(length, upperbound):
    result = np.empty(length)
    indices = np.empty(upperbound)

    for i  in range(upperbound):
        indices[i] = i

    for i in range(length):
        next = random.randint(0, upperbound-1)
        upperbound -= 1
        result[i] = indices[next]
        indices[next] = indices[upperbound]
        
    return result

#prediction
#entry = data entry to get prediction for
#tree = tree used to classify entry
#returns a prediction for entry (either 0 or 1)
def prediction(entry, tree):
    while len(tree.children) !=0:
        currentSplitIndex = tree.splitIndex
        currentSplitValue = tree.splitValue
        if entry[int(currentSplitIndex)] < currentSplitValue:
            tree = tree.children[0]
        else:
            tree = tree.children[1]

    tup = tree.tuple
    if tup[0] == tup[1]:
        a = 0
    if tup[0] > tup[1]:
        return 1
    else:
        return 0