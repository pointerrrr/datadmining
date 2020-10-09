import numpy as np
from anytree import AnyNode, RenderTree
import random

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
            ll, rl = getClassDistribution(lClassifier)
            toDoNodes.append(AnyNode(tuple=(ll, rl, np.array(lList), np.array(lClassifier)), parent=curNode))
        if len(rList) > 0:
            lr, rr = getClassDistribution(rClassifier)
            toDoNodes.append(AnyNode(tuple=(lr, rr, np.array(rList), np.array(rClassifier)), parent=curNode))
        if len(lList) != 0 or len(rList) != 0:
            curNode.splitIndex = currentSplitIndex
            curNode.splitValue = currentSplitValue

    return start

def tree_pred(x, tr):
    predictionList = []
    for i in x:
        predictionList.append(prediction(i, tr))
    return predictionList

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

def getClassDistribution(classifier):
    goodNodes = 0
    for i in classifier:
        goodNodes += i
    return (goodNodes, len(classifier) - goodNodes)

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

    splits = findSplits(x, splitIndex)
    if len(splits) == 0:
        return (-1,-1,-1)
    ln, rn = getClassDistribution(node.tuple[3])
    iT = impurity((ln,rn))
    splitValues = []
    for i in splits:
        #list for left node
        l = x[:i, 0:-1]
        #classifiers for left node
        lc = x[:i,-1]
        #class distribution for left node
        ld, rd = getClassDistribution(lc)
        #impurity of left node
        iL = impurity((ld, rd))
        piL = len(l) / len(x)
        #right node..
        r = x[i:, 0:-1]
        rc = x[i:, -1]
        lr, rr = getClassDistribution(rc)
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


#find all valid splits inside node
def findSplits(x, splitIndex):
    splits = []
    prev = x[0]
    for i in range(1,len(x)):
        if x[i][splitIndex] != prev[splitIndex]:
            splits.append(i)
            prev = x[i]

    return splits

def impurity(node):
    return (node[0] / (node[0] + node[1])) * (node[1] / (node[0] + node[1]))

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

def prediction(entry, currentNode):

    while len(currentNode.children) !=0:
        currentSplitIndex = currentNode.splitIndex
        currentSplitValue = currentNode.splitValue
        if entry[int(currentSplitIndex)] < currentSplitValue:
            currentNode = currentNode.children[0]
        else:
            currentNode = currentNode.children[1]

    tup = currentNode.tuple
    if tup[0] == tup[1]:
        a = 0
    if tup[0] > tup[1]:
        return 1
    else:
        return 0

c = np.loadtxt('eclipse-metrics-packages-2.0.csv', delimiter=';')
x, y = c[:,2] + c[:,4:], c[:,3].astype(int)