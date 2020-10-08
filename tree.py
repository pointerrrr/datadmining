import numpy as np
from anytree import AnyNode, RenderTree
import random


def tree_grow(x, y, nmin, minleaf, nfeat):
    newArray = np.array(x)
    ###for i in range(len(x)):
        ###print( np.append(x[i], y[i]))
        ###newArray[i] = np.concatenate(x[i], y[i])
    # Goodnodes --> class = 1
    goodNodes = 0
    for i in range(len(newArray)):
        goodNodes += newArray[i][-1]
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
            splitValues.append(splitValue)
        splitValues = sorted(splitValues, key=lambda tup: tup[1])
        splitValues.reverse()
        splitValues = np.array(splitValues)
        lList = []
        lClassifier = []
        rList = []
        rClassifier = []
        currentSplitValue = -1
        currentSplitIndex = -1
        for i in range(len(splitValues)):
            splitValue = splitValues[i]
            if splitValue[2] == -1:
                continue
            currentSplitValue = splitValue[0]
            currentSplitIndex = splitValue[2]
            lList = []
            rList = []
            for i in range(len(nodeArray)):
                if nodeArray[i][int(splitValue[2])] < splitValue[0]:
                    lList.append(nodeArray[i])
                    lClassifier.append(nodeClassifier[i])
                else:
                    rList.append(nodeArray[i])
                    rClassifier.append(nodeClassifier[i])
            if len(lList) >= minleaf and len(rList) >= minleaf:
                break

        if len(lList) < minleaf or len(rList) < minleaf:
                continue

        if len(lList) != 0:
            ll, rl = getClassDistribution(lList, lClassifier)
            toDoNodes.append(AnyNode(tuple=(ll, rl, np.array(lList), np.array(lClassifier)), parent=curNode))
        if len(rList) != 0:
            lr, rr = getClassDistribution(rList, rClassifier)
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

def getClassDistribution(lijst, classifier):
    goodNodes = 0
    for i in range(len(lijst)):
        goodNodes += classifier[i]
    return (goodNodes, len(lijst)-goodNodes)

def consider_split(node, splitIndex):
    splitIndex = int(splitIndex)
    x = np.array(node.tuple[2].copy())
    y = np.array(node.tuple[3].copy())

    x2 = np.empty((len(x), len(x[0]) + 1))

    for i in range(len(x)):
        for j in range(len(x[0]) + 1):
            if j == len(x[0]):
                x2[i,j] = y[i]
            else:
                x2[i,j] = x[i,j]
    x = x2
    ###print(splitIndex)
    x = np.array(sorted(x, key=lambda z : z[splitIndex]))
    splits = findSplits(x, splitIndex)
    if len(splits) == 0:
        return (-1,-1,-1)
    ln, rn = getClassDistribution(node.tuple[2], node.tuple[3])
    iT = impurity((ln,rn))
    splitValues = []
    for i in splits:
        l = x[:i, 0:-1]
        lc = x[:i,-1]
        ld, rd = getClassDistribution(l, lc)
        iL = impurity((ld, rd))
        piL = len(l) / len(x)
        r = x[i:, 0:-1]
        rc = x[i:, -1]
        lr, rr = getClassDistribution(r, rc)
        iR = impurity((lr, rr))
        piR = len(r) / len(x)
        splitValues.append(iT - ((piL * iL) + (piR * iR)))
    max = (-1, -1)
    for i in range(len(splitValues)):
        if splitValues[i] > max[1]:
            max = (i, splitValues[i])
    result = ((x[splits[max[0]]][splitIndex]+ x[splits[max[0]] - 1][splitIndex] ) / 2, max[1], splitIndex)
    return result



def findSplits(x, splitIndex):
    splits = []
    squashedList = []
    for i in range(0, len(x)):
        if len(squashedList) == 0 or x[i][splitIndex] != squashedList[-1][0]:
            tuple = (0,0)
            if(x[i][-1] == 0):
                tuple = (1,0)
            else:
                tuple = (0,1)
            squashedList.append([x[i][splitIndex], tuple])
        else:
            tuple = squashedList[-1][1]
            if(x[i][-1] == 0):
                tuple = (tuple[0] + 1, tuple[1])
            else:
                tuple = (tuple[0], tuple[1] + 1)
            squashedList[-1][1] = tuple 

    previous = squashedList[0]
    total = squashedList[0][1][0] + squashedList[0][1][1]
    for i in range(1, len(squashedList)):
        if (previous[1][0] / (previous[1][0] + previous[1][1])) != (squashedList[i][1][0] / (squashedList[i][1][0] + squashedList[i][1][1])):
            previous = squashedList[i]
            splits.append(total)
        total += squashedList[i][1][0] + squashedList[i][1][1]
    return splits

def impurity(node):
    return (node[0] / (node[0] + node[1])) * (node[1] / (node[0] + node[1]))


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
        raise Exception("Uhm... what to do?")
    if tup[0] > tup[1]:
        return 1
    else:
        return 0
    return 0