import numpy as np
from anytree import AnyNode, RenderTree
import random

counter = 1

def tree_grow(x, y, nmin, minleaf, nfeat):
    newArray = np.array(x)
    ###for i in range(len(x)):
        ###print( np.append(x[i], y[i]))
        ###newArray[i] = np.concatenate(x[i], y[i])
    # Goodnodes --> class = 1
    goodNodes = 0
    for i in range(len(newArray)):
        goodNodes += newArray[i][-1]
    start = AnyNode(tuple=(goodNodes, len(y) - goodNodes, x, y), splitIndex = -1, splitValue = -1, counter = 0)
    toDoNodes = [start]

    while len(toDoNodes) > 0:
        global counter
        if counter == 36:
            a = 1
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

        if len(splitValues) == 0:
            continue

        temp = []
        for i in range(len(splitValues)):
            temp.append(splitValues[i][1])
        length = len(np.unique(np.array(temp))) 
        if(length!= len(splitValues)):
            a = 0
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
                temp = nodeArray[i][int(splitValue[2])]
                temp2 = splitValue[0]
                if nodeArray[i][int(splitValue[2])] <= splitValue[0]:
                    lList.append(np.array(nodeArray[i]))
                    lClassifier.append(nodeClassifier[i])
                else:
                    rList.append(nodeArray[i])
                    rClassifier.append(np.array(nodeClassifier[i]))
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
            
            curNode.counter = counter
            counter += 1

    return start

def tree_pred(x, tr):
    predictionList = []
    for i in x:
        predictionList.append(prediction(i, tr))
    return predictionList

def getClassDistribution(classifier):
    goodNodes = 0
    for i in range(len(classifier)):
        goodNodes += classifier[i]
    return (goodNodes, len(classifier)-goodNodes)

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
    ln, rn = getClassDistribution(node.tuple[3])
    iT = impurity((ln,rn))
    splitValues = []
    for i in splits:
        l = x[:i, 0:-1]
        lc = x[:i,-1]
        ld, rd = getClassDistribution(lc)
        iL = impurity((ld, rd))
        piL = len(l) / len(x)
        r = x[i:, 0:-1]
        rc = x[i:, -1]
        lr, rr = getClassDistribution(rc)
        iR = impurity((lr, rr))
        piR = len(r) / len(x)
        splitValues.append(iT - ((piL * iL) + (piR * iR)))
    max = (-1, 0)
    for i in range(len(splitValues)):
        if splitValues[i] > max[1]:
            max = (i, splitValues[i])
    if max[0] == -1:
        return (-1,-1,-1)
    result = ((x[splits[max[0]]][splitIndex]+ x[splits[max[0]] - 1][splitIndex] ) / 2, max[1], splitIndex)
    return result



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
    return 0