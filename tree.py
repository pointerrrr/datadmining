import numpy as np
from anytree import AnyNode, RenderTree
import random


def tree_grow(x, y, nmin, minleaf, nfeat):
    newArray = np.empty(len(x))
    for i in range(len(x)):
        print( np.append(x[i], y[i]))
        newArray[i] = np.concatenate(x[i], y[i])
    goodNodes = 0
    for i in range(len(newArray)):
        goodNodes += newArray[i][-1]
    start = AnyNode(tuple=(goodNodes, len(y) - goodNodes, newArray), splitIndex = -1, splitValue = -1)
    toDoNodes = [start]

    while len(toDoNodes) > 0:
        curNode = toDoNodes[0]
        del toDoNodes[0]

        nodeArray = curNode.tuple[2]

        if len(curNode.tuple[2]) < nmin:
            continue

        possibleSplits = random_unique_list(nfeat, len(curNode.tuple[2][0] - 1))
        splitValues = []
        for i in range(len(possibleSplits)):
            splitValue = consider_split(curNode, possibleSplits[i])
            splitValues.append(splitValue)
        splitValues = sorted(splitValues, key=lambda tup: tup[1])
        splitValues.reverse()
        lList = []
        rList = []
        currentSplitValue = -1
        currentSplitIndex = -1
        for splitValue in splitValues:
            if splitValue[2] == -1:
                continue
            currentSplitValue = splitValue[0]
            currentSplitIndex = splitValue[2]
            lList = []
            rList = []
            for i in range(len(nodeArray)):
                if nodeArray[i][splitValue[2]] < splitValue[0]:
                    lList.append(nodeArray[i])
                else:
                    rList.append(nodeArray[i])
            if len(lList) >= minleaf and len(rList) >= minleaf:
                break

        if len(lList) != 0:
            ll, rl = getClassDistribution(lList)
            toDoNodes.append(AnyNode(tuple=(ll, rl, np.array(lList)), parent=curNode))
        if len(rList) != 0:
            lr, rr = getClassDistribution(rList)
            toDoNodes.append(AnyNode(tuple=(lr, rr, np.array(rList)), parent=curNode))
        if len(lList) != 0 or len(rList) != 0:
            curNode.splitIndex = currentSplitIndex
            curNode.splitValue = currentSplitValue
        
        curNode = 0

    return start

def getClassDistribution(lijst):
    goodNodes = 0
    for i in range(len(lijst)):
        goodNodes += lijst[i][-1]
    return (goodNodes, len(lijst)-goodNodes)

def consider_split(node, splitIndex):
    splitIndex = int(splitIndex)
    x = node.tuple[2].copy()
    print(splitIndex)
    x = sorted(x, key=lambda z : z[splitIndex])
    splits = findSplits(x, splitIndex)
    if len(splits) == 0:
        return (-1,-1,-1)
    ln, rn = getClassDistribution(node.tuple[2])
    iT = impurity((ln,rn))
    splitValues = []
    for i in splits:
        l = x[0:i]
        ld, rd = getClassDistribution(l)
        iL = impurity((ld, rd))
        piL = len(l) / len(x)
        r = x[i:]
        lr, rr = getClassDistribution(r)
        iR = impurity((lr, rr))
        piR = len(r) / len(x)
        splitValues.append(iT - (piL * iL + piR * iR))
    max = (-1, -1)
    for i in range(len(splitValues)):
        if splitValues[i] > max[1]:
            max = (i, splitValues[i])

    return ((x[splits[max[0]]][splitIndex]+ x[splits[max[0]] - 1][splitIndex] ) / 2, splitValues[max[0]], splitIndex)



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

def tree_pred(x, tr):
    predictionList = []
    for i in x:
        predictionList.append(prediction(i, tr))
    return predictionList

def prediction(entry, tree):
    currentNode = tree

    while currentNode !=0:
        if(len(currentNode.children) == 0):
            tup = currentNode.tuple
            if(tup[0] > tup[1]):
                return 1
            else:
                return 0
        else:
            currentSplitIndex = currentNode.splitIndex
            currentSplitValue = currentNode.splitValue
            if entry[currentSplitIndex] < currentSplitValue:
                currentNode = currentNode.children[0]
            else:
                currentNode = currentNode.children[1]
    return 0

x= np.array([
    [22,0,0,28,1],
    [46,0,1,32,0],
    [24,1,1,24,1],
    [25,0,0,27,1],
    [29,1,1,32,0],
    [45,1,1,30,0],
    [63,1,1,58,1],
    [36,1,0,52,1],
    [23,0,1,40,0],
    [50,1,1,28,0]
    ])

xx= [
    [22,0,0,28,1],
    [46,0,1,32,0],
    [24,1,1,24,1],
    [25,0,0,27,1],
    [29,1,1,32,0],
    [45,1,1,30,0],
    [63,1,1,58,1],
    [36,1,0,52,1],
    [23,0,1,40,0],
    [50,1,1,28,0]
    ]
# 32.5 | 0.5 | 0.5 | > 20 | 0.5
y = [0,0,0,0,0,1,1,1,1,1]
print(RenderTree(tree_grow(x,y,0,0,5)))

#print(tree_pred(xx, tree_grow(x,y,3,0,5)))

print("newTree!!!")
"""x= [
    [22,0,0,28,1],
    [46,0,1,32,0],
    [24,1,1,24,1],
    [25,0,0,27,1],
    [29,1,1,32,0],
    [45,1,1,30,0],
    [63,1,1,58,1],
    [36,1,0,52,1],
    [23,0,1,40,0],
    [50,1,1,28,0]
    ]"""
#print(RenderTree(tree_grow(x,y,3,0,5)))
#tree_grow(x,y,0,0,5)

"""[[22, 0, 0, 28, 1], 
 [46, 0, 1, 32, 0], 
 [25, 0, 0, 27, 1], 
 [23, 0, 1, 40, 0], 
 [24, 1, 1, 24, 1], 
 [29, 1, 1, 32, 0], 
 [45, 1, 1, 30, 0], 
 [63, 1, 1, 58, 1], 
 [36, 1, 0, 52, 1], 
 [50, 1, 1, 28, 0]]"""