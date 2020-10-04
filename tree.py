import numpy as np
from anytree import Node, RenderTree
import random

def tree_grow(x, y, nmin, minleaf, nfeat):
    for i in range(len(x)):
        x[i].append(y[i])
    goodNodes = 0
    for i in range(len(x)):
        goodNodes += x[i][len(x[i] - 1)]
    start = Node((goodNodes, len(y) - goodNodes, x))

    toDoNodes = [start]

    while len(toDoNodes) > 0:
        curNode = toDoNodes[0]
        del toDoNodes[0]

        if len(curNode.item2) < nmin:
            continue

        possibleSplits = random_unique_list(nfeat, len(curNode.item2))
        splitValues = []
        for i in range(len(possibleSplits)):
            splitValue = consider_splitt(curNode, possibleSplits[i])
            splitValues.append(splitValue)
        max = (-1, -1, -1)
        for i in range(len(splitValues)):
            if splitValues[i].item2 > max.item2:
                max = (splitValues[i], i)
        
        lList = []
        rList = []

        for i in range(len(x)):
            if x[i][max.item3] < max.item1:
                lList.append(x[i])
            else:
                rList.append(x[i])



        lNode = Node(((lList.count(1), lList.count(0)), lList), curNode)
        rNode = Node(((rList.count(1), rList.count(0)), rList), curNode)

        toDoNodes.append(lNode)
        toDoNodes.append(rNode)
        
        curNode = 0

    return blob

def tree_spliit(node):

    return 0

def consider_spliit(node, splitIndex):
    x = node.item2.copy()
    sorted(x, key=lambda z : z[splitIndex])
    splits = findSplits(x, splitIndex)
    iT = impurity(node)
    splitValues = []
    for i in splits:
        l = x[0:i]
        iL = impurity(l.count(1), l.count(0), l)
        piL = len(l) / len(x)
        r = x[i:]
        iR = impurity((r.count(1), r.count(0)), r)
        piR = len(r) / len(x)
        splitValues.append(iT - (piL * iL + piR * iR))
    max = (-1, -1)
    for i in range(len(splitValues)):
        if splitValues[i] > max.item2:
            max = (i, splitValues[i])

    return ((x[splits[max.item1]] + x[splits[max.item1] - 1] ) / 2, max.item2)

def findSplits(x, splitIndex):
    splits = []

    previous = x[0][-1]

    for i in range(1, len(x)):
        if previous != x[i][-1]:
            previous = x[i][-1]
            splits.append(i)
    return splits

def impurity(node):
    return (node.item1 / (node.item1 + node.item2)) * (node.item2 / (node.item1 + node.item2))


def random_unique_list(length, bound):
    result = np.empty(length)
    for i in range(length):
        next = random.randint(bound)
        while result.contains(next):
            next = random.randint(bound)
        result[i] = next
    return result

def tree_pred(x, tr):
    return 0

tree_grow(0,0,0,0,0)