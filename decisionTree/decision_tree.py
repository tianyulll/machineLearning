import csv
import sys
import numpy as np

class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.attr = None
        self.vote = None

def stopCondition(data):
    if len(data) == 0 or len(data[-1]) == 0:
        return True
    if data[-1].count(1) == len(data) or data[-1].count(0) == len(data):
        return True
    for i in range(len(data)-1):
        if data[i].count(1) != 0 and data[i].count(1) != len(data[i]):
            return False
    return True

# Determine the label at leaf
def majorityVote(data):
    label=data[-1]
    if label.count(1) >= label.count(0):
        return 1
    else:
        return 0

def entropy(data):
    one = data.count(1)/len(data)
    zero = data.count(0)/len(data)
    entropy = -(one*np.log2(one) + zero*np.log2(zero))
    return one,zero, entropy

# calculates the mutual information for 1 attribute
def splitHelper(xlist, label):
    prob1, prob0, _ = entropy(xlist)
    pos1 = np.where(np.array(xlist) == 1)[0]
    pos0 = np.where(np.array(xlist) == 0)[0]
    y0 = [label[i] for i in pos0]
    y1 = [label[i] for i in pos1]
    if len(y0) == 0: y00, y01 = 0,0
    else: y00, y01 = y0.count(0)/len(y0), y0.count(1)/len(y0)
    if len(y1) == 0: y10, y11 = 0, 0
    else: y10, y11 = y1.count(0)/len(y1), y1.count(1)/len(y1)
    return prob0*(-y00*np.log2(y00) - y01*np.log2(y01)) \
         + prob1*(-y10*np.log2(y10) - y11*np.log2(y11))

# split criterion = mutual information
# finds the attribute of max mutual info
def splitMutual(data):
    _,_,hy = entropy(data[-1])
    mutualInfoList = []
    for i in range(len(data)-1):
       mutualI = splitHelper(data[i], data[-1])
       print("calculated:", mutualI)
       mutualInfoList.append(hy - mutualI)
    return mutualInfoList.index(max(mutualInfoList))

# Partitioin Dataset to D_left, D_right
def splitData(data):
    x = splitMutual(data)
    xlist = data[x]
    pos1 = np.where(np.array(xlist) == 1)[0]
    pos0 = np.where(np.array(xlist) == 0)[0]
    left = [[l[i] for i in pos0] for l in data]
    right = [[l[i] for i in pos1] for l in data]
    return left, right

# train with data, return with root
def trainModel(data, depth, counter):
    p = Node()
    counter += 1
    if stopCondition(data) or counter >= depth:
        print("stopped")
        p.vote=majorityVote(data)
        return p
    else:
        left, right = splitData(data)
        p.left = trainModel(left, depth, counter)
        p.right = trainModel(right, depth, counter)
        return p

def predict(root, data):
    pass


# dict stores attributes as keys 
# and values as a list 
def parseInput(fileName):
    file = csv.reader(open(fileName, 'r'), delimiter="\t")
    columns = list(zip(*file))
    attributeName=[]
    attributeData=[]
    for col in columns:
        attributeName.append(col[0])
        attributeData.append([int(i) for i in col[1:]])
    return attributeName, attributeData

if __name__ == '__main__':
    train = sys.argv[1]
    test = sys.argv[2]
    maxdepth = sys.argv[3]
    # train_out = sys.argv[4]
    # test_out = sys.argv[5]
    # metrics = sys.argv[6]

    _, train_data = parseInput(train)
    _, test_data = parseInput(test)
    root = trainModel(train_data, int(maxdepth), 0)
    