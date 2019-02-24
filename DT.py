import pandas as pd
import numpy as np
from pprint import pprint
from operator import itemgetter
import copy
            
# define the tree class
class Tree(object):
    def __init__(node):
        node.ltree = None
        node.rtree = None
        node.data = None  # to record the feature or the decision
        node.splitpoint =  None
    def printtree(node):
        if (node != None):
            print(node.data, node.splitpoint)
            if (node.ltree != None):
                print(str(node.splitpoint) +" left:")
                node.ltree.printtree()
            if (node.rtree != None):
                print(str(node.splitpoint) +" right:")
                node.rtree.printtree()
                
# entropy I(a,b,c)
def entropy(classes):
    classes,counts = np.unique(classes,return_counts = True)
    entropy = 0
    for i in range(len(classes)):
        entropy += -counts[i]/np.sum(counts)*np.log2(counts[i]/np.sum(counts))
    ##print(entropy)
    return entropy

# information gain
def infogain(data,feature,classname):
    classes = data[classname]
    beforetest = entropy(classes)
    feature_val,counts = np.unique(data[feature],return_counts = True)
    aftertest = 0.0
    for i in range(len(feature_val)):
        subclasses = data.where(data[feature] == feature_val[i]).dropna()
        aftertest += counts[i]/np.sum(counts)*entropy(subclasses[classname])
    info_gain = beforetest - aftertest
    ##print(info_gain)
    return info_gain

def splitpoints(data,features,classname):
    for feature in features:
        cols = data[[feature,classname]]
        originalcol = cols[feature]
        sortcols = cols.sort_values(feature)
        uniquev = np.unique(data[feature])
        # record all split points
        sp = []
        for i in range(0,len(uniquev)-1):
            value_i = uniquev[i]
            value_j = uniquev[i+1]  
            point = (value_i+value_j)/2
            sp.append(point)
        maxgain = 0
        maxgain_point = 0
        # compare the split points
        for point in sp:
            data[feature] = data[feature].map(lambda x: ("under"+str(point)) if x < point else ("obove"+str(point)))
            gain = infogain(data,feature,classname)
            data[feature] = originalcol
            if ( gain > maxgain):
                maxgain = gain
                maxgain_point = point
    ##print(dataset)
    return (maxgain_point,maxgain)

## ID3 algorithm
def ID3(examples,features,classname,parent_decision,tree):
    # If no examples left, return a leaf node with the majority decision of the examples in the parent node
    if len(examples)==0:
        tree.data = parent_decision    
    # If all examples belong to the same class i, return a leaf node with decision i
    elif len(np.unique(examples[classname])) <= 1:
        tree.data = np.unique(examples[classname])[0]
    # If no features left, return a leaf node with the majority decision of the examples
    elif len(features) ==0:
        majority_index = np.argmax(np.unique(examples[classname],return_counts=True)[1])
        tree.data = np.unique(examples[classname])[majority_index]
    # Else:
    else:
        # choose feature of maximum gain
        maxgain = 0
        maxgain_feature = features[0]
        maxgain_point = 0
        for feature in features:
            splitpoint, gain = splitpoints(examples,[feature],classname)
            if gain > maxgain:
                maxgain = gain
                maxgain_feature = feature
                maxgain_point = splitpoint
        # set up the major decision of the parent node(current node)
        majority_index = np.argmax(np.unique(examples[classname],return_counts=True)[1])
        parent_decision =  np.unique(examples[classname])[majority_index]
        # set up the tree
        tree.data = maxgain_feature
        tree.splitpoint = maxgain_point
        ##set up remaining features
        ##sub_features = [x for x in features if x != maxgain_feature]
        # loop for values of best feature
        ## less than the treshold value
        leftdata = examples.where(examples[maxgain_feature] < maxgain_point).dropna()
        tree.ltree = Tree()
        ID3(leftdata,features,classname,parent_decision,tree.ltree)
        ## more than the treshold value
        rightdata = examples.where(examples[maxgain_feature] >= maxgain_point).dropna()
        tree.rtree = Tree()
        ID3(rightdata,features,classname,parent_decision,tree.rtree)

def predict(data, root):
    demonimator = len(data)
    count = 0
    for index in range(0,len(data)):
        tree = copy.deepcopy(root)
        row = data.iloc[index]
        value = tree.data
        while (True):
            if (value == 0 or value == 1 or value == 2):
                break
            else:
                if (row[value] < tree.splitpoint):
                    tree = tree.ltree
                else:
                    tree = tree.rtree
                value = tree.data
        if (row['flowerclass'] == value):
            count += 1
    return (count/len(data))
    
    
def Q1():
    ## import dataset
    datasetA = pd.read_csv('set_a.csv',names=['sepalL','sepalW','petalL','petalW','flowerclass',])
    features = ['sepalL','sepalW','petalL','petalW']
    classname = 'flowerclass'
    root = Tree()
    ID3(datasetA,features,classname,1,root)
    root.printtree()
    p = predict(datasetA,root)
    print(p)
    ################  this is for part 3 #################
    ##datasetB = pd.read_csv('set_b.csv',names=['sepalL','sepalW','petalL','petalW','flowerclass',])
    ##p = predict(datasetB,root)
    ##print(p)    
Q1()