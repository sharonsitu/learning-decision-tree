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
        node.data = None # to record the feature or the decision
        node.splitpoint =  None
        node.currentdepth = 0
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

## let's define 10 as the maxdepth we could go, and we start from 1
maxdepth = 10
def ID3(examples,features,classname,parent_decision,tree):
    # If no examples left, return a leaf node with the majority decision of the examples in the parent node
    if len(examples) == 0:
        tree.data = parent_decision    
    # If all examples belong to the same class i, return a leaf node with decision i
    elif len(np.unique(examples[classname])) <= 1:
        tree.data = np.unique(examples[classname])[0]
    # If no features left, return a leaf node with the majority decision of the examples
    elif len(features) == 0 or (tree.currentdepth + 1) > maxdepth:
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
        # loop for values of best feature
        ## less than the treshold value
        leftdata = examples.where(examples[maxgain_feature] < maxgain_point).dropna()
        tree.ltree = Tree()
        tree.ltree.currentdepth = tree.currentdepth + 1
        ID3(leftdata,features,classname,parent_decision,tree.ltree)
        ## more than the treshold value
        rightdata = examples.where(examples[maxgain_feature] >= maxgain_point).dropna()
        tree.rtree = Tree()
        tree.rtree.currentdepth = tree.currentdepth + 1
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

def ten_fold(data,features,classname):
    datasize = len(data)
    subsetsize = datasize/10
    # try different depth
    for i in range (0,10):
        global maxdepth
        maxdepth = i+1
        # shuffle the data
        data = data.sample(frac=1).reset_index(drop=True)
        # try 10 learning
        validation_acc_sum = 0
        training_acc_sum = 0  
        for j in range(0,10):
            left = 10*j
            right = left+9
            validation = data.loc[left:right,:]
            training = data.loc[:left-1,:].append(data.loc[right+1:,])
            mytree = Tree()
            majority_index = np.argmax(np.unique(data[classname],return_counts=True)[1])
            majority_decision = np.unique(data[classname])[majority_index]
            ID3(training,features,classname,majority_decision,mytree)
            ##mytree.printtree()
            validation_acc = predict(validation,mytree)
            training_acc = predict(training,mytree)
            ##print(validation_error)
            ##print(training_error)
            validation_acc_sum += validation_acc
            training_acc_sum += training_acc
        avg_valid_acc = round(validation_acc_sum/10,2)
        avg_train_acc = round(training_acc_sum/10,2)
        print(str(maxdepth)+":")
        print(avg_valid_acc)
        print(avg_train_acc)

    
    
def Q2():
    ## import dataset
    datasetA = pd.read_csv('set_a.csv',names=['sepalL','sepalW','petalL','petalW','flowerclass',])
    features = ['sepalL','sepalW','petalL','petalW']
    classname = 'flowerclass'
    ten_fold(datasetA,features,classname)
    ## now we could see that maxdepth 3 having hightest prediction accuracy
    global maxdepth
    maxdepth = 3
    mytree = Tree()
    majority_index = np.argmax(np.unique(datasetA[classname],return_counts=True)[1])
    majority_decision = np.unique(datasetA[classname])[majority_index]
    ID3(datasetA,features,classname,majority_decision,mytree)
    mytree.printtree()
    acc = predict(datasetA,mytree)
    print(acc)
    ################  this is for part 3 #################
    ##datasetB = pd.read_csv('set_b.csv',names=['sepalL','sepalW','petalL','petalW','flowerclass',])
    ##acc = predict(datasetB,mytree)
    ##print(acc)
    
Q2()