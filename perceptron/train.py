

import os
import numpy
import sys
from random import shuffle




""" Add features in a review into dictionary
    Combine each review with flag (1: positive, -1: negative)
    Add each review and its flag into list
"""
def get_vectors_and_set(file_name, features_set, flag):
    vectors = []
    data = open(file_name)
    features = [each_line for each_line in data]
    features = [each_line.split() for each_line in features]
    data.close()
    for each_line in features:
        features_set = features_set.union(set(each_line))
        feats = {}
        for word in each_line:
            feats[word] = 1
        vectors.append((feats,flag))
    return vectors, features_set




""" Training algorithm
"""
def percept_train(train_data, weights, bias):
    numErrors = 0

    for review in train_data:
        feat_set, y = review[0], review[1]
        activation = sum(weights[feat]*feat_set[feat] for feat in feat_set) 
        activation += bias 

        if ((y*activation) <= 0):
            numErrors += 1
            for feat in feat_set:
                weights[feat] = weights[feat] + y*feat_set[feat]
            bias += y 

    train_error = (float(numErrors)/len(train_data))*100

    return (weights, bias, train_error)




""" Testing algorithm
"""
def percept_test(test_data, weights, bias):
    totalCorrect = 0
    numErrors = 0

    for review in test_data:
        feat_set, y = review[0], review[1]
        activation = sum(weights[feat]*feat_set[feat] for feat in feat_set) 
        activation += bias 

        if (numpy.sign(activation) == y):
            totalCorrect += 1
        else:
            numErrors += 1

    test_error = (float(numErrors)/len(test_data))*100

    return test_error



if __name__ == "__main__":
    #Change the director to your own data folder
    os.chdir("C:\\Users\LiuXJ\Desktop\postgraduate\Data mining\Assignment 1\CA1data\data")
    
    #Open files and prepare the data
    features_set = set()
    train_pos, features_set = get_vectors_and_set("train.positive", features_set, 1)
    train_neg, features_set = get_vectors_and_set("train.negative", features_set, -1)
    test_pos, features_set = get_vectors_and_set("test.positive", features_set, 1)
    test_neg, features_set = get_vectors_and_set("test.negative", features_set, -1)
    
    train_data = train_pos + train_neg
    test_data = test_pos + test_neg
    
    #Initialize weights and bias
    features_set = list(features_set)
    weights = {}
    for each_feature in features_set:
        weights[each_feature] = 0
    bias = 0
    
    #Execute and print the result
    for i in range(15):
        shuffle(train_data) #Before each time of train, make the train data in random order
        weights, bias, train_error = percept_train(train_data, weights, bias)
        test_error = percept_test(test_data, weights, bias)
        print("Iteration ",i+1)
        print("Train error rate:", train_error,"%")
        print(" Test error rate:", test_error,"%")
        print ("---------------------------------------------------\n")





