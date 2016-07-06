
import os
import numpy
import sys
import random
import copy





"""Set features, vectors from the file. 
Labels_set is the set that contains all the eight types of label.
features_set is the set that contains all the features in all the reviews.
vectors is a list that contains 408 (the numbers of reviews) tuples. 
The first element in the tuple is a dictionary, its keys are the features in the review, and their value is 1.
The second element in the tuple is the label of the review.
vectors = [({feats:1},label), ( ), ( ), ...]"""
def get_features(file_name):
    file = open(file_name)
    data = [each_line for each_line in file]
    data = [each_line.split() for each_line in data]
    file.close()
    labels_set = set()
    features_set = set()
    vectors = []
    for each_line in data:
        labels_set.add(each_line[0])
        features = {}
        norm = numpy.sqrt(len(each_line)-1)
        for i in range(1, len(each_line)):
            features_set.add(each_line[i])
            features[each_line[i]] = 1/float(norm)
        vectors.append((features,each_line[0]))
    return labels_set, features_set, vectors


#Set a vector as center
def set_to_center(features_set, vector):
    center = {}.fromkeys(features_set, 0)
    for each_feat in vector:
        if each_feat in features_set:
            center[each_feat] = vector[each_feat]
    return center


"""Initialize clusters.
clusters is a list that contains k clusters.
A cluster is a list contain 3 elements.
The first element of a cluster is its center -- a dictionary.
The second element of a cluster is a list that contains all the vectors that belong to it.
The third element of a cluster is a float number -- the || ||2 of its centre.
clusters = [[{centre},[vectors],norm_square], [ ], [ ], ...]"""
def initialization(k, vectors, features_set):
    samples = random.sample(vectors, k)
    clusters = []
    for each_vector in samples:
        norm_square = 1#sum(each_vector[0][feat] * each_vector[0][feat] for feat in each_vector[0])
        center = set_to_center(features_set, each_vector[0])
        clusters.append([center,[],norm_square])
    return clusters



"""Calculate Euclidean distance.
norm_square is the || ||2 of the center.
The total features set has more than 4 thousand features, but a review (vector) has only two hundred.
In this way to calculate Euclidean distance we can avoid calculating many 0 features."""
def eucl_dist(vector, center, norm_square):
    dist = norm_square
    for each_feat in vector:
        dist -= center[each_feat] * center[each_feat]
        dist += (center[each_feat] - vector[each_feat]) * (center[each_feat] - vector[each_feat])
    return dist


#Assign each vector to its closest cluster center
def cluster_vectors(clusters, vectors):
    new_clusters = copy.deepcopy(clusters)
    for each_cluster in new_clusters:
        each_cluster[1] = []
    for each_vector in vectors:
        dist_list = []
        for each_cluster in new_clusters:
            norm_square = each_cluster[2]
            dist = eucl_dist(each_vector[0], each_cluster[0], norm_square)
            dist_list.append(dist)
        idx = dist_list.index(min(dist_list))
        #print(dist_list)
        new_clusters[idx][1].append(each_vector)
    flag = 1
    if new_clusters != clusters:
        flag = 0
    return new_clusters, flag
            

#Calculate the cluster center by mean (option 1)
def center_mean(clusters):
    for each_cluster in clusters:
        norm_square = 0
        num_vectors = len(each_cluster[1])
        each_cluster[0] = each_cluster[0].fromkeys(each_cluster[0], 0)
        for each_vector in each_cluster[1]:
            for each_feat in each_vector[0]:
                each_cluster[0][each_feat] += each_vector[0][each_feat]
        for each_feat in each_cluster[0]:
            each_cluster[0][each_feat] = each_cluster[0][each_feat]/float(num_vectors)
            norm_square += each_cluster[0][each_feat] * each_cluster[0][each_feat]
        each_cluster[2] = norm_square
    return clusters



#Choose the vector closest to the mean center as the center (option 2)
def center_vector(clusters,features_set):
    clusters = center_mean(clusters)
    for each_cluster in clusters:
        dist_list = []
        norm_square = each_cluster[2]
        for each_vector in each_cluster[1]:
            dist = eucl_dist(each_vector[0], each_cluster[0], norm_square)
            dist_list.append(dist)
        idx = dist_list.index(min(dist_list))
        each_cluster[0] = set_to_center(features_set, each_cluster[1][idx][0])
        each_cluster[2] = 1
    return clusters
        
        
#Train
def train(clusters, vectors, option, features_set):
    flag = 0
    iteration = 0
    while flag == 0:
        clusters, flag = cluster_vectors(clusters, vectors)
        if option == 1:
            clusters = center_mean(clusters)
        elif option ==2:
            clusters = center_vector(clusters, features_set)
        else:
            print("No such option")
            sys.exit()
        iteration += 1
    return clusters, iteration


#Find the key with the most value
def find_most_key(key_dict):
    most_key = ""
    num_most_key = 0
    for each_key in key_dict:
        if key_dict[each_key] > num_most_key:
            most_key = each_key
            num_most_key = key_dict[each_key]
    return most_key, num_most_key



#Calculate precistion
def calculate_precision(info_label, info_cluster, exist_label):
    precision = {}.fromkeys(exist_label, 0)
    for each_label in precision:
        precision[each_label] = float(info_label[each_label])/info_cluster[each_label]
    return precision



#Calculate recall
def calculate_recall(info_label, num_label, exist_label):
    recall = {}.fromkeys(exist_label, 0)
    for each_label in recall:
        recall[each_label] = float(info_label[each_label])/num_label[each_label]
    return recall



#Calculate F-score
def calculate_f_score(precision, recall, exist_label):
    f_score = {}.fromkeys(exist_label, 0)
    for each_label in f_score:
        f_score[each_label] = 2 * float(precision[each_label]) * recall[each_label] / (precision[each_label] + recall[each_label])
    return f_score



#Calculate average macro value
def calculate_macro(dictionary):
    macro = sum(dictionary[each_key] for each_key in dictionary) / float(len(dictionary))
    return macro



#Evaluate clusters
def evaluate_clusters(clusters, labels_set, vectors):
    num_label = {}.fromkeys(labels_set, 0)
    info_label = {}.fromkeys(labels_set, 0)
    info_cluster = {}.fromkeys(labels_set, 0)
    for each_vector in vectors:
        num_label[each_vector[1]] += 1
    for each_cluster in clusters:
        label_count = {}.fromkeys(labels_set, 0)
        for each_vector in each_cluster[1]:
            label_count[each_vector[1]] += 1
        most_label, num_most_label = find_most_key(label_count)
        info_label[most_label] += num_most_label
        info_cluster[most_label] += len(each_cluster[1])
        
    exist_label = set()
    for each_label in info_label:
        if info_label[each_label] != 0:
            exist_label.add(each_label)
    labels = len(exist_label)
    
    precision = calculate_precision(info_label, info_cluster, exist_label)
    recall = calculate_recall(info_label, num_label, exist_label)
    f_score = calculate_f_score(precision, recall, exist_label)
    return calculate_macro(precision), calculate_macro(recall), calculate_macro(f_score), labels


#plot graph
def plot_graph(prec_list, reca_list, f_list, k1, k2):
    from matplotlib import pyplot as plt
    plt.plot(range(k1, k2+1), prec_list, '-b', label='Macro Precision')
    plt.plot(range(k1, k2+1), reca_list, '-r', label='Macro Recall')
    plt.plot(range(k1, k2+1), f_list, '-g', label='Macro F-score')
    plt.axis([k1,k2,0,(max(max(prec_list), max(reca_list), max(f_list))+0.2)])
    plt.xlabel('Number of cluster')
    plt.ylabel('Score')
    plt.legend(loc='upper right')
    plt.show(block=True)



def process(k1, k2, file_name, option):
    labels_set, features_set, vectors = get_features(file_name)
    prec_list = []
    reca_list = []
    f_list = []
    for i in range(k1, k2+1):
        prec = []
        reca = []
        f_sc = []
        label = []
        itera = []
        for j in range(0, 10):
            debug_flag = 0
            while debug_flag == 0:
                clusters = initialization(i, vectors, features_set)
                debug_flag = 1
                debug_clusters, useless_flag = cluster_vectors(clusters, vectors)
                for each_debug_cluster in debug_clusters:
                    if len(each_debug_cluster[1]) == 0:
                        debug_flag = 0
            clusters, iteration = train(clusters, vectors, option, features_set)
            macro_precision, macro_recall, macro_f_score, labels = evaluate_clusters(clusters, labels_set, vectors)
            prec.append(macro_precision)
            reca.append(macro_recall)
            f_sc.append(macro_f_score)
            label.append(labels)
            itera.append(iteration)
        prec_list.append(numpy.mean(prec))
        reca_list.append(numpy.mean(reca))
        f_list.append(numpy.mean(f_sc))
        
        print("Clusters: %.2d  Labels: %.1f  Precision: %.3f  Recall: %.3f  F-score: %.3f  Iterations: %.1f" % (i, numpy.mean(label), numpy.mean(prec), numpy.mean(reca), numpy.mean(f_sc), numpy.mean(itera)))
    return prec_list, reca_list, f_list




#Execute
if __name__ == "__main__":    
    os.chdir("./data")#Change this to your own director
    k1 = 2
    k2 = 20
    print "\nPlease choose one of the following training options:"
    print "1. Select the mean in a cluster as the cluster centre"
    print "2. Select the instance that is closest to the mean as the cluster centre"
    option = int(raw_input())
    prec_list, reca_list, f_list = process(k1, k2, "CA2data.txt", option) # execute the main method
    plot_graph(prec_list, reca_list, f_list, k1, k2) 
