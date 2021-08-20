#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 18:44:43 2019

@author: ffm5105
"""

import numpy as np
import pickle
import os
import random

def load_data(visit_file, label_file, time_file):
    claims = np.array(pickle.load(open(visit_file, 'rb')))
    labels = np.array(pickle.load(open(label_file, 'rb')))
    times = np.array(pickle.load(open(time_file, 'rb')))
    print(len(labels), np.sum(labels), len(labels)-np.sum(labels) )
    
    num = int(random.uniform(1, 10000))
    np.random.seed(num)
    data_size = len(claims)
    ind = np.random.permutation(data_size)
    #print(ind[:10])
    
    n_test = int(0.15 * data_size)
    n_validate = int(0.1 * data_size)
    
    test_indices = ind[:n_test]
    validate_indices = ind[n_test:n_test + n_validate]
    train_indices = ind[n_test + n_validate:]
    
    train_claims = claims[train_indices]
    train_labels = labels[train_indices]
    train_times = times[train_indices]

    test_claims = claims[test_indices]
    test_labels = labels[test_indices]
    test_times = times[test_indices]

    validate_claims = claims[validate_indices]
    validate_labels = labels[validate_indices]
    validate_times = times[validate_indices]

    def len_argsort(seq):
        return sorted(range(len(seq)), key = lambda x:len(seq[x]))
        
    train_sorted_index = len_argsort(train_claims)
    train_claims = [train_claims[i] for i in train_sorted_index]
    train_labels = [train_labels[i] for i in train_sorted_index]
    train_times = [train_times[i] for i in train_sorted_index]

    test_sorted_index = len_argsort(test_claims)
    test_claims = [test_claims[i] for i in test_sorted_index]
    test_labels = [test_labels[i] for i in test_sorted_index]
    test_times = [test_times[i] for i in test_sorted_index]

    validate_sorted_index = len_argsort(validate_claims)
    validate_claims = [validate_claims[i] for i in validate_sorted_index]
    validate_labels = [validate_labels[i] for i in validate_sorted_index]
    validate_times = [validate_times[i] for i in validate_sorted_index]

    train_set = (train_claims, train_labels, train_times)
    validate_set = (validate_claims, validate_labels, validate_times)
    test_set = (test_claims, test_labels, test_times)
    
    print('# train:', sum(train_labels), len(train_labels))
    print('# test:', sum(test_labels), len(test_labels))
    print('# validatiob:', sum(validate_labels), len(validate_labels))
    
    return train_set, validate_set, test_set

path = './data/'
diseases = ['hf', 'copd', 'kidney', 'amnesia', 'dementias']

for d in diseases:
    print(d)
    visit_file = path + d + '/' + d + '_visits_new.pickle'
    label_file = path + d + '/' + d + '_labels_new.pickle'
    time_file = path + d + '/' + d + '_times_new.pickle'
    train_set, validate_set, test_set = load_data(visit_file, label_file, time_file)
    
    training = path + d + '/model_inputs/' + d + '_training_new.pickle'
    validation = path + d + '/model_inputs/' + d + '_validation_new.pickle'
    testing = path + d + '/model_inputs/' + d + '_testing_new.pickle'
    
    if os.path.isdir(path + d + '/model_inputs'):
        pass
    else:
        os.mkdir(path + d + '/model_inputs')
    
    with open(training, 'wb') as f:
        pickle.dump(train_set, f, protocol=0)
    with open(validation, 'wb') as f:
        pickle.dump(validate_set, f, protocol=0)
    with open(testing, 'wb') as f:
        pickle.dump(test_set, f, protocol=0)