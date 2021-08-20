#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 19:27:59 2019

@author: ffm5105
"""

import os
import pickle
import numpy as np
import rnn_tools
import torch
from torch.autograd import Variable
import time
import random
import rnn_model
import shutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def train_model(training_file = 'training_file',
                validation_file = 'validation_file',
                testing_file = 'testing_file',
                n_diagnosis_codes = 10000,
                n_labels = 2,
                output_file = 'output_file',
                batch_size = 100,
                dropout_rate = 0.9,
                L2_reg = 0.001,
                n_epoch = 1000,
                log_eps = 1e-8,
                visit_size = 256,
                hidden_size = 256,
                use_gpu = False,
                model_name = '',
                running_data = ''):
    options = locals().copy()

    print('building the model ...')
    if model_name == 'gru' or model_name == 'gru2' or model_name == 'gru3':
        rnn = rnn_model.GRU(options)
    if model_name == 'lstm':
        rnn = rnn_model.LSTM(options)
    if use_gpu:
        rnn = rnn.cuda()
    focal_loss = torch.nn.functional.cross_entropy
    rnn.train()
    print('constructing the optimizer ...')
    optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-4, weight_decay = options['L2_reg'])
    print('done!')

    print('loading data ...')
    train, validate, test= rnn_tools.load_data(training_file, validation_file, testing_file)
    n_batches = int(np.ceil(float(len(train[0])) / float(batch_size)))
    
    print('training start')
    best_train_cost = 0.0
    best_validate_cost = 100000000.0
    best_test_cost = 0.0
    epoch_duaration = 0.0
    best_epoch = 0.0
    best_parameters_file = ''

    for epoch in range(n_epoch):
        iteration = 0
        cost_vector = []
        start_time = time.time()
        samples = random.sample(range(n_batches), n_batches)
        counter = 0
        
        for index in samples:
            batch_diagnosis_codes = train[0][batch_size * index : batch_size * (index + 1)]
            for ind in range(len(batch_diagnosis_codes)):
                if len(batch_diagnosis_codes[ind]) > 50:
                    batch_diagnosis_codes[ind] = batch_diagnosis_codes[ind][-50:]
            batch_labels = train[1][batch_size * index : batch_size * (index + 1)]
            t_diagnosis_codes, t_labels, t_mask = rnn_tools.pad_matrix(batch_diagnosis_codes, batch_labels, options)
            
            if use_gpu:
                t_diagnosis_codes = Variable(torch.FloatTensor(t_diagnosis_codes).cuda())
                t_labels = Variable(torch.LongTensor(t_labels).cuda())
                t_mask = Variable(torch.FloatTensor(t_mask).cuda())
            else:
                t_diagnosis_codes = Variable(torch.FloatTensor(t_diagnosis_codes))
                t_labels = Variable(torch.LongTensor(t_labels))
                t_mask = Variable(torch.FloatTensor(t_mask))
            
            optimizer.zero_grad()
            logit = rnn(t_diagnosis_codes, t_mask)
                
            loss = focal_loss(logit, t_labels)
            loss.backward()
            optimizer.step()
            
            cost_vector.append(loss.cpu().data.numpy())
            
            if (iteration % 50 == 0):
                print('epoch:%d, iteration:%d/%d, cost:%f' % (epoch, iteration, n_batches, loss.cpu().data.numpy()))
            iteration += 1
        
        duration = time.time() - start_time
        print('epoch:%d, mean_cost:%f, duration:%f' % (epoch, np.mean(cost_vector), duration))

        train_cost = np.mean(cost_vector)
        validate_cost = rnn_tools.calculate_cost(rnn, validate, options, focal_loss)
        test_cost = rnn_tools.calculate_cost(rnn, test, options, focal_loss)
        epoch_duaration += duration
        
        if validate_cost < best_validate_cost:
            best_validate_cost = validate_cost
            best_train_cost = train_cost
            best_test_cost = test_cost
            best_epoch = epoch
            
            shutil.rmtree(output_file)
            os.mkdir(output_file)
            
            torch.save(rnn.state_dict(), output_file + model_name + '.' + str(epoch))
            best_parameters_file = output_file + model_name + '.' + str(epoch)
        buf = 'Best Epoch:%d, Train_Cost:%f, Valid_Cost:%f, Test_Cost:%f' % (best_epoch, best_train_cost, best_validate_cost, best_test_cost)
        print(buf)
        
    # testing
    print(best_parameters_file)
    rnn.load_state_dict(torch.load(best_parameters_file))
    rnn.eval()
    n_batches = int(np.ceil(float(len(test[0])) / float(batch_size)))
    y_true = np.array([])
    y_pred = np.array([])
    for index in range(n_batches):
        batch_diagnosis_codes = test[0][batch_size * index : batch_size * (index + 1)]
        for ind in range(len(batch_diagnosis_codes)):
            if len(batch_diagnosis_codes[ind]) > 50:
                batch_diagnosis_codes[ind] = batch_diagnosis_codes[ind][-50:]
        batch_labels = test[1][batch_size * index : batch_size * (index + 1)]
        t_diagnosis_codes, t_labels, t_mask = rnn_tools.pad_matrix(batch_diagnosis_codes, batch_labels, options)
        
        if use_gpu:
            t_diagnosis_codes = Variable(torch.FloatTensor(t_diagnosis_codes).cuda())
            t_mask = Variable(torch.FloatTensor(t_mask).cuda())
        else:
            t_diagnosis_codes = Variable(torch.FloatTensor(t_diagnosis_codes))
            t_mask = Variable(torch.FloatTensor(t_mask))
        
        logit = rnn(t_diagnosis_codes, t_mask)
                
        if use_gpu:
            prediction = torch.max(logit, 1)[1].view((len(t_labels),)).data.cpu().numpy()
        else:
            prediction = torch.max(logit, 1)[1].view((len(t_labels),)).data.numpy()
        
        y_true = np.concatenate((y_true, t_labels))
        y_pred = np.concatenate((y_pred, prediction))
        
    accuary = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    print(accuary, precision, recall, f1, roc_auc)
    return (accuary, precision, recall, f1, roc_auc)

if __name__ == '__main__':
    # parameters
    batch_size = 50
    dropout_rate = 0.5
    L2_reg = 0.001
    log_eps = 1e-8
    n_epoch = 20
    n_labels = 2 # binary classification
    visit_size = 256
    hidden_size = 256
    
    use_gpu = True
    model_name = 'gru'
    disease_list = ['copd', 'hf', 'kidney']
    for disease in disease_list:
    
        path = 'data/'+disease+'/model_inputs/'
        trianing_file = path + disease + '_training_new.pickle'
        validation_file = path + disease + '_validation_new.pickle'
        testing_file = path + disease + '_testing_new.pickle'

        dict_file = 'data/' + disease + '/' + disease + '_code2idx_new.pickle'
        n_diagnosis_codes = len(pickle.load(open(dict_file, 'rb')))

        output_file_path = model_name + '_outputs/'
        if os.path.isdir(output_file_path):
            pass
        else:
            os.mkdir(output_file_path)
        results = []
        for k in range(5):
            accuary, precision, recall, f1, roc_auc = train_model(trianing_file, validation_file,
                                               testing_file, n_diagnosis_codes, n_labels,
                                               output_file_path, batch_size, dropout_rate,
                                               L2_reg, n_epoch, log_eps, visit_size, hidden_size,
                                               use_gpu, model_name)
            results.append([accuary, precision, recall, f1, roc_auc])
        results = np.array(results)
        print(np.mean(results, 0))
