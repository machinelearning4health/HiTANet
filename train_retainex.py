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
from models.retainEx import RETAIN_EX
import shutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.multiprocessing import Process
import traceback


def prefetch_data(samples, train, queue, batch_size, max_len, n_diagnosis_codes):
    ind = 0
    print("start prefetching data...")
    np.random.seed(os.getpid())
    while ind < len(samples):
        try:
            index = samples[ind]
            data = get_data(train, index, batch_size, max_len, n_diagnosis_codes)
            queue.put(data)
            ind += 1
        except Exception as e:
            traceback.print_exc()
            raise e

def init_parallel_jobs(dbs, samples, train, queue, batch_size, max_len, n_diagnosis_codes):
    tasks = [Process(target=prefetch_data, args=(samples, train, queue, batch_size, max_len, n_diagnosis_codes)) for k in range(dbs)]
    for task in tasks:
        task.daemon = True
        task.start()
    return tasks

def train_model(training_file='training_file',
                validation_file='validation_file',
                testing_file='testing_file',
                n_diagnosis_codes=10000,
                n_labels=2,
                output_file='output_file',
                batch_size=100,
                dropout_rate=0.5,
                L2_reg=0.001,
                n_epoch=1000,
                log_eps=1e-8,
                visit_size=512,
                hidden_size=256,
                use_gpu=False,
                model_name='',
                disease = 'hf',
                code2id = None,
                running_data='',
                max_code = 50,
                gamma=0.5):
    options = locals().copy()
    print('building the model ...')
    model = RETAIN_EX(n_diagnosis_codes, hidden_size, 2, True, True, 1)
    focal_loss = torch.nn.functional.cross_entropy
    print('constructing the optimizer ...')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay = options['L2_reg'])
    print('done!')

    print('loading data ...')
    train, validate, test = rnn_tools.load_data(training_file, validation_file, testing_file)
    n_batches = int(np.ceil(float(len(train[0])) / float(batch_size)))

    print('training start')
    best_train_cost = 0.0
    best_validate_cost = 100000000.0
    best_test_cost = 0.0
    epoch_duaration = 0.0
    best_epoch = 0.0
    max_len = 50
    best_parameters_file = ''
    if use_gpu:
        model.cuda()
    model.train()
    for epoch in range(n_epoch):
        iteration = 0
        cost_vector = []
        start_time = time.time()
        samples = random.sample(range(n_batches), n_batches)
        counter = 0
        #training_queue = Queue(30)
        #training_tasks = init_parallel_jobs(1, samples, train, training_queue, batch_size, max_len)
        for index in samples:
            batch_diagnosis_codes = train[0][batch_size * index: batch_size * (index + 1)]
            batch_time_step = train[2][batch_size * index: batch_size * (index + 1)]
            batch_labels = train[1][batch_size * index: batch_size * (index + 1)]
            for ind in range(len(batch_diagnosis_codes)):
                if len(batch_diagnosis_codes[ind]) > 50:
                    batch_diagnosis_codes[ind] = batch_diagnosis_codes[ind][-50:]
                    batch_time_step[ind] = batch_time_step[ind][-50:]
            lengths = np.array([len(seq) for seq in batch_diagnosis_codes])
            maxlen = np.max(lengths)
            #[batch_diagnosis_codes, batch_time_step, batch_labels, lengths, maxlen] = training_queue.get(block=True)
            t_diagnosis_codes, t_labels, t_mask, t_time, t_mask_final = rnn_tools.pad_matrix_retainEx(batch_diagnosis_codes, batch_labels, batch_time_step, options)
            if use_gpu:
                t_diagnosis_codes = Variable(torch.LongTensor(t_diagnosis_codes).cuda())
                t_labels = Variable(torch.LongTensor(t_labels).cuda())
                t_mask = Variable(torch.FloatTensor(t_mask).cuda())
                t_mask_final = Variable(torch.FloatTensor(t_mask_final).cuda())
                t_time = Variable(torch.FloatTensor(t_time).cuda())
            else:
                t_diagnosis_codes = Variable(torch.LongTensor(t_diagnosis_codes))
                t_labels = Variable(torch.LongTensor(t_labels))
                t_mask = Variable(torch.FloatTensor(t_mask))
                t_mask_final = Variable(torch.FloatTensor(t_mask_final))
                t_time = Variable(torch.FloatTensor(t_time))
            optimizer.zero_grad()
            predictions = model([t_diagnosis_codes, t_mask, t_time, t_mask_final], 1)
            loss = focal_loss(predictions, t_labels)
            loss.backward()
            optimizer.step()

            cost_vector.append(loss.cpu().data.numpy())

            if (iteration % 50 == 0):
                print('epoch:%d, iteration:%d/%d, cost:%f' % (epoch, iteration, n_batches, loss.cpu().data.numpy()))
                #print(self_attention[:,0,0].squeeze().cpu().data.numpy())
                #print(time_weight[:, 0])
                #print(prior_weight[:, 0])
                #print(model.time_encoder.time_weight[0:10])
                #print(self_weight[:, 0])
            iteration += 1

        duration = time.time() - start_time
        print('epoch:%d, mean_cost:%f, duration:%f' % (epoch, np.mean(cost_vector), duration))

        train_cost = np.mean(cost_vector)
        validate_cost = rnn_tools.calculate_cost_retainEx(model, validate, options, max_len, focal_loss)
        test_cost = rnn_tools.calculate_cost_retainEx(model, test, options, max_len, focal_loss)
        print('epoch:%d, validate_cost:%f, duration:%f' % (epoch, validate_cost, duration))
        epoch_duaration += duration

        train_cost = np.mean(cost_vector)
        epoch_duaration += duration

        if validate_cost < best_validate_cost:
            best_validate_cost = validate_cost
            best_train_cost = train_cost
            best_test_cost = test_cost
            best_epoch = epoch

            shutil.rmtree(output_file)
            os.mkdir(output_file)

            torch.save(model.state_dict(), output_file + model_name + '.' + str(epoch))
            best_parameters_file = output_file + model_name + '.' + str(epoch)
        buf = 'Best Epoch:%d, Train_Cost:%f, Valid_Cost:%f, Test_Cost:%f' % (
        best_epoch, best_train_cost, best_validate_cost, best_test_cost)
        print(buf)
        #for training_task in training_tasks:
        #    training_task.terminate()
    # testing
    #best_parameters_file = output_file + model_name + '.' + str(8)
    model.load_state_dict(torch.load(best_parameters_file))
    model.eval()
    n_batches = int(np.ceil(float(len(test[0])) / float(batch_size)))
    y_true = np.array([])
    y_pred = np.array([])
    for index in range(n_batches):
        batch_diagnosis_codes = test[0][batch_size * index: batch_size * (index + 1)]
        batch_time_step = test[2][batch_size * index: batch_size * (index + 1)]
        batch_labels = test[1][batch_size * index: batch_size * (index + 1)]
        for ind in range(len(batch_diagnosis_codes)):
            if len(batch_diagnosis_codes[ind]) > 50:
                batch_diagnosis_codes[ind] = batch_diagnosis_codes[ind][-50:]
                batch_time_step[ind] = batch_time_step[ind][-50:]
        lengths = np.array([len(seq) for seq in batch_diagnosis_codes])
        maxlen = np.max(lengths)
        # [batch_diagnosis_codes, batch_time_step, batch_labels, lengths, maxlen] = training_queue.get(block=True)
        t_diagnosis_codes, t_labels, t_mask, t_time, t_mask_final = rnn_tools.pad_matrix_retainEx(batch_diagnosis_codes, batch_labels,
                                                                                batch_time_step, options)
        if use_gpu:
            t_diagnosis_codes = Variable(torch.LongTensor(t_diagnosis_codes).cuda())
            t_mask = Variable(torch.FloatTensor(t_mask).cuda())
            t_mask_final = Variable(torch.FloatTensor(t_mask_final).cuda())
            t_time = Variable(torch.FloatTensor(t_time).cuda())
        else:
            t_diagnosis_codes = Variable(torch.LongTensor(t_diagnosis_codes))
            t_mask = Variable(torch.FloatTensor(t_mask))
            t_mask_final = Variable(torch.FloatTensor(t_mask_final))
            t_time = Variable(torch.FloatTensor(t_time))
        logit = model([t_diagnosis_codes, t_mask, t_time, t_mask_final], None)
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
    L2_reg = 1e-3
    log_eps = 1e-8
    n_epoch = 30
    n_labels = 2  # binary classification
    visit_size = 256
    hidden_size = 128
    gamma = 0.5

    use_gpu = True
    disease_list = ['copd', 'hf', 'kidney']
    for disease in disease_list:
        model_name = 'retainx_L4_wt_1e-4_focal%.2f' %(gamma)
        print(model_name)
        path = 'data/'+disease+'/model_inputs/'
        trianing_file = path + disease + '_training_new.pickle'
        validation_file = path + disease + '_validation_new.pickle'
        testing_file = path + disease + '_testing_new.pickle'

        dict_file = 'data/' + disease + '/' + disease + '_code2idx_new.pickle'
        code2id = pickle.load(open(dict_file, 'rb'))
        n_diagnosis_codes = len(pickle.load(open(dict_file, 'rb')))+1

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
                                                                  use_gpu, model_name, disease=disease, code2id=code2id, gamma=gamma)
            results.append([accuary, precision, recall, f1, roc_auc])
        results = np.array(results)
        print(np.mean(results, 0))

