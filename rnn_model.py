#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 19:27:59 2019

@author: ffm5105
"""
import torch
import torch.nn as nn
from torch.autograd import Variable

class GRU(nn.Module):
    
    def __init__(self, options):
        super(GRU,self).__init__()
        self.options = options
        
        n_diagnosis_codes = options['n_diagnosis_codes']
        visit_size = options['visit_size']
        hidden_size = options['hidden_size']
        
        n_labels = options['n_labels']
        dropout_rate = options['dropout_rate']
        
        self.gpu = options['use_gpu']
        
        # code embedding layer
        self.embed = nn.Linear(n_diagnosis_codes, visit_size)
        
        # relu layer
        self.relu = nn.ReLU()
        
        # gru layer
        self.gru = nn.GRU(visit_size, hidden_size, num_layers = 1, batch_first = False)
        
        # dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # fully connected layer
        self.fc = nn.Linear(hidden_size, n_labels)
        
        # softmax
        self.softmax = nn.Softmax()


    def forward(self, x, mask):
        x = self.embed(x) # (n_visits, n_samples, visit_size)
        x = self.relu(x)
        
        if self.gpu:
            h0 = Variable(torch.FloatTensor(torch.zeros(1, x.size()[1], x.size()[2])).cuda())
        else:
            h0 = Variable(torch.FloatTensor(torch.zeros(1, x.size()[1], x.size()[2])))
        output, h_n = self.gru(x, h0) # output (seq_len, batch, hidden_size)
                                         #h_n (num_layers, batch, hidden_size)
        mask = mask.unsqueeze(-1).expand_as(output) 
        
        x = output * mask
        x = x.sum(0)

        x = self.dropout(x) # (n_samples, hidden_size)
        logit = self.fc(x) # (n_samples, n_labels)
        return logit


class GRUSelfAttention(nn.Module):

    def __init__(self, options):
        super(GRUSelfAttention, self).__init__()
        self.options = options

        n_diagnosis_codes = options['n_diagnosis_codes']
        visit_size = options['visit_size']
        hidden_size = options['hidden_size']

        n_labels = options['n_labels']
        dropout_rate = options['dropout_rate']

        self.gpu = options['use_gpu']

        # code embedding layer
        self.embed = nn.Linear(n_diagnosis_codes, visit_size)

        # relu layer
        self.relu = nn.ReLU()

        # gru layer
        self.gru = nn.GRU(visit_size, hidden_size, num_layers=1, batch_first=False)

        # dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # fully connected layer
        self.fc = nn.Linear(hidden_size, n_labels)
        self.weight_layer = torch.nn.Linear(hidden_size, 1)

        # softmax
        self.softmax = nn.Softmax()

    def forward(self, x, mask):
        x = self.embed(x)  # (n_visits, n_samples, visit_size)
        x = self.relu(x)

        if self.gpu:
            h0 = Variable(torch.FloatTensor(torch.zeros(1, x.size()[1], x.size()[2])).cuda())
        else:
            h0 = Variable(torch.FloatTensor(torch.zeros(1, x.size()[1], x.size()[2])))
        output, h_n = self.gru(x, h0)  # output (seq_len, batch, hidden_size)
        weight = torch.softmax(self.weight_layer(output), 0)
        output = output * weight
        # h_n (num_layers, batch, hidden_size)
        mask = mask.unsqueeze(-1).expand_as(output)

        x = output * mask
        x = x.sum(0)

        x = self.dropout(x)  # (n_samples, hidden_size)
        logit = self.fc(x)  # (n_samples, n_labels)
        return logit

class LSTM(nn.Module):
    
    def __init__(self, options):
        super(LSTM,self).__init__()
        self.options = options
        
        n_diagnosis_codes = options['n_diagnosis_codes']
        visit_size = options['visit_size']
        hidden_size = options['hidden_size']
        
        n_labels = options['n_labels']
        dropout_rate = options['dropout_rate']
        
        self.gpu = options['use_gpu']
        
        # code embedding layer
        self.embed = nn.Linear(n_diagnosis_codes, visit_size)
        
        # relu layer
        self.relu = nn.ReLU()
        
        # gru layer
        self.lstm = nn.LSTM(visit_size, hidden_size, num_layers = 1, batch_first = False)
        
        # dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # fully connected layer
        self.fc = nn.Linear(hidden_size, n_labels)
        
        # softmax
        self.softmax = nn.Softmax()


    def forward(self, x, mask):
        x = self.embed(x) # (n_visits, n_samples, visit_size)
        x = self.relu(x)
        
        if self.gpu:
            h0 = Variable(torch.FloatTensor(torch.zeros(1, x.size()[1], x.size()[2])).cuda())
            c0 = Variable(torch.FloatTensor(torch.zeros(1, x.size()[1], x.size()[2])).cuda())
        else:
            h0 = Variable(torch.FloatTensor(torch.zeros(1, x.size()[1], x.size()[2])))
            c0 = Variable(torch.FloatTensor(torch.zeros(1, x.size()[1], x.size()[2])))
        
        output, h_n = self.lstm(x, (h0, c0)) # output (seq_len, batch, hidden_size)
                                         #h_n (num_layers, batch, hidden_size)
        mask = mask.unsqueeze(-1).expand_as(output) 
        
        x = output * mask
        x = x.sum(0)

        x = self.dropout(x) # (n_samples, hidden_size)
        logit = self.fc(x) # (n_samples, n_labels)
        return logit


class LSTMTimeAware(nn.Module):

    def __init__(self, options):
        super(LSTMTimeAware, self).__init__()
        self.options = options

        n_diagnosis_codes = options['n_diagnosis_codes']
        visit_size = options['visit_size']
        hidden_size = options['hidden_size']

        n_labels = options['n_labels']
        dropout_rate = options['dropout_rate']

        self.gpu = options['use_gpu']

        # code embedding layer
        self.embed = nn.Linear(n_diagnosis_codes, visit_size)

        # relu layer
        self.relu = nn.ReLU()
        self.I = nn.Linear(visit_size, hidden_size)
        self.F = nn.Linear(visit_size, hidden_size)
        self.OG = nn.Linear(visit_size, hidden_size)
        self.C = nn.Linear(visit_size, hidden_size)
        self.DE = nn.Linear(visit_size, hidden_size)
        self.O = nn.Linear(visit_size, hidden_size)
        # dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # fully connected layer
        self.fc = nn.Linear(hidden_size, n_labels)

        # softmax
        self.softmax = nn.Softmax()

    def step(self, prev_hidden_memory, input, time):
        ev_hidden_state, prev_cell = tf.unstack(prev_hidden_memory)


        # Map elapse time in days or months
        T = self.map_elapse_time(time)

        # Decompose the previous cell if there is a elapse time
        C_ST = tf.nn.tanh(tf.matmul(prev_cell, self.W_decomp) + self.b_decomp)
        C_ST_dis = tf.multiply(T, C_ST)
        # if T is 0, then the weight is one
        prev_cell = prev_cell - C_ST + C_ST_dis

        # Input gate
        i = tf.sigmoid(tf.matmul(x, self.Wi) + tf.matmul(prev_hidden_state, self.Ui) + self.bi)

        # Forget Gate
        f = tf.sigmoid(tf.matmul(x, self.Wf) + tf.matmul(prev_hidden_state, self.Uf) + self.bf)

        # Output Gate
        o = tf.sigmoid(tf.matmul(x, self.Wog) + tf.matmul(prev_hidden_state, self.Uog) + self.bog)

        # Candidate Memory Cell
        C = tf.nn.tanh(tf.matmul(x, self.Wc) + tf.matmul(prev_hidden_state, self.Uc) + self.bc)

        # Current Memory cell
        Ct = f * prev_cell + i * C

        # Current Hidden state
        current_hidden_state = o * tf.nn.tanh(Ct)

        return tf.stack([current_hidden_state, Ct])

    def forward(self, x, mask):
        x = self.embed(x)  # (n_visits, n_samples, visit_size)
        x = self.relu(x)

        if self.gpu:
            h0 = Variable(torch.FloatTensor(torch.zeros(1, x.size()[1], x.size()[2])).cuda())
            c0 = Variable(torch.FloatTensor(torch.zeros(1, x.size()[1], x.size()[2])).cuda())
        else:
            h0 = Variable(torch.FloatTensor(torch.zeros(1, x.size()[1], x.size()[2])))
            c0 = Variable(torch.FloatTensor(torch.zeros(1, x.size()[1], x.size()[2])))

        output, h_n = self.lstm(x, (h0, c0))  # output (seq_len, batch, hidden_size)
        # h_n (num_layers, batch, hidden_size)
        mask = mask.unsqueeze(-1).expand_as(output)

        x = output * mask
        x = x.sum(0)

        x = self.dropout(x)  # (n_samples, hidden_size)
        logit = self.fc(x)  # (n_samples, n_labels)
        return logit