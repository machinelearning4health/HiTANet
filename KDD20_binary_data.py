#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 15:43:33 2019

@author: ffm5105
"""
"""
Data Format:
    0: patient id
    1: claim id
    2: sex
    3: race
    4: age group
    5: location
    6: country
    7: time
    8: procedure codes
    9: diagnosis codes
"""

import pickle
import datetime
diseases = ['copd', 'hf', 'kidney', 'amnesia', 'dementias']

for disease in diseases:
    case_file = 'data/' + disease + '/' + disease + '_case_patient_data_new.txt'
    control_file = 'data/' + disease + '/' + disease + '_control_patient_data_new.txt'
    visit_file = 'data/' + disease + '/' + disease + '_visits_new.pickle'
    label_file = 'data/' + disease + '/' + disease + '_labels_new.pickle'
    identity_file = 'data/' + disease + '/' + disease + '_identity_new.pickle'
    dict_file = 'data/' + disease + '/' + disease + '_code2idx_new.pickle'
    time_file = 'data/' + disease + '/' + disease + '_times_new.pickle'
    code2id = {}
    claims = [] # patient, visit, code [patient:[visit:[codes], [codes], ...], [[],[], ...], ...]
    labels = [] # label [patient's label, label, ...]
    times = []
    identity = []
    patient_claims = {}
    patient_labels = {}
    patient_date = {}
    patient_identity = {}
    files = [control_file, case_file]
    for i in range(len(files)):
        with open(files[i], 'r') as f:
            for line in f.readlines():
                strs = line.strip().split('\t')
                #print(strs)
                pid = strs[0]
                sex = strs[2]
                race = strs[3]
                age = strs[4]
                time = strs[7]
                try:
                    datetime.datetime.strptime(time, '%Y-%m-%d')
                except:
                    print("here")
                diagnosis = strs[-1].split(',')
                if pid not in patient_claims.keys():
                    patient_claims[pid] = []
                    patient_labels[pid] = i
                    patient_date[pid] = []
                    patient_identity[pid] = [sex, race, age]
                
                # code to index
                binary_visit = []
                for code in diagnosis:
                    if code not in code2id.keys():
                        code2id[code] = len(code2id)
                    binary_visit.append(code2id[code])
                
                # visits for each patient
                patient_claims[pid].append(binary_visit)
                patient_date[pid].append(time)
    for key in patient_date.keys():
        data_list = patient_date[key]
        data_list.sort(key=lambda x:datetime.datetime.strptime(x, '%Y-%m-%d'))
        final_time = datetime.datetime.strptime(data_list[-1], '%Y-%m-%d')
        for k in range(len(patient_date[key])):
            time_now = datetime.datetime.strptime(patient_date[key][k], '%Y-%m-%d')
            delta = final_time - time_now
            patient_date[key][k] = int(delta.days)
    # all patients' data
    for pid, pclaims in patient_claims.items():
        claims.append(pclaims)
        labels.append(patient_labels[pid])
        times.append(patient_date[pid])
        identity.append(patient_identity[pid])
    
    # write files
    with open(visit_file, 'wb') as f:
        pickle.dump(claims, f, protocol = 0)
    with open(label_file, 'wb') as f:
        pickle.dump(labels, f, protocol = 0)
    with open(time_file, 'wb') as f:
        pickle.dump(times, f, protocol = 0)
    with open(dict_file, 'wb') as f:
        pickle.dump(code2id, f, protocol = 0)
    with open(identity_file, 'wb') as f:
        pickle.dump(identity, f, protocol = 0)
        
                
                