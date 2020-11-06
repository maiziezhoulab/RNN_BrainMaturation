# basic packages #
import os
import numpy as np
import pickle

import sys
sys.path.append('.')
from utils.tools import mkdir_p

# plot heatmap #
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#SVM#
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split#, GridSearchCV
from sklearn.svm import SVC

from .PSTH_compute_H import Get_H

def split_test_train(x,y,test_num=1):

    labels = sorted(set(y))
    trial_per_label = int(len(y)/len(labels))

    y_test = list()
    y_train = list()

    for label in labels:
        if label == labels[0]:
            x_train = x[y==label,:][test_num:,:]
            x_test = x[y==label,:][0:test_num,:]
        else:
            x_train = np.concatenate((x_train,x[y==label,:][test_num:,:]),axis=0)
            x_test = np.concatenate((x_test,x[y==label,:][0:test_num,:]),axis=0)

        for i in range(trial_per_label-test_num):
            y_train.append(label)
        for i in range(test_num):
            y_test.append(label)


    return x_train, x_test, np.array(y_train), np.array(y_test)

def Decoder_analysis(hp,
                    model_dir,
                    trial_num,
                    rule='odr',
                    task_mode='test',
                    window_size=2,
                    stride=1,
                    #test_size=.25,
                    test_num=2,
                    ):

    task_info_file = model_dir+'/task_info.pkl'
    with open(task_info_file,'rb') as tinfr:
        task_info = pickle.load(tinfr)
    
    H = Get_H(hp,model_dir,trial_num,rule,task_mode=task_mode,)

    time_len = np.size(H,0)

    time_bin_num = int((time_len-window_size)//stride+1)

    tb_set = dict()

    #if rule == 'odr':
    out_loc = np.array(task_info[rule]['in_loc'])
    #elif rule == 'odrd':

    row = list()
    for b_num in range(time_bin_num):
        start = b_num*stride
        end = start+window_size
        #scaler = StandardScaler()
        #tb_set[b_num] = scaler.fit_transform(H[start:end,:,:].mean(axis=0))
        tb_set[b_num] = H[start:end,:,:].mean(axis=0)
        row.append(str(start*hp['dt']/1000))

    init_data = np.full([len(row),len(row)], np.nan)
    df = pd.DataFrame(data=init_data,  columns=row, index=row)

    for b_num1 in range(time_bin_num):
        
        #x_train, x_test, y_train, y_test = train_test_split(tb_set[b_num1],out_loc,test_size=test_size)
        x_train, x_test, y_train, y_test = split_test_train(tb_set[b_num1],out_loc,test_num=test_num)
        clf = SVC()
        clf.fit(x_train, y_train)
        #clf.fit(tb_set[b_num1],out_loc)

        key1 = str(b_num1*stride*hp['dt']/1000)

        for b_num2 in range(time_bin_num):
            key2 = str(b_num2*stride*hp['dt']/1000)
            
            if b_num1 == b_num2:
                score = clf.score(x_test,y_test)
            else:
                score = clf.score(tb_set[b_num2],out_loc)
            
            
            #score = clf.score(tb_set[b_num2],out_loc)
            

            df.loc[key1,key2] = score

    fig, ax = plt.subplots(figsize=(20,16))
    sns.heatmap(df, annot=False, ax=ax,cmap="rainbow",vmin=0.125, vmax=1)
    save_path = 'figure/figure_'+model_dir.rstrip('/').split('/')[-1]+'/'+rule+'/'
    plt.savefig(save_path+rule+'_'+str(trial_num)+'_w'+str(window_size*hp['dt'])+'ms_s'+str(stride*hp['dt'])+'ms.pdf',bbox_inches='tight')
    plt.close()

def realneuron_Decoder_analysis(neuron_file='data/adultFiringRate8trials.txt',
                                test_size=.25,
                                test_num=4,
                                time_len=176,
                                dt=20,
                                loc_num=8,
                                trial_per_loc=8,
                                window_size=2,
                                stride=1,
                                artifact_trial_num_per_loc=0,):
    real_neuron_reaction = list()
    with open(neuron_file,'r') as nf:
        for line in nf:
            line = line.rstrip('\n').split(',')[1:]
            for i in range(len(line)):
                line[i] = float(line[i])
            real_neuron_reaction.append(line)
    real_neuron_reaction = np.array(real_neuron_reaction)

    tb_set = dict()
    time_bin_num = int((time_len-window_size)//stride+1)

    for b_num in range(time_bin_num):
        tb_set[b_num] = list()
        for loc in range(loc_num):
            real_trials = list()
            for tn in range(trial_per_loc):
                start = loc*trial_per_loc*time_len+tn*time_len+b_num*stride
                end = start+window_size
                tb_set[b_num].append(real_neuron_reaction[start:end,:].mean(axis=0))
                real_trials.append(real_neuron_reaction[start:end,:].mean(axis=0))
            real_trials = np.array(real_trials)

            for i in range(artifact_trial_num_per_loc):
                arti_trial = list()
                for n in range(np.size(real_trials,1)):
                    arti_trial.append(np.random.choice(real_trials[:,n]))
                tb_set[b_num].append(arti_trial)
        
        tb_set[b_num] = np.array(tb_set[b_num])

    row = list()
    for b_num in range(time_bin_num):
        start_point = b_num*stride
        row.append(str(start_point*dt/1000))
    
    init_data = np.full([len(row),len(row)], np.nan)
    df = pd.DataFrame(data=init_data,  columns=row, index=row)

    out_loc = list()
    for loc in range(loc_num):
        for tn in range(trial_per_loc+artifact_trial_num_per_loc):
            out_loc.append(int(loc+1))
    out_loc = np.array(out_loc)

    scaler = StandardScaler()

    for b_num1 in range(time_bin_num):
        #x_train, x_test, y_train, y_test = train_test_split(scaler.fit_transform(tb_set[b_num1]),out_loc,test_size=test_size)
        x_train, x_test, y_train, y_test = split_test_train(scaler.fit_transform(tb_set[b_num1]),out_loc,test_num=test_num)
        clf = SVC()
        clf.fit(x_train, y_train)
        #clf.fit(tb_set[b_num1],out_loc)

        key1 = str(b_num1*stride*dt/1000)

        for b_num2 in range(time_bin_num):
            key2 = str(b_num2*stride*dt/1000)
            
            if b_num1 == b_num2:
                score = clf.score(x_test,y_test)
            else:
                _, x1_test, _, y1_test = split_test_train(scaler.fit_transform(tb_set[b_num2]),out_loc,test_num=test_num)
                #_, x1_test, _, y1_test = split_test_train(scaler.fit_transform(tb_set[b_num2]),out_loc,test_num=trial_per_loc+artifact_trial_num_per_loc-test_num)
                score = clf.score(x1_test,y1_test)
            
            
            #score = clf.score(tb_set[b_num2],out_loc)
            

            df.loc[key1,key2] = score

    fig, ax = plt.subplots(figsize=(20,16))
    sns.heatmap(df, annot=False, ax=ax,cmap="rainbow",vmin=0.125, vmax=1)
    ax.set_xlabel("time/s")
    ax.set_ylabel("time/s")

    sample_name = neuron_file.split('/')[-1].split('.')[0]
    plt.savefig('figure/realneuron_'+sample_name+'_w'+str(window_size*dt)+'ms_s'+str(stride*dt)+'ms_arti_'+str(artifact_trial_num_per_loc)+'_per_loc.pdf',bbox_inches='tight')
    plt.close()