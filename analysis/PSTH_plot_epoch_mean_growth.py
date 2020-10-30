# basic packages #
import os
import numpy as np
import pickle

# gen_PSTH_log #
from .PSTH_gen_PSTH_log import gen_PSTH_log

# plot #
import matplotlib.pyplot as plt

def plot_epoch_mean_growth(hp,
                        log,
                        trial_list,
                        model_dir,
                        rule,
                        seltive_epoch,
                        analy_epoch,
                        n_types=('exh_neurons','mix_neurons'),
                        norm = True,
                        PSTH_log = None,):
    
    print('\nStart ploting epoch mean growth')
    save_path = 'figure/figure_'+model_dir.rstrip('/').split('/')[-1]+'/'+rule+'/'+seltive_epoch+'/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    with open(model_dir+'/task_info.pkl','rb') as tinf:
        task_info = pickle.load(tinf)

    if PSTH_log is None:
        PSTH_log = gen_PSTH_log(hp,trial_list,model_dir,rule,seltive_epoch,n_types=n_types,norm=norm)

    is_dict = False
    is_list = False
    if isinstance(trial_list, dict):
        temp_list = list()
        is_dict = True
        for value in trial_list[rule].values():
            temp_list += value
        temp_list = sorted(set(temp_list))
    elif isinstance(trial_list, list):
        temp_list = trial_list
        is_list = True

    fig, ax = plt.subplots()
    for trial_num in temp_list:
        growth = log['perf_'+rule][trial_num//log['trials'][1]]
        if (is_list and growth > hp['mid_target_perf']) or (is_dict and trial_num in trial_list[rule]['mature']):
            color = 'red'
        elif (is_list and growth > hp['early_target_perf']) or (is_dict and trial_num in trial_list[rule]['mid']):
            color = 'blue'
        elif is_list or (is_dict and trial_num in trial_list[rule]['early']):
            color = 'green'

        mean_value = PSTH_log[trial_num][:,task_info[rule]['epoch_info'][analy_epoch][0]:task_info[rule]['epoch_info'][analy_epoch][1]].mean()

        ax.scatter(trial_num, mean_value, marker = '+',color = color)
    plt.savefig(save_path+analy_epoch+'_epoch_mean_growth.png')
    plt.savefig(save_path+analy_epoch+'_epoch_mean_growth.pdf')

    print('\tfinish')