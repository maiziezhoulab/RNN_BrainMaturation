# basic packages #
import os
import numpy as np
import pickle
from collections import OrderedDict

# independent ttest #
from scipy.stats import ttest_ind

# gen_PSTH_log #
from .PSTH_gen_PSTH_log import gen_PSTH_log

# plot heatmap #
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
    
def plot_period_ttest_heatmap(
                        hp,
                        log,
                        trial_list,
                        model_dir,
                        rule,
                        seltive_epoch,
                        analy_epoch,
                        n_types=('exh_neurons','mix_neurons'),
                        norm = True,
                        PSTH_log = None,):

    print('\nStart ploting period ttest heatmap')

    save_path = 'figure/figure_'+model_dir.rstrip('/').split('/')[-1]+'/'+rule+'/'+seltive_epoch+'/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    with open(model_dir+'/task_info.pkl','rb') as tinf:
        task_info = pickle.load(tinf)

    if PSTH_log is None:
        PSTH_log = gen_PSTH_log(hp,trial_list,model_dir,rule,seltive_epoch,n_types=n_types,norm=norm)

    if isinstance(trial_list, dict):
        temp_list = list()
        for value in trial_list[rule].values():
            temp_list += value
        temp_list = sorted(set(temp_list))
    elif isinstance(trial_list, list):
        temp_list = trial_list

    row = [str(trial_num)+'_'+str(log['perf_'+rule][trial_num//log['trials'][1]])[:4] for trial_num in temp_list]
    init_data = np.full([len(row),len(row)], np.nan)
    df = pd.DataFrame(data=init_data,  columns=row, index=row)

    print('\tindependent Ttest...')
    count = 0
    for trial_num1 in temp_list:
        for trial_num2 in temp_list:
            value1 = PSTH_log[trial_num1][:,task_info[rule]['epoch_info'][analy_epoch][0]:task_info[rule]['epoch_info'][analy_epoch][1]].mean(axis=1)
            key1 = str(trial_num1)+'_'+str(log['perf_'+rule][trial_num1//log['trials'][1]])[:4]

            value2 = PSTH_log[trial_num2][:,task_info[rule]['epoch_info'][analy_epoch][0]:task_info[rule]['epoch_info'][analy_epoch][1]].mean(axis=1)
            key2 = str(trial_num2)+'_'+str(log['perf_'+rule][trial_num2//log['trials'][1]])[:4]
            t, p = ttest_ind(value1,value2)

            df.loc[str(key1),str(key2)] = p

            count+=1
            process = str(count/(len(temp_list)**2)*100).split('.')
            print ("\r\t processing... "+process[0]+'.'+process[1][0]+'%', end="",flush=True)
    print('\n\tfinish')
    print('\tploting')
    fig, ax = plt.subplots(figsize=(40,32))
    sns.heatmap(df, annot=False, ax=ax)
    plt.savefig(save_path+'rule_'+rule+'_sel_'+seltive_epoch+'_analy_'+analy_epoch+'_heatmap.pdf')
    plt.savefig(save_path+'rule_'+rule+'_sel_'+seltive_epoch+'_analy_'+analy_epoch+'_heatmap.png')
    print('\tfinish')
