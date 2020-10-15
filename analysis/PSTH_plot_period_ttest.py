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

def plot_period_ttest_by_growth(
                            hp,
                            log,
                            trial_list,
                            model_dir,
                            rule,
                            seltive_epoch,
                            analy_epoch,
                            n_types=('exh_neurons','mix_neurons'),
                            norm = True,):

    with open(model_dir+'/task_info.pkl','rb') as tinf:
        task_info = pickle.load(tinf)

    PSTH_log = gen_PSTH_log(hp,trial_list,model_dir,rule,seltive_epoch,norm=norm)

    matur_stages = ['early','mid', 'mature']
    fire_rate_dict = dict()
    for m_key in matur_stages:
        fire_rate_dict[m_key] = OrderedDict()

    #Classify by growth
    for trial_num in trial_list:
        growth = log['perf_'+rule][trial_num//log['trials'][1]]
        #if abs(growth-hp['early_target_perf'])<=0.05:
        if growth <= hp['early_target_perf']:
            fire_rate_dict[matur_stages[0]][trial_num] = \
                PSTH_log[trial_num][:,task_info[rule]['epoch_info'][analy_epoch][0]:task_info[rule]['epoch_info'][analy_epoch][1]].mean(axis=1)
        #elif abs(growth-hp['mid_target_perf'])<=0.05:
        elif growth <= hp['mid_target_perf']:
            fire_rate_dict[matur_stages[1]][trial_num] = \
                PSTH_log[trial_num][:,task_info[rule]['epoch_info'][analy_epoch][0]:task_info[rule]['epoch_info'][analy_epoch][1]].mean(axis=1)
        #elif abs(growth-hp['mature_target_perf'])<=0.05:
        else:
            fire_rate_dict[matur_stages[2]][trial_num] = \
                PSTH_log[trial_num][:,task_info[rule]['epoch_info'][analy_epoch][0]:task_info[rule]['epoch_info'][analy_epoch][1]].mean(axis=1)
#NOT DONE YET#
    
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
    #growth = log['perf_'+rule][trial_num//log['trials'][1]]

    row = [str(trial_num)+'_'+str(log['perf_'+rule][trial_num//log['trials'][1]])[:4] for trial_num in trial_list]
    init_data = np.full([len(row),len(row)], np.nan)
    df = pd.DataFrame(data=init_data,  columns=row, index=row)

    print('\tindependent Ttest...')
    count = 0
    for trial_num1 in trial_list:
        for trial_num2 in trial_list:
            value1 = PSTH_log[trial_num1][:,task_info[rule]['epoch_info'][analy_epoch][0]:task_info[rule]['epoch_info'][analy_epoch][1]].mean(axis=1)
            key1 = str(trial_num1)+'_'+str(log['perf_'+rule][trial_num1//log['trials'][1]])[:4]

            value2 = PSTH_log[trial_num2][:,task_info[rule]['epoch_info'][analy_epoch][0]:task_info[rule]['epoch_info'][analy_epoch][1]].mean(axis=1)
            key2 = str(trial_num2)+'_'+str(log['perf_'+rule][trial_num2//log['trials'][1]])[:4]
            t, p = ttest_ind(value1,value2)

            df.loc[str(key1),str(key2)] = p

            count+=1
            process = str(count/(len(trial_list)**2)*100).split('.')
            print ("\r\t processing... "+process[0]+'.'+process[1][0]+'%', end="",flush=True)
    print('\n\tfinish')
    print('\tploting')
    fig, ax = plt.subplots(figsize=(40,32))
    sns.heatmap(df, annot=False, ax=ax)
    plt.savefig(save_path+'rule_'+rule+'_sel_'+seltive_epoch+'_analy_'+analy_epoch+'_heatmap.pdf')
    plt.savefig(save_path+'rule_'+rule+'_sel_'+seltive_epoch+'_analy_'+analy_epoch+'_heatmap.png')
    print('\tfinish')
