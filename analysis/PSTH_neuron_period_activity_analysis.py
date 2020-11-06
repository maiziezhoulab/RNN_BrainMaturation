# basic packages #
import os
import numpy as np
import pickle

# gen_PSTH_log #
from .PSTH_gen_PSTH_log import gen_PSTH_log

# for ANOVA  #
#from scipy import stats
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# plot #
from matplotlib import pyplot as plt

def neuron_period_activity_analysis(hp,
                                    log,
                                    trial_list,
                                    model_dir,
                                    rule,
                                    seltive_epoch,
                                    analy_epoch,
                                    n_types=('exh_neurons','mix_neurons'),
                                    norm = True,
                                    PSTH_log = None,
                                    last_step = True,
                                    bin_wid=0.5,):
    
    print("Start neuron period activity analysis")
    print("\trule: "+rule+" selective epoch: "+seltive_epoch+" analysis epoch: ", analy_epoch)
    
    with open(model_dir+'/task_info.pkl','rb') as tinf:
        task_info = pickle.load(tinf)

    if PSTH_log is None:
        PSTH_log = gen_PSTH_log(hp,trial_list,model_dir,rule,seltive_epoch,n_types=n_types,norm=norm)

    if isinstance(analy_epoch, str):
        start = task_info[rule]['epoch_info'][analy_epoch][0]
        end = task_info[rule]['epoch_info'][analy_epoch][1]
    elif isinstance(analy_epoch, (tuple,list)):
        start = int(analy_epoch[0]/hp["dt"])
        end = int(analy_epoch[1]/hp["dt"])
    else:
        raise ValueError('Wrong analy_epoch format!')

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

    
    trial_sort_by_matur = dict()
    fire_rate_dict = dict()

    for trial_num in temp_list:
        growth = log['perf_'+rule][trial_num//log['trials'][1]]
        if (is_list and growth > hp['mid_target_perf']) or (is_dict and trial_num in trial_list[rule]['mature']):
            mature = 'mature'
        elif (is_list and growth > hp['early_target_perf']) or (is_dict and trial_num in trial_list[rule]['mid']):
            mature = 'mid'
        elif is_list or (is_dict and trial_num in trial_list[rule]['early']):
            mature = 'early'
        
        if mature not in trial_sort_by_matur:
            trial_sort_by_matur[mature] = list()
            fire_rate_dict[mature] = list()
        trial_sort_by_matur[mature].append(trial_num)

    if last_step:
        for key, value in trial_sort_by_matur.items():
            trial_sort_by_matur[key] = value[-1:]

    for mature_key, sub_trial_list in trial_sort_by_matur.items():
        for trial_num in sub_trial_list:
            fire_rate_dict[mature_key] += list(PSTH_log[trial_num][:,start:end].mean(axis = 1))

    #ANOVA#
    #f,p = stats.f_oneway(*list(fire_rate_dict.values()))

    dict_melt = dict()
    dict_melt['Maturation'] = list()
    dict_melt['Fire_rate'] = list()
    for key,value in fire_rate_dict.items():
        dict_melt['Maturation'] += [key for i in range(len(value))]
        dict_melt['Fire_rate'] += list(value)

    df_melt = pd.DataFrame(dict_melt)
    model = ols('Fire_rate~C(Maturation)',data=df_melt).fit()
    anova_table = anova_lm(model, typ = 2)

    p = anova_table['PR(>F)'][0]
    df_g = anova_table['df'][0]
    df_res = anova_table['df'][1]

    #print("\tP value:",anova_table['PR(>F)'][0])

    # plot #
    colors = {'early':'green','mid':'blue','mature':'red'}
    save_path = 'figure/figure_'+model_dir.rstrip('/').split('/')[-1]+'/'+rule+'/'+seltive_epoch+'/'+'_'.join(n_types)+'/'
    fig,axes = plt.subplots(2,1,figsize=(12,15))
    for mature,fire_rate in fire_rate_dict.items():
        axes[0].hist(fire_rate,bins=int(max(fire_rate)/bin_wid)+1,histtype="step",alpha=0.6,\
            color=colors[mature],label=mature+' mean:%.3f'%(np.mean(fire_rate)),density=1)
    axes[0].legend()
    axes[0].set_xlabel("activity")

    m_keys = list(fire_rate_dict.keys())
    axes[1].boxplot([fire_rate_dict[m_key] for m_key in m_keys], labels = m_keys)
    axes[1].set_ylabel("activity")

    fig.suptitle("rule: "+rule+" selective epoch: "+seltive_epoch+" analysis epoch: "+str(analy_epoch)+\
        "\n p value: %.3e"%(p)+" group df: %.1f"%(df_g)+" residual df: %.1f"%(df_res))

    if isinstance(analy_epoch, str):
        plt.savefig(save_path+rule+'_'+analy_epoch+'_activity_oneway_anova_analysis.png')
        plt.savefig(save_path+rule+'_'+analy_epoch+'_activity_oneway_anova_analysis.pdf')
    elif isinstance(analy_epoch, (tuple,list)):
        plt.savefig(save_path+rule+'_'+str(analy_epoch[0])+'_'+str(analy_epoch[1])+'_activity_oneway_anova_analysis.png')
        plt.savefig(save_path+rule+'_'+str(analy_epoch[0])+'_'+str(analy_epoch[1])+'_activity_oneway_anova_analysis.pdf')