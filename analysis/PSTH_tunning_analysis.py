# basic packages #
import os
import numpy as np
import pickle

import sys
sys.path.append('.')
from utils.tools import mkdir_p

# matplotlib #
from matplotlib import pyplot as plt


def max_central(max_dir,tuning):

    temp_len = len(tuning)
    if temp_len%2 == 0:
        mc_len = temp_len + 1
    else:
        mc_len = temp_len

    firerate_max_central = np.zeros(mc_len)
    for i in range(temp_len):
        new_index = (i-max_dir+temp_len//2)%temp_len
        firerate_max_central[new_index] = tuning[i]
    if temp_len%2 == 0:
        firerate_max_central[-1] = firerate_max_central[0]

    return firerate_max_central


def tunning_analysis(
                    hp,
                    log,
                    model_dir, 
                    rule,
                    epoch,
                    trial_list,
                    n_types=('exh_neurons','mix_neurons'),
                    gaussion_fit = True,
                    height_ttest = True, 
                    ):
    
    if gaussion_fit:
        # curve fit #
        from scipy.optimize import curve_fit
        import math

        def gaussian(x, a,u, sig):
            return a*np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (sig * math.sqrt(2 * math.pi))

    if height_ttest:
        # independent t-test #
        from scipy.stats import ttest_ind

    tuning_store = dict()
    info_store = dict()
    height_store = dict()
    
    for trial_num in trial_list:
        growth = log['perf_'+rule][trial_num//log['trials'][1]]
        if growth <= hp['early_target_perf']:
            mature_key = "early"
        elif growth <= hp['mid_target_perf']:
            mature_key = "mid"
        else:
            mature_key = "mature"

        if mature_key not in tuning_store:
            tuning_store[mature_key] = list()
            info_store[mature_key] = list()
            height_store[mature_key] = list()

        n_list = list()

        read_name = model_dir+'/'+str(trial_num)+'/neuron_info_'+rule+'_'+epoch+'.pkl'
        with open(read_name,'rb') as nf:
            ninf = pickle.load(nf)

        for ntype in n_types:
            n_list += ninf[ntype]
        n_list = list(set(n_list))

        trial_avrg_tuning = list()
        for neuron_inf in n_list:
            max_dir = neuron_inf[2]
            tuning = ninf['firerate_loc_order'][neuron_inf[0]]
            trial_avrg_tuning.append(max_central(max_dir,tuning))
            height_store[mature_key].append(tuning.max()-tuning.min())

        trial_avrg_tuning = np.array(trial_avrg_tuning).mean(axis=0)

        tuning_store[mature_key].append(trial_avrg_tuning)
        info_store[mature_key].append((len(n_list),growth))

    for key in tuning_store.keys():
        tuning_store[key] = np.array(tuning_store[key])
        height_store[key] = np.array(height_store[key])


    fig,ax = plt.subplots(figsize=(16,10))

    if len(trial_list) == 1:
        step = 'None'
    else:
        step = str(trial_list[1]-trial_list[0])
    trial_range = str((trial_list[0],trial_list[-1]))
    title = 'Rule:'+rule+' Epoch:'+epoch+' trial range:'+trial_range+' step:'+step

    for mature_key in tuning_store.keys():

        if mature_key == 'mature':
            color = 'red'
        elif mature_key == 'mid':
            color = 'blue'
        elif mature_key == 'early':
            color = 'green'

        temp_tuning = tuning_store[mature_key].mean(axis=0)
        temp_x = np.arange(len(temp_tuning))

        avg_n_number = int(np.array([x[0] for x in info_store[mature_key]]).mean())
        avg_growth = np.array([x[1] for x in info_store[mature_key]]).mean()

        ax.scatter(temp_x, temp_tuning, marker = '+',color = color, s = 70 ,\
                                label = mature_key+' avg_n_num(int):'+str(avg_n_number)+' avg_growth:%.1f'%(avg_growth))

        if gaussion_fit:
            gaussian_x = np.arange(-0.1,len(temp_tuning)-0.9,0.1)
            paras , _ = curve_fit(gaussian,temp_x,temp_tuning+(-1)*np.min(temp_tuning),\
                p0=[np.max(temp_tuning)+1,len(temp_tuning)//2,1])
            gaussian_y = gaussian(gaussian_x,paras[0],paras[1],paras[2])-np.min(temp_tuning)*(-1)
            width = paras[2]

            ax.plot(gaussian_x, gaussian_y, color=color,linestyle = '--',\
                label = mature_key+' curve_width:%.2f'%(width*2))

    if height_ttest:
        maturation = list(tuning_store.keys())
        title += '\nHeight independent T-test p-value: '
        for i in range(len(maturation)-1):
            for j in range(i+1,len(maturation)):
                t_h, p_h = ttest_ind(height_store[maturation[i]],height_store[maturation[j]])
                title += maturation[i]+'-'+maturation[j]+':%.1e '%(p_h)

    fig.suptitle(title)
    ax.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)

    save_path = 'figure/figure_'+model_dir.rstrip('/').split('/')[-1]+'/'+rule+'/'+epoch+'/'+'_'.join(n_types)+'/'
    mkdir_p(save_path)
    plt.savefig(save_path+rule+'_'+epoch+'_'+trial_range+'_step_'+step+'_tuning_analysis.pdf',bbox_inches='tight')
    plt.savefig(save_path+rule+'_'+epoch+'_'+trial_range+'_step_'+step+'_tuning_analysis.eps',bbox_inches='tight')
    plt.savefig(save_path+rule+'_'+epoch+'_'+trial_range+'_step_'+step+'_tuning_analysis.png',bbox_inches='tight')

    plt.close()