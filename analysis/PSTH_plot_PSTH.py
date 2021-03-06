# basic packages #
import os
import numpy as np
import pickle

import sys
sys.path.append('.')
from utils.tools import mkdir_p

# gen_PSTH_log #
from .PSTH_gen_PSTH_log import gen_PSTH_log

from .PSTH_compute_H import Get_H

# plot #
from matplotlib import pyplot as plt

def plot_PSTH(
            hp,
            log,
            model_dir, 
            rule,
            epoch,
            trial_list,
            n_types=('exh_neurons','mix_neurons'),
            plot_oppo_dir = False,
            norm = True,
            PSTH_log = None,
            ):

    print("Start ploting PSTH")
    print("\trule: "+rule+" selective epoch: "+epoch)

    n_number = dict()

    if PSTH_log is None:
        PSTH_log = gen_PSTH_log(hp,trial_list,model_dir,rule,epoch,n_types=n_types,norm=norm)

        if plot_oppo_dir:
            PSTH_log_oppo = gen_PSTH_log(hp,trial_list,model_dir,rule,epoch,n_types=n_types,norm=norm,oppo_sel_dir=plot_oppo_dir)
            for key,value in PSTH_log_oppo.items():
                PSTH_log_oppo[key] = value.mean(axis=0)

    for key,value in PSTH_log.items():
        n_number[key] = np.size(value,0)
        PSTH_log[key] = value.mean(axis=0)

    data_to_plot = dict()
    data_types = ["PSTH","n_num","growth"]
    if plot_oppo_dir:
        data_types.append("PSTH_oppo")

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
    
    for trial_num in temp_list:
        growth = log['perf_'+rule][trial_num//log['trials'][1]]

        #if growth <= hp['early_target_perf']:
        #if hp['early_target_perf']-0.05 <= growth <= hp['early_target_perf']+0.05:
        if (is_list and growth > hp['mid_target_perf']) or (is_dict and trial_num in trial_list[rule]['mature']):
            m_key = "mature"
        #elif growth <= hp['mid_target_perf']:
        #elif hp['mid_target_perf']-0.05 <= growth <= hp['mid_target_perf']+0.05:
        elif (is_list and growth > hp['early_target_perf']) or (is_dict and trial_num in trial_list[rule]['mid']):
            m_key = "mid"
        #else:
        #elif hp['mature_target_perf']-0.05 <= growth <= hp['mature_target_perf']+0.05:
        elif is_list or (is_dict and trial_num in trial_list[rule]['early']):
            m_key = "early"
        #else:
            #continue

        if m_key not in data_to_plot:
            data_to_plot[m_key] = dict()
            for data_type in data_types:
                data_to_plot[m_key][data_type] = list()

        data_to_plot[m_key]["PSTH"].append(PSTH_log[trial_num])
        data_to_plot[m_key]["growth"].append(growth)
        data_to_plot[m_key]["n_num"].append(n_number[trial_num])
        if plot_oppo_dir:
            data_to_plot[m_key]["PSTH_oppo"].append(PSTH_log_oppo[trial_num])

    for m_key in data_to_plot.keys():
        for data_type in data_types:
            data_to_plot[m_key][data_type] = np.array(data_to_plot[m_key][data_type]).mean(axis=0)

    # plot #
    save_path = 'figure/figure_'+model_dir.rstrip('/').split('/')[-1]+'/'+rule+'/'+epoch+'/'+'_'.join(n_types)+'/'
    
    if is_dict or len(temp_list) == 1:
        step = 'None'
    else:
        step = str(temp_list[1]-temp_list[0])

    if is_dict:
        trial_range = 'auto_choose'
    else:
        trial_range = str((temp_list[0],temp_list[-1]))
    
    title = 'Rule:'+rule+' Epoch:'+epoch+' Neuron_type:'+'_'.join(n_types)+' trial range:'+trial_range+' step:'+step

    colors = {"early":"green","mid":"blue","mature":"red",}

    fig,ax = plt.subplots(figsize=(14,10))
    fig.suptitle(title)

    for m_key in data_to_plot.keys():
        ax.plot(np.arange(len(data_to_plot[m_key]["PSTH"]))*hp['dt']/1000, data_to_plot[m_key]["PSTH"],\
            label= m_key+'_%.2f'%(data_to_plot[m_key]["growth"])+'_n%d'%(data_to_plot[m_key]["n_num"]), color=colors[m_key])
        if plot_oppo_dir:
            ax.plot(np.arange(len(data_to_plot[m_key]["PSTH_oppo"]))*hp['dt']/1000, data_to_plot[m_key]["PSTH_oppo"],\
                label= m_key+'_opposite_sel_dir', color=colors[m_key], linestyle = '--')
    
    ax.set_xlabel("time/s")
    ax.set_ylabel("activity")
    ax.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)

    mkdir_p(save_path)
    plt.savefig(save_path+rule+'_'+epoch+'_'+trial_range+'_step_'+step+'_PSTH.pdf',bbox_inches='tight')
    plt.savefig(save_path+rule+'_'+epoch+'_'+trial_range+'_step_'+step+'_PSTH.eps',bbox_inches='tight')
    plt.savefig(save_path+rule+'_'+epoch+'_'+trial_range+'_step_'+step+'_PSTH.png',bbox_inches='tight')

    plt.close()
