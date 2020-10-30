# basic packages #
import os
import numpy as np
import pickle

import sys
sys.path.append('.')
from utils.tools import mkdir_p,load_hp,load_log

# plot #
from matplotlib import pyplot as plt

def gen_PSTH_log_neuron_selected(
                                trial_list,
                                model_dir,
                                rule,
                                seltive_epoch,
                                selected_n_list,
                                n_types=('exh_neurons','mix_neurons'),
                                norm = True,
                                oppo_sel_dir = False):

    with open(model_dir+'/task_info.pkl','rb') as tinf:
        task_info = pickle.load(tinf)

    PSTH_log = dict()
    print('\tGenerating PSTH '+rule+' '+seltive_epoch)

    count = 0

    if isinstance(trial_list, dict):
        temp_list = list()
        for value in trial_list[rule].values():
            temp_list += value
        temp_list = sorted(set(temp_list))
    elif isinstance(trial_list, list):
        temp_list = trial_list
    
    for trial_num in temp_list:
        
        H_file = model_dir+'/'+str(trial_num)+'/H_'+rule+'.pkl'
        with open(H_file,'rb') as hf:
            H = pickle.load(hf)

        n_info_file = model_dir+'/'+str(trial_num)+'/neuron_info_'+rule+'_'+seltive_epoch+'.pkl'
        with open(n_info_file,'rb') as ninf:
            n_info = pickle.load(ninf)

        n_list_temp = list()
        for n_type in n_types:
            n_list_temp = list(set(n_list_temp+n_info[n_type]))

        n_list = [n for n in n_list_temp if n[0] in selected_n_list]

        psth_log_temp = list()
        for neuron in n_list:

            if oppo_sel_dir:
                loc = (neuron[2]+len(task_info[rule]['in_loc_set'])//2)%len(task_info[rule]['in_loc_set'])
                if len(task_info[rule]['in_loc_set'])%2:
                    psth_temp = (H[:,task_info[rule]['in_loc'] == loc, neuron[0]].mean(axis=1)+\
                        H[:,task_info[rule]['in_loc'] == (loc+1)%len(task_info[rule]['in_loc_set']), neuron[0]].mean(axis=1))/2.0
                else:
                    psth_temp = H[:,task_info[rule]['in_loc'] == loc, neuron[0]].mean(axis=1)

            else:
                loc = neuron[2]
                psth_temp = H[:,task_info[rule]['in_loc'] == loc, neuron[0]].mean(axis=1)

            
            fix_level = H[task_info[rule]['epoch_info']['fix1'][0]:task_info[rule]['epoch_info']['fix1'][1], \
                task_info[rule]['in_loc'] == loc, neuron[0]].mean(axis=1).mean(axis=0)

            psth_norm = psth_temp/fix_level-1

            if norm:
                psth_log_temp.append(psth_norm)
            else:
                psth_log_temp.append(psth_temp)


        try:
            PSTH_log[trial_num] = np.array(psth_log_temp)
        except:
            pass

        count+=1
        process = count/len(temp_list)*100
        print ("\r\t processing... %.1f%%"%(process), end="",flush=True)

    print('\n\tfinish')

    return PSTH_log

def plot_PSTH_neuron_selected(hp,
                            log,
                            model_dir, 
                            rule,
                            epoch,
                            trial_list,
                            selected_n_list=None,
                            n_types_selected=('exh_neurons','mix_neurons'),
                            n_types_intersection_limit=('selective_neurons',),#('exh_neurons','mix_neurons'),#('active_neurons',),
                            plot_oppo_dir = False,
                            norm = True,
                            PSTH_log = None,):

    print("Start ploting neuron selected PSTH")
    print("\trule: "+rule+" selective epoch: "+epoch)

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

    if selected_n_list is None:
        n_info_file = model_dir+'/'+str(temp_list[-1])+'/neuron_info_'+rule+'_'+epoch+'.pkl'
        with open(n_info_file,'rb') as ninf:
            n_info = pickle.load(ninf)

        n_list = list()
        for n_type in n_types_selected:
            n_list = list(set(n_list+n_info[n_type]))

        selected_n_list = [n[0] for n in n_list]

    for trial_num in temp_list:
        n_info_file = model_dir+'/'+str(trial_num)+'/neuron_info_'+rule+'_'+epoch+'.pkl'
        with open(n_info_file,'rb') as ninf:
            n_info = pickle.load(ninf)

        n_index_temp = list()
        for n_type in n_types_intersection_limit:
            n_index_temp = list(set(n_index_temp+n_info[n_type]))
        n_index_temp = [n[0] for n in n_index_temp]

        selected_n_list = list(set(selected_n_list)&set(n_index_temp))
    
    print("%d neuron(s) selected"%(len(selected_n_list)))

    n_number = dict()

    if PSTH_log is None:
        PSTH_log = gen_PSTH_log_neuron_selected(trial_list,model_dir,rule,epoch,\
            selected_n_list=selected_n_list,n_types=n_types_intersection_limit,norm=norm)

        if plot_oppo_dir:
            PSTH_log_oppo = gen_PSTH_log_neuron_selected(trial_list,model_dir,rule,epoch,\
                selected_n_list=selected_n_list,n_types=n_types_intersection_limit,norm=norm,oppo_sel_dir=plot_oppo_dir)
            for key,value in PSTH_log_oppo.items():
                PSTH_log_oppo[key] = value.mean(axis=0)

    for key,value in PSTH_log.items():
        n_number[key] = np.size(value,0)
        PSTH_log[key] = value.mean(axis=0)

    data_to_plot = dict()
    maturation = ["early","mid","mature"]
    data_types = ["PSTH","n_num","growth"]
    if plot_oppo_dir:
        data_types.append("PSTH_oppo")
    
    for m_key in maturation:
        data_to_plot[m_key] = dict()
        for data_type in data_types:
            data_to_plot[m_key][data_type] = list()
    
    for trial_num in temp_list:
        growth = log['perf_'+rule][trial_num//log['trials'][1]]

        if (is_list and growth > hp['mid_target_perf']) or (is_dict and trial_num in trial_list[rule]['mature']):
            m_key = "mature"
        elif (is_list and growth > hp['early_target_perf']) or (is_dict and trial_num in trial_list[rule]['mid']):
            m_key = "mid"
        elif is_list or (is_dict and trial_num in trial_list[rule]['early']):
            m_key = "early"

        data_to_plot[m_key]["PSTH"].append(PSTH_log[trial_num])
        data_to_plot[m_key]["growth"].append(growth)
        data_to_plot[m_key]["n_num"].append(n_number[trial_num])
        if plot_oppo_dir:
            data_to_plot[m_key]["PSTH_oppo"].append(PSTH_log_oppo[trial_num])

    for m_key in maturation:
        for data_type in data_types:
            data_to_plot[m_key][data_type] = np.array(data_to_plot[m_key][data_type]).mean(axis=0)

    # plot #
    save_path = 'figure/figure_'+model_dir.rstrip('/').split('/')[-1]+'/'+rule+'/'+epoch+'/'
    if is_dict or len(temp_list) == 1:
        step = 'None'
    else:
        step = str(temp_list[1]-temp_list[0])

    if is_dict:
        trial_range = 'auto_choose'
    else:
        trial_range = str((temp_list[0],temp_list[-1]))
    title = 'Rule:'+rule+' Epoch:'+epoch+' trial range:'+trial_range+' step:'+step

    colors = {"early":"green","mid":"blue","mature":"red",}

    fig,ax = plt.subplots(figsize=(14,10))
    fig.suptitle(title)

    for m_key in maturation:
        ax.plot(np.arange(len(data_to_plot[m_key]["PSTH"]))*hp['dt']/1000, data_to_plot[m_key]["PSTH"],\
            label= m_key+'_%.2f'%(data_to_plot[m_key]["growth"])+'_n%d'%(data_to_plot[m_key]["n_num"]), color=colors[m_key])
        if plot_oppo_dir:
            ax.plot(np.arange(len(data_to_plot[m_key]["PSTH_oppo"]))*hp['dt']/1000, data_to_plot[m_key]["PSTH_oppo"],\
                label= m_key+'_opposite_sel_dir', color=colors[m_key], linestyle = '--')
    
    ax.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)

    mkdir_p(save_path)
    plt.savefig(save_path+rule+'_'+epoch+'_'+trial_range+'_step_'+step+'neuron_selected_PSTH.pdf',bbox_inches='tight')
    plt.savefig(save_path+rule+'_'+epoch+'_'+trial_range+'_step_'+step+'neuron_selected_PSTH.eps',bbox_inches='tight')
    plt.savefig(save_path+rule+'_'+epoch+'_'+trial_range+'_step_'+step+'neuron_selected_PSTH.png',bbox_inches='tight')

    plt.close()

    with open(save_path+rule+'_'+epoch+'_'+trial_range+'_step_'+step+"selected_neurons.txt","w") as tf:
        tf.write(str(selected_n_list))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model_dir = 'data/6tasks'
    hp = load_hp(model_dir)
    log = load_log(model_dir)
    trial_list = range(520960,628480+1,1280)

    plot_PSTH_neuron_selected(hp,log,model_dir,'odr','stim1',trial_list,plot_oppo_dir = False,)
    plot_PSTH_neuron_selected(hp,log,model_dir,'odr','delay1',trial_list,plot_oppo_dir = False,)
    plot_PSTH_neuron_selected(hp,log,model_dir,'odrd','stim1',trial_list,plot_oppo_dir = True,)
    plot_PSTH_neuron_selected(hp,log,model_dir,'odrd','delay1',trial_list,plot_oppo_dir = True,)
    plot_PSTH_neuron_selected(hp,log,model_dir,'odrd','delay2',trial_list,plot_oppo_dir = True,)

    #selected_n_list = [20, 120, 182, 101, 232, 166, 168, 225, 231, 111, 46, 136, 42, 41, 183, 149, 237]
    #plot_PSTH_neuron_selected(hp,log,model_dir,'odr','stim1',trial_list,selected_n_list,plot_oppo_dir = False,)
    #plot_PSTH_neuron_selected(hp,log,model_dir,'odr','delay1',trial_list,selected_n_list,plot_oppo_dir = False,)