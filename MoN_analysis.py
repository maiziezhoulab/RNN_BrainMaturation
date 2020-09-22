from utils import tools
import os
import pickle
from utils.tools import mkdir_p
import numpy as np

# plot #
from matplotlib import pyplot as plt

from analysis.PSTH_print_basic_info import print_basic_info
from analysis.PSTH_compute_H import compute_H, Get_H

#from PSTH_gen_neuron_info import generate_neuron_info
#from PSTH_tunning_analysis import tunning_analysis

# for ANOVA and paired T test analysis and plot #
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

def generate_neuron_info_mon(
                        hp,
                        model_dir, 
                        epochs, 
                        trial_list, 
                        rules, 
                        norm = True, 
                        annova_p_value = 0.05,
                        paired_ttest_p = 0.05, 
                        abs_active_thresh = 1e-3,
                        recompute = False):

    with open(model_dir+'/task_info.pkl','rb') as tinf:
        task_info = pickle.load(tinf)

    for rule in rules:

        for epoch in epochs:
            if epoch == 'stim2' or epoch == 'delay2':
                in_loc = 'in_loc_2'
            else:
                in_loc = 'in_loc'

            print('\nStart '+rule+' '+epoch+':')

            count = 0

            for trial_num in trial_list:

                count+=1
                process = count/len(trial_list)*100
                print ("\r processing... %.1f%%"%(process), end="",flush=True)

                save_name = model_dir+'/'+str(trial_num)+'/neuron_info_'+rule+'_'+epoch+'.pkl'

                if os.path.isfile(save_name) and not recompute:
                    continue

                neuron_info = dict()

                H = Get_H(hp,model_dir,trial_num,rule,save_H=False,task_mode='test',)

                for info_type in ['selective_neurons','active_neurons','exh_neurons','inh_neurons','mix_neurons','firerate_loc_order']:
                    neuron_info[info_type] = list()

                for neuron in range(hp['n_rnn']):
                    neuron_data_abs = dict()
                    neuron_data_norm = dict()
                    neuron_data = dict()
                    firerate_abs = list()
                    firerate_norm = list()
                    firerate = list()

                    paired_ttest_count = 0

                    for loc in task_info[rule]['in_loc_set']:
                        #axis = 1 for trial-wise mean, 0 for time-wise mean
                        #paired T test
                        #TODO:put the paired t-test in the outer(rule) for loop 

                        fix_level = H[task_info[rule]['epoch_info']['fix1'][0]:task_info[rule]['epoch_info']['fix1'][1],\
                            task_info[rule][in_loc] == loc, neuron].mean(axis=0)
                        stim1_level = H[task_info[rule]['epoch_info']['stim1'][0]:task_info[rule]['epoch_info']['stim1'][1],\
                            task_info[rule][in_loc] == loc, neuron].mean(axis=0)
                        paired_ttest_result = ttest_rel(fix_level, stim1_level)[1]
                        if paired_ttest_result <= paired_ttest_p:
                            paired_ttest_count += 1

                        #ANOVA prepare
                        neuron_data_abs[loc] = H[task_info[rule]['epoch_info'][epoch][0]:task_info[rule]['epoch_info'][epoch][1],\
                            task_info[rule][in_loc] == loc, neuron].mean(axis=0)
                        neuron_data_norm[loc] = neuron_data_abs[loc]/fix_level.mean(axis=0)-1
                        if norm:
                            neuron_data[loc] = neuron_data_norm[loc]
                        else:
                            neuron_data[loc] = neuron_data_abs[loc]
                        firerate_abs.append(neuron_data_abs[loc].mean())
                        firerate_norm.append(neuron_data_norm[loc].mean())
                        firerate.append(neuron_data[loc].mean())

                    if max(firerate_abs) > abs_active_thresh and paired_ttest_count > 0:

                        max_index = firerate.index(max(firerate))

                        neuron_info['active_neurons'].append((neuron,None,max_index))
                        #ANOVA
                        data_frame = pd.DataFrame(neuron_data)
                        data_frame_melt = data_frame.melt()
                        data_frame_melt.columns = ['Location','Fire_rate']
                        model = ols('Fire_rate~C(Location)',data=data_frame_melt).fit()
                        anova_table = anova_lm(model, typ = 2)

                        if anova_table['PR(>F)'][0] <= annova_p_value:
                            neuron_info['selective_neurons'].append((neuron,anova_table['PR(>F)'][0],max_index))

                            if max(firerate_norm) < 0:
                                neuron_info['inh_neurons'].append((neuron,anova_table['PR(>F)'][0],max_index))
                            elif min(firerate_norm) >= 0:
                            #else:
                                neuron_info['exh_neurons'].append((neuron,anova_table['PR(>F)'][0],max_index))
                            else:
                                neuron_info['mix_neurons'].append((neuron,anova_table['PR(>F)'][0],max_index))
                    
                    neuron_info['firerate_loc_order'].append(firerate)

                neuron_info['firerate_loc_order'] = \
                    np.array(neuron_info['firerate_loc_order'])

            #if len(trial_list) == 1:
                #save_name = model_dir+'/neuron_info_'+rule+'_'+epoch+'_'+str(trial_list[0])+'.pkl'
            #else:
                #save_name = model_dir+'/neuron_info_'+rule+'_'+epoch+'_'+\
                    #str(trial_list[0])+'_'+str(trial_list[-1])+'_step'+str(trial_list[1]-trial_list[0])+'.pkl'

                with open(save_name,'wb') as inf:
                    pickle.dump(neuron_info,inf)


def get_mon_trials(m_nm, chosen_direction, chosen_epoch_inlocs, m_or_n):

    filt_epoch_dir = chosen_epoch_inlocs == chosen_direction

    if m_or_n == 'match':
        return [z[0] and z[1] for z in zip(filt_epoch_dir,m_nm)]
    elif m_or_n == 'non-match':
        return [z[0] and not z[1] for z in zip(filt_epoch_dir,m_nm)]


def gen_PSTH_log_mon(
                hp,
                trial_list,
                model_dir,
                rule,
                seltive_epoch,
                n_types=('exh_neurons','mix_neurons'),
                norm = True,
                oppo_sel_dir = False):

    with open(model_dir+'/task_info.pkl','rb') as tinf:
        task_info = pickle.load(tinf)

    PSTH_log = dict()
    PSTH_log['match'] = dict()
    PSTH_log['non-match'] = dict()

    if seltive_epoch == 'stim2' or seltive_epoch == 'delay2':
        in_loc = 'in_loc_2'
    else:
        in_loc = 'in_loc'
    chosen_epoch_inlocs = task_info[rule][in_loc]

    stim1_locs = task_info[rule]['in_loc']
    stim2_locs = task_info[rule]['in_loc_2']
    m_nm = [st[0]==st[1] for st in zip(stim1_locs,stim2_locs)]

    print('\tGenerating PSTH '+rule+' '+seltive_epoch)

    count = 0
    for trial_num in trial_list:
        
        H = Get_H(hp,model_dir,trial_num,rule,save_H=False,task_mode='test',)

        n_info_file = model_dir+'/'+str(trial_num)+'/neuron_info_'+rule+'_'+seltive_epoch+'.pkl'
        with open(n_info_file,'rb') as ninf:
            n_info = pickle.load(ninf)

        n_list = list()
        for n_type in n_types:
            n_list = list(set(n_list+n_info[n_type]))

        for m_or_n in ['match','non-match']:
            psth_log_temp = list()

            for neuron in n_list:

                if oppo_sel_dir:
                    loc = (neuron[2]+len(task_info[rule]['in_loc_set'])//2)%len(task_info[rule]['in_loc_set'])
                    if len(task_info[rule]['in_loc_set'])%2:
                        psth_temp = (H[:,get_mon_trials(m_nm, loc, chosen_epoch_inlocs, m_or_n), neuron[0]].mean(axis=1)+\
                            H[:,get_mon_trials(m_nm, (loc+1)%len(task_info[rule]['in_loc_set']), chosen_epoch_inlocs, m_or_n), neuron[0]].mean(axis=1))/2.0
                    else:
                        psth_temp = H[:,get_mon_trials(m_nm, loc, chosen_epoch_inlocs, m_or_n), neuron[0]].mean(axis=1)

                else:
                    loc = neuron[2]
                    psth_temp = H[:,get_mon_trials(m_nm, loc, chosen_epoch_inlocs, m_or_n), neuron[0]].mean(axis=1)

            
                fix_level = H[task_info[rule]['epoch_info']['fix1'][0]:task_info[rule]['epoch_info']['fix1'][1], \
                    get_mon_trials(m_nm, loc, chosen_epoch_inlocs, m_or_n), neuron[0]].mean(axis=1).mean(axis=0)

                psth_norm = psth_temp/fix_level-1

                if norm:
                    psth_log_temp.append(psth_norm)
                else:
                    psth_log_temp.append(psth_temp)


            try:
                PSTH_log[m_or_n][trial_num] = np.array(psth_log_temp)
            except:
                pass

        count+=1
        process = count/len(trial_list)*100
        print ("\r\t processing... %.1f%%"%(process), end="",flush=True)

    print('\n\tfinish')

    return PSTH_log

def plot_PSTH_mon(
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
        PSTH_log = gen_PSTH_log_mon(hp,trial_list,model_dir,rule,epoch,n_types=n_types,norm=norm)

        if plot_oppo_dir:
            PSTH_log_oppo = gen_PSTH_log_mon(hp,trial_list,model_dir,rule,epoch,n_types=n_types,norm=norm,oppo_sel_dir=plot_oppo_dir)
            for m_or_n in ['match','non-match']:
                for key,value in PSTH_log_oppo[m_or_n].items():
                    PSTH_log_oppo[m_or_n][key] = value.mean(axis=0)

    for m_or_n in ['match','non-match']:
        n_number[m_or_n] = dict()
        for key,value in PSTH_log[m_or_n].items():
            n_number[m_or_n][key] = np.size(value,0)
            PSTH_log[m_or_n][key] = value.mean(axis=0)

    data_to_plot = dict()
    maturation = ["infant","young","adult"]
    data_types = ["PSTH","n_num","growth"]
    if plot_oppo_dir:
        data_types.append("PSTH_oppo")
    
    for m_key in maturation:
        data_to_plot[m_key] = dict()
        for m_or_n in ['match','non-match']:
            data_to_plot[m_key][m_or_n] = dict()
            for data_type in data_types:
                data_to_plot[m_key][m_or_n][data_type] = list()
    
    for trial_num in trial_list:
        growth = log['perf_'+rule][trial_num//log['trials'][1]]

        if growth <= hp['infancy_target_perf']:
            m_key = "infant"
        elif growth <= hp['young_target_perf']:
            m_key = "young"
        else:
            m_key = "adult"
        
        for m_or_n in ['match','non-match']:
            data_to_plot[m_key][m_or_n]["PSTH"].append(PSTH_log[m_or_n][trial_num])
            data_to_plot[m_key][m_or_n]["growth"].append(growth)
            data_to_plot[m_key][m_or_n]["n_num"].append(n_number[m_or_n][trial_num])
            if plot_oppo_dir:
                data_to_plot[m_key][m_or_n]["PSTH_oppo"].append(PSTH_log_oppo[m_or_n][trial_num])

    for m_key in maturation:
        for data_type in data_types:
            for m_or_n in ['match','non-match']:
                data_to_plot[m_key][m_or_n][data_type] = np.array(data_to_plot[m_key][m_or_n][data_type]).mean(axis=0)

    # plot #
    save_path = 'figure/figure_'+model_dir+'/'+rule+'/'+epoch+'/'+'_'.join(n_types)+'/'
    if len(trial_list) == 1:
        step = 'None'
    else:
        step = str(trial_list[1]-trial_list[0])
    trial_range = str((trial_list[0],trial_list[-1]))
    title = 'Rule:'+rule+' Epoch:'+epoch+' Neuron_type:'+'_'.join(n_types)+' trial range:'+trial_range+' step:'+step

    colors = {"infant":"green","young":"blue","adult":"red",}

    fig,axes = plt.subplots(2,1,figsize=(14,21))
    fig.suptitle(title)

    for ax_num, m_or_n in enumerate(['match','non-match']):
        for m_key in maturation:
            try:
                axes[ax_num].plot(np.arange(len(data_to_plot[m_key][m_or_n]["PSTH"]))*hp['dt']/1000, data_to_plot[m_key][m_or_n]["PSTH"],\
                    label= m_key+'_%.2f'%(data_to_plot[m_key][m_or_n]["growth"])+'_n%d'%(data_to_plot[m_key][m_or_n]["n_num"]), color=colors[m_key])
            except:
                pass
            if plot_oppo_dir:
                try:
                    axes[ax_num].plot(np.arange(len(data_to_plot[m_key][m_or_n]["PSTH_oppo"]))*hp['dt']/1000, data_to_plot[m_key][m_or_n]["PSTH_oppo"],\
                        label= m_key+'_opposite_sel_dir', color=colors[m_key], linestyle = '--')
                except:
                    pass
        axes[ax_num].legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
        axes[ax_num].set_title(m_or_n)

    mkdir_p(save_path)
    plt.savefig(save_path+rule+'_'+epoch+'_'+trial_range+'_step_'+step+'_PSTH.pdf',bbox_inches='tight')
    plt.savefig(save_path+rule+'_'+epoch+'_'+trial_range+'_step_'+step+'_PSTH.eps',bbox_inches='tight')
    plt.savefig(save_path+rule+'_'+epoch+'_'+trial_range+'_step_'+step+'_PSTH.png',bbox_inches='tight')

    plt.close()

def plot_PSTH_passive_mbymon(
        hp,
        log,
        model_dir, 
        mature_rule,
        epoch,
        trial_list,
        n_types=('exh_neurons','mix_neurons'),
        plot_oppo_dir = False,
        norm = True,
        PSTH_log = None,
        rule = 'match_or_non_passive',
        ):

    print("Start ploting PSTH")
    print("\trule: "+rule+" selective epoch: "+epoch)

    n_number = dict()

    if PSTH_log is None:
        PSTH_log = gen_PSTH_log_mon(hp,trial_list,model_dir,rule,epoch,n_types=n_types,norm=norm)

        if plot_oppo_dir:
            PSTH_log_oppo = gen_PSTH_log_mon(hp,trial_list,model_dir,rule,epoch,n_types=n_types,norm=norm,oppo_sel_dir=plot_oppo_dir)
            for m_or_n in ['match','non-match']:
                for key,value in PSTH_log_oppo[m_or_n].items():
                    PSTH_log_oppo[m_or_n][key] = value.mean(axis=0)

    for m_or_n in ['match','non-match']:
        n_number[m_or_n] = dict()
        for key,value in PSTH_log[m_or_n].items():
            n_number[m_or_n][key] = np.size(value,0)
            PSTH_log[m_or_n][key] = value.mean(axis=0)

    data_to_plot = dict()
    maturation = ["infant","young","adult"]
    data_types = ["PSTH","n_num","growth"]
    if plot_oppo_dir:
        data_types.append("PSTH_oppo")
    
    for m_key in maturation:
        data_to_plot[m_key] = dict()
        for m_or_n in ['match','non-match']:
            data_to_plot[m_key][m_or_n] = dict()
            for data_type in data_types:
                data_to_plot[m_key][m_or_n][data_type] = list()
    
    for trial_num in trial_list:
        growth = log['perf_'+mature_rule][trial_num//log['trials'][1]]

        if growth <= hp['infancy_target_perf']:
            m_key = "infant"
        elif growth <= hp['young_target_perf']:
            m_key = "young"
        else:
            m_key = "adult"
        
        for m_or_n in ['match','non-match']:
            data_to_plot[m_key][m_or_n]["PSTH"].append(PSTH_log[m_or_n][trial_num])
            data_to_plot[m_key][m_or_n]["growth"].append(growth)
            data_to_plot[m_key][m_or_n]["n_num"].append(n_number[m_or_n][trial_num])
            if plot_oppo_dir:
                data_to_plot[m_key][m_or_n]["PSTH_oppo"].append(PSTH_log_oppo[m_or_n][trial_num])

    for m_key in maturation:
        for data_type in data_types:
            for m_or_n in ['match','non-match']:
                data_to_plot[m_key][m_or_n][data_type] = np.array(data_to_plot[m_key][m_or_n][data_type]).mean(axis=0)

    # plot #
    save_path = 'figure/figure_'+model_dir+'/'+rule+'/'+epoch+'/'+'_'.join(n_types)+'/'
    if len(trial_list) == 1:
        step = 'None'
    else:
        step = str(trial_list[1]-trial_list[0])
    trial_range = str((trial_list[0],trial_list[-1]))
    title = 'Rule:'+rule+' Epoch:'+epoch+' Neuron_type:'+'_'.join(n_types)+' trial range:'+trial_range+' step:'+step

    colors = {"infant":"green","young":"blue","adult":"red",}

    fig,axes = plt.subplots(2,1,figsize=(14,21))
    fig.suptitle(title)

    for ax_num, m_or_n in enumerate(['match','non-match']):
        for m_key in maturation:
            try:
                axes[ax_num].plot(np.arange(len(data_to_plot[m_key][m_or_n]["PSTH"]))*hp['dt']/1000, data_to_plot[m_key][m_or_n]["PSTH"],\
                    label= m_key+'_%.2f'%(data_to_plot[m_key][m_or_n]["growth"])+'_n%d'%(data_to_plot[m_key][m_or_n]["n_num"]), color=colors[m_key])
            except:
                pass
            if plot_oppo_dir:
                try:
                    axes[ax_num].plot(np.arange(len(data_to_plot[m_key][m_or_n]["PSTH_oppo"]))*hp['dt']/1000, data_to_plot[m_key][m_or_n]["PSTH_oppo"],\
                        label= m_key+'_opposite_sel_dir', color=colors[m_key], linestyle = '--')
                except:
                    pass
        axes[ax_num].legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
        axes[ax_num].set_title(m_or_n)

    mkdir_p(save_path)
    plt.savefig(save_path+rule+'_'+mature_rule+'_'+epoch+'_'+trial_range+'_step_'+step+'_PSTH.pdf',bbox_inches='tight')
    plt.savefig(save_path+rule+'_'+mature_rule+'_'+epoch+'_'+trial_range+'_step_'+step+'_PSTH.eps',bbox_inches='tight')
    plt.savefig(save_path+rule+'_'+mature_rule+'_'+epoch+'_'+trial_range+'_step_'+step+'_PSTH.png',bbox_inches='tight')

    plt.close()

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model_dir = 'data/set5_exp2'
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)
    #trial_list = range(704000,1122560+1,1280)
    #trial_list = [1000960,1002240,1003520,1004800]
    #trial_list = [1000960,]
    trial_list = range(999680,log['trials'][-1]+1,1280)

    recompute = False

    print_basic_info(hp,log,model_dir,smooth_growth=True,smooth_window=5)

    print("compute H")
    compute_H(hp,log,model_dir,  rules=['match_or_non','match_or_non_easy','match_or_non_passive'], trial_list=trial_list, recompute=recompute,)

    print("Generate Info")
    generate_neuron_info_mon(hp,model_dir,epochs=['stim1','delay1','stim2','delay2'],trial_list=trial_list,rules=['match_or_non','match_or_non_easy','match_or_non_passive'],recompute=recompute,)

    print('\n\nPlot')
    '''
    plot_PSTH_mon(hp, log, model_dir, 'match_or_non', 'stim1', trial_list=trial_list, plot_oppo_dir = True)
    plot_PSTH_mon(hp, log, model_dir, 'match_or_non', 'delay1', trial_list=trial_list, plot_oppo_dir = True)
    plot_PSTH_mon(hp, log, model_dir, 'match_or_non', 'stim2', trial_list=trial_list, plot_oppo_dir = True)
    plot_PSTH_mon(hp, log, model_dir, 'match_or_non', 'delay2', trial_list=trial_list, plot_oppo_dir = True)
    plot_PSTH_mon(hp, log, model_dir, 'match_or_non_easy', 'stim1', trial_list=trial_list, plot_oppo_dir = True)
    plot_PSTH_mon(hp, log, model_dir, 'match_or_non_easy', 'delay1', trial_list=trial_list, plot_oppo_dir = True)
    plot_PSTH_mon(hp, log, model_dir, 'match_or_non_easy', 'stim2', trial_list=trial_list, plot_oppo_dir = True)
    plot_PSTH_mon(hp, log, model_dir, 'match_or_non_easy', 'delay2', trial_list=trial_list, plot_oppo_dir = True)
    plot_PSTH_mon(hp, log, model_dir, 'match_or_non_passive', 'stim1', trial_list=trial_list, plot_oppo_dir = True)
    plot_PSTH_mon(hp, log, model_dir, 'match_or_non_passive', 'delay1', trial_list=trial_list, plot_oppo_dir = True)
    plot_PSTH_mon(hp, log, model_dir, 'match_or_non_passive', 'stim2', trial_list=trial_list, plot_oppo_dir = True)
    plot_PSTH_mon(hp, log, model_dir, 'match_or_non_passive', 'delay2', trial_list=trial_list, plot_oppo_dir = True)
    '''
    plot_PSTH_passive_mbymon(hp, log, model_dir, 'match_or_non', 'delay2', trial_list=trial_list, plot_oppo_dir = True)
    plot_PSTH_passive_mbymon(hp, log, model_dir, 'match_or_non', 'delay1', trial_list=trial_list, plot_oppo_dir = True)
    plot_PSTH_passive_mbymon(hp, log, model_dir, 'match_or_non', 'stim1', trial_list=trial_list, plot_oppo_dir = True)
    plot_PSTH_passive_mbymon(hp, log, model_dir, 'match_or_non', 'stim2', trial_list=trial_list, plot_oppo_dir = True)
