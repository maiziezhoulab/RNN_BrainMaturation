# basic packages #
import os
import numpy as np
import pickle

# for ANOVA and paired T test analysis and plot #
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from .PSTH_compute_H import Get_H


def generate_neuron_info(
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
                            task_info[rule]['in_loc'] == loc, neuron].mean(axis=0)
                        stim1_level = H[task_info[rule]['epoch_info']['stim1'][0]:task_info[rule]['epoch_info']['stim1'][1],\
                            task_info[rule]['in_loc'] == loc, neuron].mean(axis=0)
                        paired_ttest_result = ttest_rel(fix_level, stim1_level)[1]
                        if paired_ttest_result <= paired_ttest_p:
                            paired_ttest_count += 1

                        #ANOVA prepare
                        neuron_data_abs[loc] = H[task_info[rule]['epoch_info'][epoch][0]:task_info[rule]['epoch_info'][epoch][1],\
                            task_info[rule]['in_loc'] == loc, neuron].mean(axis=0)
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


        