# basic packages #
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import pickle

from .PSTH_compute_H import Get_H
from .PSTH_gen_neuron_info import generate_neuron_info

def gen_PSTH_log(
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
        
        H = Get_H(hp,model_dir,trial_num,rule,save_H=False,task_mode='test',)

        n_info_file = model_dir+'/'+str(trial_num)+'/neuron_info_'+rule+'_'+seltive_epoch+'.pkl'
        if not os.path.isfile(n_info_file):
            generate_neuron_info(hp,model_dir,epochs=[seltive_epoch,],trial_list=[trial_num,],rules=[rule,],verbose=False)
        with open(n_info_file,'rb') as ninf:
            n_info = pickle.load(ninf)
			

        n_list = list()
        for n_type in n_types:
            n_list = list(set(n_list+n_info[n_type]))

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
