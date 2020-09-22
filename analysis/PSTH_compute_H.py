# basic packages #
import os
import numpy as np
import tensorflow as tf
import pickle
# trial generation and network building #

import sys
sys.path.append('.')
from utils import tools

from task_and_network.task import generate_trials
from task_and_network.network import Model

def gen_task_info(hp,
                log,
                model_dir,  
                rules, 
                trial_list,
                return_trial_store=False,):
    task_info_file = model_dir+'/task_info.pkl'
    if os.path.isfile(task_info_file):
        with open(task_info_file,'rb') as tinfr:
            task_info = pickle.load(tinfr)
    else:
        task_info = dict()
    
    trial_store = dict()

    print("Epoch information:")
    for rule in rules:
        task_info[rule] = dict()
        trial_store[rule] = generate_trials(rule, hp, 'test', noise_on=False)
        
        n_stims = len([ep for ep in trial_store[rule].epochs.keys() if 'stim' in ep])
        stim_loc_log_len = int(len(trial_store[rule].input_loc)/n_stims)
        task_info[rule]['in_loc'] = np.array([np.argmax(i) for i in trial_store[rule].input_loc[:stim_loc_log_len]])
        if n_stims != 1:
            for nst in range(2,n_stims+1):
                task_info[rule]['in_loc_'+str(nst)] = \
                    np.array([np.argmax(i) for i in trial_store[rule].input_loc[(nst-1)*stim_loc_log_len:nst*stim_loc_log_len]])

        task_info[rule]['in_loc_set'] = sorted(set(task_info[rule]['in_loc']))
        task_info[rule]['epoch_info'] = trial_store[rule].epochs
        print('\t'+rule+':')
        for e_name, e_time in task_info[rule]['epoch_info'].items():
            print('\t\t'+e_name+':',e_time)

    with open(task_info_file,'wb') as tinf:
        pickle.dump(task_info, tinf)

    if return_trial_store:
        return trial_store

def compute_H_(hp, model_dir, rule, trial_num, trial=None, task_mode='test'):

    if trial is None:
        trial = generate_trials(rule, hp, task_mode, noise_on=False)
    
    sub_dir = model_dir+'/'+str(trial_num)+'/'
    model = Model(sub_dir, hp=hp)
    with tf.Session() as sess:
        model.restore()
        feed_dict = tools.gen_feed_dict(model, trial, hp)
        h = sess.run(model.h, feed_dict=feed_dict)
    return h

def compute_H(
            hp,
            log,
            model_dir,  
            rules=None, 
            trial_list=None, 
            recompute=False,
            save_H_pickle=True,
            ):
        
    if rules is not None:
        rules = rules
    else:
        rules = hp['rule_trains']
        
    if trial_list is not None:
        trial_list = trial_list
    else:
        trial_list = log['trials']

    trial_store = gen_task_info(hp,log,model_dir,rules,trial_list,return_trial_store=True)

    for rule in rules:

        for trial_num in trial_list:
            H_file = model_dir+'/'+str(trial_num)+'/H_'+rule+'.pkl'
            if recompute or not os.path.isfile(H_file):
                H_ = compute_H_(hp, model_dir, rule, trial_num, trial = trial_store[rule])
                with open(H_file,'wb') as wh:
                    pickle.dump(H_,wh)

def Get_H(hp,model_dir,trial_num,rule,save_H=False,task_mode='test',):

    H_file = model_dir+'/'+str(trial_num)+'/H_'+rule+'.pkl'

    if os.path.isfile(H_file):
        with open(H_file,'rb') as hf:
            H = pickle.load(hf)

    else:
        H = compute_H_(hp, model_dir, rule, trial_num, trial=None, task_mode=task_mode)
        if save_H:
            with open(H_file,'wb') as wh:
                pickle.dump(H,wh)
    return H  


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--modeldir', type=str, default='data/6tasks')
    parser.add_argument('--rules', type=str, default='None')
    parser.add_argument('--trial_start', type=str, default='None')
    parser.add_argument('--trial_end', type=str, default='None')
    parser.add_argument('--trial_step', type=str, default='None')
    parser.add_argument('--recompute', type=bool, default=False)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model_dir = args.modeldir
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)
    

    if args.rules == 'None':
        rules = None
    else:
        rules = args.rules.split(',')

    if args.trial_start == 'None':
        start = log['trials'][0]
    else:
        start = int(args.trial_start)

    if args.trial_end == 'None':
        end = log['trials'][-1]
    else:
        end = int(args.trial_end)

    if args.trial_end == 'None':
        step = log['trials'][1]
    else:
        step = int(args.trial_step)


    #for 6tasks folder for odr/odrd stage 
    #start = 520960
    #end = 628480
    #step = 1280*1#3#21#12#7#12

    recompute = args.recompute
    '''
    recompute = True
    start = 520960
    end = 520960+1280
    step = 1280
    '''

    #compute_H(hp,log,model_dir,  rules=rules, trial_list=range(start,end+1,step), recompute=recompute,)
    compute_H(hp,log,model_dir,  rules=['odr','odrd'], trial_list=range(start,end+1,step), recompute=recompute,)


    