import os
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
from utils import tools

def smooth(ori_array, smooth_window):
    '''smooth window should be an odd number'''
    smoothed_array = list()
    half_wlen = smooth_window//2

    for i in range(half_wlen):
        smoothed_array.append(ori_array[i])
    
    for i in range(half_wlen, len(ori_array)-half_wlen):
        temp = 0
        for j in range(-half_wlen, half_wlen+1):
            temp += ori_array[i+j]
        temp /= smooth_window
        smoothed_array.append(temp)

    for i in range(len(ori_array)-half_wlen, len(ori_array)):
        smoothed_array.append(ori_array[i])

    return smoothed_array

def print_basic_info(hp,log,model_dir,smooth_growth=True,smooth_window=5):

    print('rule trained: ', hp['rule_trains'])
    print('minimum trial number: 0')
    print('maximum trial number: ', log['trials'][-1])
    print('minimum trial step  : ', log['trials'][1])
    print('total number        : ', len(log['trials']))

    fig_pref = plt.figure(figsize=(12,9))
    plted = list()
    for rule in hp['rule_trains']:
        if isinstance (rule, list):
            for single_rule in rule:
                if single_rule not in plted:
                    plted.append(single_rule)
                    if len(log['trials']) == len(log['perf_'+single_rule]):
                        plt.plot(log['trials'], log['perf_'+single_rule], label = single_rule)
                    else:
                        n_multiple = int(len(log['perf_'+single_rule])/len(log['trials']))
                        plt_pref = list()
                        for i in range(len(log['trials'])):
                            temp = 0
                            for n in range(n_multiple):
                                temp += log['perf_'+single_rule][i*n_multiple+n]/n_multiple
                            plt_pref.append(temp)
                        plt.plot(log['trials'], plt_pref, label = single_rule)
                else:
                    continue

        else:
            growth = log['perf_'+rule]
            if smooth_growth:
                plt.plot(log['trials'], smooth(growth,smooth_window), label = rule)
            else:
                plt.plot(log['trials'], growth, label = rule)

    if 'rule_now' in log.keys():
        rule_set = list()
        trial_set = list()

        for i in range(len(log['rule_now'])):
            rules_now = '_'.join(log['rule_now'][i])+'_start'
            if rules_now not in rule_set:
                rule_set.append(rules_now)
                trial_set.append(log['trials'][i])
                plt.axvline(log['trials'][i],color="grey",linestyle = '--')
        for i in range(len(trial_set)):
            plt.text(trial_set[i],0,rule_set[i])

    tools.mkdir_p('figure/figure_'+model_dir.rstrip('/').split('/')[-1]+'/')
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.title('Growth of Performance')
    save_name = 'figure/figure_'+model_dir.rstrip('/').split('/')[-1]+'/growth_of_performance'
    plt.tight_layout()
    plt.savefig(save_name+'.png', transparent=False, bbox_inches='tight')
    plt.savefig(save_name+'.pdf', transparent=False, bbox_inches='tight')
    plt.savefig(save_name+'.eps', transparent=False, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--modeldir', type=str, default='data/6tasks')
    args = parser.parse_args()

    model_dir = args.modeldir
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)

    print_basic_info(hp,log,model_dir)