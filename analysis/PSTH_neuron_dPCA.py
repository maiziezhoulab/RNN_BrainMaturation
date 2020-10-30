# basic packages #
import os
import numpy as np
import pickle

import sys
sys.path.append('.')
from utils.tools import mkdir_p, smooth

# plot #
from matplotlib import pyplot as plt

# dPCA #
from dPCA import dPCA

from .PSTH_compute_H import Get_H

def neuron_dPCA(hp,
                log,
                model_dir,
                trial_num,
                rule,
                task_mode='test',
                appoint_loc_analysis=False,
                appoint_locs=[0,4],
                invert_tcomp_y=False,
                invert_scomp_y=False,
                invert_stcomp_y=False,
                tcomp_ylim=(None,None),
                scomp_ylim=(None,None),
                stcomp_ylim=(None,None),):

    task_info_file = model_dir+'/task_info.pkl'
    with open(task_info_file,'rb') as tinfr:
        task_info = pickle.load(tinfr)
    
    H = Get_H(hp,model_dir,trial_num,rule,task_mode=task_mode,)

    if appoint_loc_analysis:
        locs = appoint_locs
        H = H[:,[lc in appoint_locs for lc in task_info[rule]['in_loc']],:]
    else:
        locs = task_info[rule]['in_loc_set']

    trial_per_loc = int(len(task_info[rule]['in_loc'])/len(task_info[rule]['in_loc_set']))
    loc_num = len(locs)
    time_len = np.size(H,0)
    neuron_num = np.size(H,2)
    reform_H = np.zeros((trial_per_loc,neuron_num,loc_num,time_len))

    for i in range(trial_per_loc):
        for n in range(neuron_num):
            for lo in range(loc_num):
                for t in range(time_len):
                    reform_H[i,n,lo,t] = H[t,lo*trial_per_loc+i,n]

    # trial-average data
    R = np.mean(reform_H,0)

    # center data
    R -= np.mean(R.reshape((neuron_num,-1)),1)[:,None,None]

    dpca = dPCA.dPCA(labels='st',regularizer='auto')
    dpca.protect = ['t']

    Z = dpca.fit_transform(R,reform_H)

    time = np.arange(time_len)

    fig,axes = plt.subplots(1,3,figsize=(16,7))

    for loc in range(loc_num):
        axes[0].plot(time*hp['dt']/1000,Z['t'][0,loc],label='loc:'+str(locs[loc]))

    axes[0].set_title('1st time component')
    axes[0].set_ylim(tcomp_ylim[0],tcomp_ylim[1])
    if invert_tcomp_y:
        axes[0].invert_yaxis()
    axes[0].legend()
    
    for loc in range(loc_num):
        axes[1].plot(time*hp['dt']/1000,Z['s'][0,loc],label='loc:'+str(locs[loc]))
    
    axes[1].set_title('1st stimulus component')
    axes[1].set_ylim(scomp_ylim[0],scomp_ylim[1])
    if invert_scomp_y:
        axes[1].invert_yaxis()
    axes[1].legend()

    for loc in range(loc_num):
        axes[2].plot(time*hp['dt']/1000,Z['st'][0,loc],label='loc:'+str(locs[loc]))
    
    axes[2].set_title('1st mixing component')
    axes[2].set_ylim(stcomp_ylim[0],stcomp_ylim[1])
    if invert_stcomp_y:
        axes[2].invert_yaxis()
    axes[2].legend()
    plt.show()

def real_neuron_dPCA(neuron_file='data/adultFiringRate8trials.txt',
                    time_len=176,
                    trial_per_loc=8,
                    dt=20,
                    appoint_loc_analysis=False,
                    appoint_locs=[1,5],
                    invert_tcomp_y=False,
                    invert_scomp_y=False,
                    invert_stcomp_y=False,
                    tcomp_ylim=(None,None),
                    scomp_ylim=(None,None),
                    stcomp_ylim=(None,None),):

    neuron_reaction = list()
    locs_ = list()
    
    with open(neuron_file,'r') as nf:
        for line in nf:
            line = line.rstrip('\n').split(',')
            for i in range(len(line)):
                line[i] = float(line[i])
            locs_.append(int(line[0]))            
            neuron_reaction.append(line[1:])
    neuron_reaction = np.array(neuron_reaction)

    if appoint_loc_analysis:
        locs_ = appoint_locs

    #remove duplicates but keep the order of the locs
    locs = list(set(locs_))
    locs.sort(key=locs_.index)

    loc_num = len(locs)

    neuron_num = np.size(neuron_reaction,1)
    trialR = np.zeros((trial_per_loc,neuron_num,loc_num,time_len))

    for i in range(trial_per_loc):
        for n in range(neuron_num):
            for lo in range(loc_num):
                for t in range(time_len):
                    trialR[i,n,lo,t] = neuron_reaction[(locs[lo]-1)*trial_per_loc*time_len+i*time_len+t,n]

    # trial-average data
    R = np.mean(trialR,0)
    #print(R.min())

    # center data
    R -= np.mean(R.reshape((neuron_num,-1)),1)[:,None,None]
    #print(R.min())

    dpca = dPCA.dPCA(labels='st',regularizer='auto')
    dpca.protect = ['t']

    Z = dpca.fit_transform(R,trialR)

    time = np.arange(time_len)

    fig,axes = plt.subplots(1,3,figsize=(16,7))

    for loc in range(loc_num):
        axes[0].plot(time*dt/1000,smooth(Z['t'][0,loc],3),label='loc:'+str(locs[loc]))

    axes[0].set_title('1st time component')
    axes[0].set_ylim(tcomp_ylim[0],tcomp_ylim[1])
    if invert_tcomp_y:
        axes[0].invert_yaxis()
    axes[0].legend()
    
    for loc in range(loc_num):
        axes[1].plot(time*dt/1000,smooth(Z['s'][0,loc],3),label='loc:'+str(locs[loc]))
    
    axes[1].set_title('1st stimulus component')
    axes[1].set_ylim(scomp_ylim[0],scomp_ylim[1])
    if invert_scomp_y:
        axes[1].invert_yaxis()
    axes[1].legend()

    for loc in range(loc_num):
        axes[2].plot(time*dt/1000,smooth(Z['st'][0,loc],3),label='loc:'+str(locs[loc]))
    
    axes[2].set_title('1st mixing component')
    axes[2].set_ylim(stcomp_ylim[0],stcomp_ylim[1])
    if invert_stcomp_y:
        axes[2].invert_yaxis()
    axes[2].legend()
    plt.show()
