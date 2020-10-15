# basic packages #
import os
import numpy as np
import pickle

import sys
sys.path.append('.')
from utils.tools import mkdir_p

# plot #
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

#PCA#
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .PSTH_compute_H import Get_H

def neuron_PCA(hp,log,model_dir,trial_num,rule,task_mode='test',n_components=2):

    task_info_file = model_dir+'/task_info.pkl'
    with open(task_info_file,'rb') as tinfr:
        task_info = pickle.load(tinfr)
    
    H = Get_H(hp,model_dir,trial_num,rule,task_mode=task_mode,)
    time_length = len(H[:,0,0])

    save_path = 'figure/figure_'+model_dir.rstrip('/').split('/')[-1]+'/'+rule+'/'

#8conc_subplot_time

    for loc in task_info[rule]['in_loc_set']:
        if loc == 0:
            loc_meaned = H[:,task_info[rule]['in_loc'] == loc,:].mean(axis=1)
        else:
            loc_meaned = np.concatenate((loc_meaned,H[:,task_info[rule]['in_loc'] == loc,:].mean(axis=1)),axis=0)

    pca=PCA(n_components=n_components)
    n_pca=pca.fit_transform(StandardScaler().fit_transform(loc_meaned))
    #n_pca=pca.fit_transform(loc_meaned)

    fig= plt.figure(figsize=(10,5*n_components))
    time_line = np.arange(time_length)*hp['dt']/1000
    growth = log['perf_'+rule][trial_num//log['trials'][1]]

    for n_comp in range(n_components):
        index = (n_components,1,n_comp+1)
        ax = fig.add_subplot(index[0],index[1],index[2])
        ax.set_title('component:'+str(n_comp+1))

        for loc in task_info[rule]['in_loc_set']:
            pc = n_pca[loc*time_length:(loc+1)*time_length,n_comp]
            ax.plot(time_line,pc,label='loc:'+str(loc))
        ax.legend()

    plt.savefig(save_path+rule+'_comp'+str(n_components)+'_neuron_pca_trial_num_'+str(trial_num)+'_all_loc_bycomp_conc_pref%.4f_std.pdf'%(growth),bbox_inches='tight')
    plt.close()

#8conc_by_subplot
    '''
    fig= plt.figure(figsize=(20,10))
    growth = log['perf_'+rule][trial_num//log['trials'][1]]

    if growth <= hp['early_target_perf']:
        color = 'green'
    elif growth <= hp['mid_target_perf']:
        color = 'blue'
    else:
        color = 'red'

    for loc in task_info[rule]['in_loc_set']:
        if loc == 0:
            loc_meaned = H[:,task_info[rule]['in_loc'] == loc,:].mean(axis=1)
        else:
            loc_meaned = np.concatenate((loc_meaned,H[:,task_info[rule]['in_loc'] == loc,:].mean(axis=1)),axis=0)

    pca=PCA(n_components=n_components)
    #n_pca=pca.fit_transform(StandardScaler().fit_transform(loc_meaned))
    n_pca=pca.fit_transform(loc_meaned)
    for loc in task_info[rule]['in_loc_set']:
        index = 241+loc
        pc = n_pca[loc*time_length:(loc+1)*time_length,:]
        if n_components == 2:
            ax = fig.add_subplot(index)
            ax.plot(pc[:,0],pc[:,1],color=color)
        elif n_components == 3:
            ax = fig.add_subplot(index, projection='3d')
            ax.plot3D(pc[:,0],pc[:,1],pc[:,2],color=color)
        ax.set_title('loc:'+str(loc))

    plt.savefig(save_path+rule+'_comp'+str(n_components)+'_neuron_pca_trial_num_'+str(trial_num)+'_all_loc_bysub_conc_pref%.4f.pdf'%(growth),bbox_inches='tight')
    plt.close()
    '''


#8conc_by_color
    '''
    growth = log['perf_'+rule][trial_num//log['trials'][1]]
    fig= plt.figure(figsize=(10,10))
    if n_components == 2:
        ax = plt.axes()
    elif n_components == 3:
        ax = plt.axes(projection='3d')

    for loc in task_info[rule]['in_loc_set']:
        if loc == 0:
            loc_meaned = H[:,task_info[rule]['in_loc'] == loc,:].mean(axis=1)
        else:
            loc_meaned = np.concatenate((loc_meaned,H[:,task_info[rule]['in_loc'] == loc,:].mean(axis=1)),axis=0)

    pca=PCA(n_components=n_components)
    n_pca=pca.fit_transform(StandardScaler().fit_transform(loc_meaned))
    #n_pca=pca.fit_transform(loc_meaned)

    for loc in task_info[rule]['in_loc_set']:
        pc = n_pca[loc*time_length:(loc+1)*time_length,:]
        if n_components == 2:
            ax.plot(pc[:,0],pc[:,1],label='loc:'+str(loc))
        elif n_components == 3:
            ax.plot3D(pc[:,0],pc[:,1],pc[:,2],label='loc:'+str(loc))
    plt.legend()

    plt.savefig(save_path+rule+'_comp'+str(n_components)+'_neuron_pca_trial_num_'+str(trial_num)+'_all_loc_bycolor_conc_pref%.4f_std.pdf'%(growth),bbox_inches='tight')
    plt.close()
    '''

#indipendent_by_sub_plot
    '''
    fig= plt.figure(figsize=(20,10))

    for loc in task_info[rule]['in_loc_set']:
        loc_meaned = H[:,task_info[rule]['in_loc'] == loc,:].mean(axis=1)
        pca=PCA(n_components=3)
        #n_pca=pca.fit_transform(StandardScaler().fit_transform(loc_meaned))
        n_pca=pca.fit_transform(loc_meaned)
        index = 240+loc
        ax = fig.add_subplot(index, projection='3d')
        pc1 = n_pca[:,0]
        pc2 = n_pca[:,1]
        pc3 = n_pca[:,2]
        ax.scatter3D(pc1,pc2,pc3)
        ax.set_title('loc:'+str(loc))

    plt.savefig(save_path+rule+'_neuron_pca_trial_num_'+str(trial_num)+'_all_loc.pdf',bbox_inches='tight')
    plt.close()
    '''

def real_neuron_PCA(neuron_file='data/adultneuron.txt', time_len=176, direc_num=8,n_components=2):

    neuron_reaction = list()
    with open(neuron_file,'r') as nf:
        for line in nf:
            line = line.rstrip('\n').split(',')
            for i in range(len(line)):
                line[i] = float(line[i])
            neuron_reaction.append(line)
    neuron_reaction = np.array(neuron_reaction).T

    pca=PCA(n_components=n_components)
    n_pca=pca.fit_transform(StandardScaler().fit_transform(neuron_reaction))
    #n_pca=pca.fit_transform(neuron_reaction)

    '''

    fig= plt.figure(figsize=(10,10))

    if n_components == 2:
        ax = plt.axes()
    elif n_components == 3:
        ax = plt.axes(projection='3d')

    for loc in range(direc_num):
        pc = n_pca[loc*time_len:(loc+1)*time_len,:]
        if n_components == 2:
            ax.plot(pc[:,0],pc[:,1],label='loc:'+str(loc+1))
        elif n_components == 3:
            ax.plot3D(pc[:,0],pc[:,1],pc[:,2],label='loc:'+str(loc+1))
        
    plt.legend()

    sample_name = neuron_file.split('/')[-1].split('.')[0]
    save_path = 'figure/real_neuron_pca/'
    mkdir_p(save_path)
    plt.savefig(save_path+sample_name+'comp'+str(n_components)+'_all_loc_bycolor_conc_std.pdf',bbox_inches='tight')
    plt.close()
    '''

    #8conc_subplot_time

    fig= plt.figure(figsize=(10,5*n_components))
    time_line = np.arange(time_len)*20/1000

    for n_comp in range(n_components):
        index = (n_components,1,n_comp+1)
        ax = fig.add_subplot(index[0],index[1],index[2])
        ax.set_title('component:'+str(n_comp+1))

        for loc in range(direc_num):
            pc = n_pca[loc*time_len:(loc+1)*time_len,n_comp]
            ax.plot(time_line,pc,label='loc:'+str(loc))
        ax.legend()

    sample_name = neuron_file.split('/')[-1].split('.')[0]
    save_path = 'figure/real_neuron_pca/'
    mkdir_p(save_path)
    plt.savefig(save_path+sample_name+'comp'+str(n_components)+'_all_loc_bycomp_conc_std.pdf',bbox_inches='tight')
    plt.close()