# basic packages #
import os
import numpy as np
import pickle

import sys
sys.path.append('.')
from utils import tools
#plot
import matplotlib.pyplot as plt

from .PSTH_compute_H import Get_H

def sample_neuron_by_trial(hp,log,model_dir,rule,epoch,trial_list,n_type,):

    with open(model_dir+'/task_info.pkl','rb') as tinf:
        task_info = pickle.load(tinf)

    save_root_folder = 'figure/figure_'+model_dir.rstrip('/').split('/')[-1]+'/'+rule+'/'+epoch+'/sample_neuron/'
    tools.mkdir_p(save_root_folder)

    for trial_num in trial_list:
        H = Get_H(hp,model_dir,trial_num,rule,save_H=False,task_mode='test',)

        with open(model_dir+'/'+str(trial_num)+'/neuron_info_'+rule+'_'+epoch+'.pkl','rb') as inf:
            neuron_info = pickle.load(inf)

        n_list = neuron_info[n_type]
        if not n_list: #if empty
            continue
        #sample_n:(neuron,p,sel_dir)

        perf = log['perf_'+rule][trial_num//log['trials'][1]]
        if perf<=hp['infancy_target_perf']:
            color = 'green'
            save_folder = save_root_folder+str(trial_num)+'infan/'+n_type+'/'
        elif perf<=hp['young_target_perf']:
            color = 'blue'
            save_folder = save_root_folder+str(trial_num)+'young/'+n_type+'/'
        else:
            color = 'red'
            save_folder = save_root_folder+str(trial_num)+'adult/'+n_type+'/'

        tools.mkdir_p(save_folder)

        posi_list = [1,2,5,8,7,6,3,0]
        period_mean = [0 for _ in range(9)]
        for sample_n in n_list:
            fig,ax = plt.subplots(3,3,figsize=(10,10))

            max_ = 0
            min_ = 0
            psth = dict()
            time = np.arange(len(H[:,0,0]))*hp['dt']/1000

            for loc in task_info[rule]['in_loc_set']:

                psth[loc] = H[:,task_info[rule]['in_loc'] == loc,sample_n[0]].mean(axis=1)
                period_mean[loc] = \
                    H[task_info[rule]['epoch_info'][epoch][0]:task_info[rule]['epoch_info'][epoch][1],task_info[rule]['in_loc'] == loc,sample_n[0]].mean()

                max_temp = np.max(psth[loc])
                min_temp = np.min(psth[loc])

                if max_temp>max_:
                    max_ = max_temp
                if min_temp<min_:
                    min_ = min_temp

            period_mean[-1] = period_mean[0]
            period_mean = np.array(period_mean)
            period_mean /= period_mean.max()

            for loc in task_info[rule]['in_loc_set']:
                ax[posi_list[loc]//3][posi_list[loc]%3].set_ylim(min_-0.1*abs(max_),max_+0.1*abs(max_))
                ax[posi_list[loc]//3][posi_list[loc]%3].plot(time,psth[loc],color=color)
                ax[posi_list[loc]//3][posi_list[loc]%3].set_xticks(np.arange(0,np.max(time),1))
                if loc in [1,2,3]:
                    ax[posi_list[loc]//3][posi_list[loc]%3].yaxis.set_ticks_position('right')
                if loc in [7,0,1]:
                    ax[posi_list[loc]//3][posi_list[loc]%3].xaxis.set_ticks_position('top')

            axis = plt.subplot(3,3,5,projection='polar')
            axis.set_theta_zero_location('N')
            axis.set_theta_direction(-1)
            
            axis.set_yticks([])
            #axis.set_xticks([])

            theta1 = np.arange(0, 2 * np.pi + 0.00000001, np.pi / 4)
            axis.plot(theta1,period_mean,color=color)

            theta2 = np.arange(0,2*np.pi,2*np.pi/360)
            sel_dir_point = np.zeros(360)
            sel_dir_point[sample_n[2]*45] = 1
            axis.plot(theta2,sel_dir_point,color='black')

            title = 'Rule:'+rule+' Epoch:'+epoch+' Neuron:'+str(sample_n[0])+' SelectDir:'+str(sample_n[2])+\
                ' Perf:'+str(perf)[:4]
            plt.suptitle(title)

            plt.savefig(save_folder+str(sample_n[0])+'.png',transparent=False)
            plt.savefig(save_folder+str(sample_n[0])+'.eps',transparent=False)
            plt.savefig(save_folder+str(sample_n[0])+'.pdf',transparent=False)

            plt.close()

            #polar only
            fig_p,ax_p = plt.subplots(figsize=(10,11),subplot_kw=dict(projection='polar'))
            ax_p.set_theta_zero_location('N')
            ax_p.set_theta_direction(-1)
            
            ax_p.set_yticks([])
            #ax_p.set_xticks([])

            ax_p.plot(theta1,period_mean,color=color)
            ax_p.plot(theta2,sel_dir_point,color='black')

            plt.suptitle(title)
            plt.savefig(save_folder+str(sample_n[0])+'_polar.png',transparent=False)
            plt.savefig(save_folder+str(sample_n[0])+'_polar.eps',transparent=False)
            plt.savefig(save_folder+str(sample_n[0])+'_polar.pdf',transparent=False)

            plt.close()