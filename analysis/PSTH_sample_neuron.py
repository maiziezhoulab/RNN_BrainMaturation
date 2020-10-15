# basic packages #
import os
import numpy as np
import pickle

import sys
sys.path.append('.')
from utils import tools

#plot
import matplotlib.pyplot as plt

def sample_neuron(hp,log,model_dir,rule,epoch,trial_list,n_type):

    with open(model_dir+'/task_info.pkl','rb') as tinf:
        task_info = pickle.load(tinf)

    with open(model_dir+'/H_'+rule+'.pkl','rb') as hf:
        H = pickle.load(hf)

    #if len(trial_list) == 1:
        #read_name = model_dir+'/neuron_info_'+rule+'_'+epoch+'_'+str(trial_list[0])+'.pkl'
    #else:
        #read_name = model_dir+'/neuron_info_'+rule+'_'+epoch+'_'+\
            #str(trial_list[0])+'_'+str(trial_list[-1])+'_step'+str(trial_list[1]-trial_list[0])+'.pkl'

    read_name = model_dir+'/neuron_info_'+rule+'_'+epoch+'.pkl'

    with open(read_name,'rb') as inf:
        neuron_info = pickle.load(inf)

    save_folder = 'figure/figure_'+model_dir.rstrip('/').split('/')[-1]+'/'+rule+'/'+epoch+'/sample_neuron/'+n_type+'/'
    tools.mkdir_p(save_folder)

    for trial_num in trial_list:
        n_list = neuron_info[trial_num][n_type]
        if not n_list:
            continue
        #sample_n:(neuron,p,sel_dir)
        if n_list[0][1] is None:
            sample_n = n_list[np.random.randint(0,len(n_list))]
        else:
            sample_n = sorted(n_list,key=lambda x:x[1])[0]

        perf = log['perf_'+rule][trial_num//log['trials'][1]]
        if perf<=hp['early_target_perf']:
            color = 'green'
        elif perf<=hp['mid_target_perf']:
            color = 'blue'
        else:
            color = 'red'

        fig = plt.figure()
        posi_list = [2,3,6,9,8,7,4,1]

        max_ = 0
        min_ = 0
        psth = dict()
        time = np.arange(len(H[trial_num][:,0,0]))*hp['dt']/1000

        for loc in task_info[rule]['in_loc_set']:

            psth[loc] = H[trial_num][:,task_info[rule]['in_loc'] == loc,sample_n[0]].mean(axis=1)
            #psth[loc] = H[trial_num][:,task_info[rule]['in_loc'] == loc,:]
            #psth[loc] = psth[loc][:,:,[x[0] for x in n_list]].mean(axis = 1)

            max_temp = np.max(psth[loc])
            min_temp = np.min(psth[loc])

            if max_temp>max_:
                max_ = max_temp
            if min_temp<min_:
                min_ = min_temp

        for loc in task_info[rule]['in_loc_set']:
            plt.subplot(3,3,posi_list[loc])
            plt.ylim(min_-0.5,max_+0.5)
            plt.plot(time,psth[loc],color=color)
            plt.xticks(np.arange(0,np.max(time),1))

        axis = plt.subplot(3,3,5,projection='polar')
        axis.set_theta_zero_location('N')
        axis.set_theta_direction(-1)
        axis.set_xticks([])
        axis.set_yticks([])
        x = np.arange(0,2*np.pi,2*np.pi/360)
        y = np.zeros(360)
        y[sample_n[2]*45] = 1
        axis.plot(x,y,color=color)

        title = 'Rule:'+rule+' Epoch:'+epoch+' Neuron:'+str(sample_n[0])+' SelectDir:'+str(sample_n[2])+\
            ' Perf:'+str(perf)[:4]
        plt.title(title)

        plt.savefig(save_folder+str(trial_num)+'.png',transparent=False)
        plt.savefig(save_folder+str(trial_num)+'.eps',transparent=False)
        plt.savefig(save_folder+str(trial_num)+'.pdf',transparent=False)

        plt.close()
        

if __name__ == "__main__":
    model_dir = 'data/6tasks'
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)

    #for 6tasks folder for odr/odrd stage 
    start = 520960
    end = 628480
    step = 1280*1#3#21#12#7#12

    trial_list = range(start,end+1,step)

    #sample_neuron(hp,log,model_dir,'odr','stim1',trial_list,'mix_neurons')
    #sample_neuron(hp,log,model_dir,'odr','delay1',trial_list,'mix_neurons')
    #sample_neuron(hp,log,model_dir,'odrd','stim1',trial_list,'mix_neurons')
    #sample_neuron(hp,log,model_dir,'odrd','delay2',trial_list,'mix_neurons')

    #sample_neuron(hp,log,model_dir,'odr','stim1',trial_list,'exh_neurons')
    #sample_neuron(hp,log,model_dir,'odr','delay1',trial_list,'exh_neurons')
    #sample_neuron(hp,log,model_dir,'odr','delay1',trial_list,'selective_neurons')
    sample_neuron(hp,log,model_dir,'odrd','delay2',trial_list,'exh_neurons')