# basic packages #
import numpy as np

import sys
sys.path.append('.')
from utils import tools

# plot #
import matplotlib.pyplot as plt

# DL & task #
import tensorflow as tf
from task_and_network import task
from task_and_network.task import generate_trials
from task_and_network.network import Model,popvec

def get_perf(y_hat, y_loc):
    """Get performance.

    Args:
      y_hat: Actual output. Numpy array (Time, Batch, Unit)
      y_loc: Target output location (-1 for fixation).
        Numpy array (Time, Batch)

    Returns:
      perf: Numpy array (Batch,)
    """
    if len(y_hat.shape) != 3:
        raise ValueError('y_hat must have shape (Time, Batch, Unit)')
    # Only look at last time points
    y_loc = y_loc[-1]
    y_hat = y_hat[-1]

    # Fixation and location of y_hat
    y_hat_fix = y_hat[..., 0]
    y_hat_loc = popvec(y_hat[..., 1:])

    # Fixating? Correctly saccading?
    fixating = y_hat_fix > 0.5

    original_dist = y_loc - y_hat_loc
    dist = np.minimum(abs(original_dist), 2*np.pi-abs(original_dist))*180/np.pi
    corr_loc = dist < 36

    # Should fixate?
    should_fix = y_loc < 0

    # performance
    perf = should_fix * fixating + (1-should_fix) * corr_loc * (1-fixating)
    return perf, list(dist)

def get_abs_dist(loc_a,loc_b,total_len):
    return np.minimum(abs(loc_a-loc_b),total_len-abs(loc_a-loc_b))

def odrd_distractor_analysis(hp,log,model_dir,trial_list,):

    early_record = list()
    mid_record = list()
    mature_record = list()

    early_saccade_dir = dict()
    mid_saccade_dir = dict()
    mature_saccade_dir = dict()
    for d in range(hp['n_eachring']//2+1):
        early_saccade_dir[d] = list()
        mid_saccade_dir[d] = list()
        mature_saccade_dir[d] = list()

    is_dict = False
    is_list = False
    if isinstance(trial_list, dict):
        temp_list = list()
        is_dict = True
        for value in trial_list['odrd'].values():
            temp_list += value
        temp_list = sorted(set(temp_list))
    elif isinstance(trial_list, list):
        temp_list = trial_list
        is_list = True

    for trial_num in temp_list:

        saccade_dir_temp= dict()
        for d in range(hp['n_eachring']//2+1):
            saccade_dir_temp[d] = list()

        temp_pref_list = np.zeros(hp['n_eachring']//2+1)

        model = Model(model_dir+'/'+str(trial_num)+'/', hp=hp)
        with tf.Session() as sess:
            model.restore()

            for stim1 in range(0,hp['n_eachring']):
                for distrac in range(0,hp['n_eachring']):
                    task_mode = 'test-'+str(stim1)+'-'+str(distrac)
                    trial = generate_trials('odrd', hp, task_mode)
                    feed_dict = tools.gen_feed_dict(model, trial, hp)
                    y_hat = sess.run(model.y_hat,feed_dict=feed_dict)
                    temp_pref, dist = get_perf(y_hat, trial.y_loc)
                    temp_pref = np.mean(temp_pref)
                    temp_pref_list[get_abs_dist(stim1,distrac,hp['n_eachring'])] += temp_pref/hp['n_eachring']

                    saccade_dir_temp[get_abs_dist(stim1,distrac,hp['n_eachring'])] += dist

        for i in range(1,len(temp_pref_list)-1):
            temp_pref_list[i] /= 2

        if len(temp_pref_list)%2 == 0:
            temp_pref_list[-1] /= 2

        matur = log['perf_odrd'][trial_num//log['trials'][1]]
        if (is_list and matur > hp['mid_target_perf']) or (is_dict and trial_num in trial_list['odrd']['mature']):
            mature_record.append(temp_pref_list)
            for key,value in saccade_dir_temp.items():
                mature_saccade_dir[key] += value
        elif (is_list and matur > hp['early_target_perf']) or (is_dict and trial_num in trial_list['odrd']['mid']):
            mid_record.append(temp_pref_list)
            for key,value in saccade_dir_temp.items():
                mid_saccade_dir[key] += value
        elif is_list or (is_dict and trial_num in trial_list['odrd']['early']):
            early_record.append(temp_pref_list)
            for key,value in saccade_dir_temp.items():
                early_saccade_dir[key] += value

    early_trial_count = len(early_record)
    mid_trial_count = len(mid_record)
    mature_trial_count = len(mature_record)

    early_record = np.array(early_record).mean(axis=0)
    mid_record = np.array(mid_record).mean(axis=0)
    mature_record = np.array(mature_record).mean(axis=0)

    abs_dist_list = np.arange(hp['n_eachring']//2+1)

    fig,ax = plt.subplots(figsize=(10,6))
    try:
        ax.plot(abs_dist_list,early_record,color='green',label='early trial_num:'+str(early_trial_count))
    except:
        pass
    try:
        ax.plot(abs_dist_list,mid_record,color='blue',label='mid trial_num:'+str(mid_trial_count))
    except:
        pass
    try:
        ax.plot(abs_dist_list,mature_record,color='red',label='mature trial_num:'+str(mature_trial_count))
    except:
        pass
    ax.set_xticks(abs_dist_list)
    ax.set_ylabel("perf")
    ax.set_xlabel("distance between distractor and stim1 ($\\times$%.1f$\degree$)"%(360/hp['n_eachring']))
    ax.legend()

    save_folder = 'figure/figure_'+model_dir.rstrip('/').split('/')[-1]+'/odrd/'
    tools.mkdir_p(save_folder)
    save_pic = save_folder+'odrd_distractor_analysis_by_growth'
    plt.savefig(save_pic+'.png',transparent=False)
    plt.savefig(save_pic+'.eps',transparent=False)
    plt.savefig(save_pic+'.pdf',transparent=False)

    plt.close(fig)

    for d in range(hp['n_eachring']//2+1):
        fig,axes = plt.subplots(1,3,figsize=(18,6))
        axes[0].hist(early_saccade_dir[d],bins=30,range=(0,180), histtype="stepfilled",alpha=0.6, color="green")
        axes[0].set_title("early")
        axes[1].hist(mid_saccade_dir[d],bins=30,range=(0,180), histtype="stepfilled",alpha=0.6, color="blue")
        axes[1].set_title("mid")
        axes[2].hist(mature_saccade_dir[d],bins=30,range=(0,180), histtype="stepfilled",alpha=0.6, color="red")
        axes[2].set_title("mature")
        for i in range(3):
            axes[i].set_xlabel("distance to stim1($\degree$)")
        fig.suptitle("distractor distance: "+str(d))

        save_pic = save_folder+'saccade_distribut_analysis_by_growth_dis_'+str(d)
        plt.savefig(save_pic+'.png',transparent=False)
        plt.savefig(save_pic+'.eps',transparent=False)
        plt.savefig(save_pic+'.pdf',transparent=False)

        plt.close(fig)

