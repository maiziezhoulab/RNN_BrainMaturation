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

def saccade_distribut_analysis(hp,log,rule,model_dir,trial_list,):

    early_saccade_dir = list()
    mid_saccade_dir = list()
    mature_saccade_dir = list()

    is_dict = False
    is_list = False
    if isinstance(trial_list, dict):
        temp_list = list()
        is_dict = True
        for value in trial_list[rule].values():
            temp_list += value
        temp_list = sorted(set(temp_list))
    elif isinstance(trial_list, list):
        temp_list = trial_list
        is_list = True

    for trial_num in temp_list:

        saccade_dir_temp= list()

        model = Model(model_dir+'/'+str(trial_num)+'/', hp=hp)
        with tf.Session() as sess:
            model.restore()

            for i in range(0,hp['n_eachring']):
                trial = generate_trials(rule, hp, 'test')
                feed_dict = tools.gen_feed_dict(model, trial, hp)
                y_hat = sess.run(model.y_hat,feed_dict=feed_dict)
                _, dist = get_perf(y_hat, trial.y_loc)

                saccade_dir_temp += dist

        matur = log['perf_'+rule][trial_num//log['trials'][1]]
        if (is_list and matur > hp['mid_target_perf']) or (is_dict and trial_num in trial_list[rule]['mature']):
            mature_saccade_dir += saccade_dir_temp
        elif (is_list and matur > hp['early_target_perf']) or (is_dict and trial_num in trial_list[rule]['mid']):
            mid_saccade_dir += saccade_dir_temp
        elif is_list or (is_dict and trial_num in trial_list[rule]['early']):
            early_saccade_dir += saccade_dir_temp


    fig,axes = plt.subplots(1,3,figsize=(18,6))
    axes[0].hist(early_saccade_dir,bins=30,range=(0,180), histtype="stepfilled",alpha=0.6, color="green")
    axes[0].set_title("early")
    axes[1].hist(mid_saccade_dir,bins=30,range=(0,180), histtype="stepfilled",alpha=0.6, color="blue")
    axes[1].set_title("mid")
    axes[2].hist(mature_saccade_dir,bins=30,range=(0,180), histtype="stepfilled",alpha=0.6, color="red")
    axes[2].set_title("mature")
    fig.suptitle("saccade distribut analysis")

    save_folder = 'figure/figure_'+model_dir.rstrip('/').split('/')[-1]+'/'+rule+'/'
    save_pic = save_folder+'saccade_distribut_analysis_by_growth'
    plt.savefig(save_pic+'.png',transparent=False)
    plt.savefig(save_pic+'.eps',transparent=False)
    plt.savefig(save_pic+'.pdf',transparent=False)

    plt.close(fig)