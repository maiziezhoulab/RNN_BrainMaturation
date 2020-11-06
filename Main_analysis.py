from utils import tools
import os
import numpy as np

from analysis.PSTH_print_basic_info import print_basic_info
from analysis.PSTH_compute_H import compute_H, gen_task_info
from analysis.PSTH_gen_neuron_info import generate_neuron_info

from analysis.PSTH_tunning_analysis import tunning_analysis
from analysis.PSTH_gen_PSTH_log import gen_PSTH_log
from analysis.PSTH_plot_epoch_mean_growth import plot_epoch_mean_growth
from analysis.PSTH_seldir_analysis import seldir_analysis
from analysis.PSTH_plot_PSTH import plot_PSTH

from analysis.PSTH_neuron_period_activity_analysis import neuron_period_activity_analysis
from analysis.PSTH_odrd_distractor_analysis import odrd_distractor_analysis
from analysis.PSTH_saccade_distribut_analysis import saccade_distribut_analysis

from analysis.PSTH_sample_neuron_by_trial import sample_neuron_by_trial
from analysis.PSTH_neuron_dPCA import neuron_dPCA, real_neuron_dPCA
from analysis.PSTH_Decoder_analysis import Decoder_analysis, realneuron_Decoder_analysis


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model_dir = 'data/6tasks_paper'
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)
    
    recompute = False

    trial_list_odr_odrd = range(520960,log['trials'][-1]+1,1280)
    trial_list_antisacc = range(0,102400+1,1280)
    trial_list_all = range(log['trials'][0],log['trials'][-1]+1,1280)

    print_basic_info(hp,log,model_dir,smooth_growth=True,smooth_window=5)

    print("compute H")
    gen_task_info(hp,log,model_dir,['odr','odrd'],) #only generate task_info.pkl
    #compute_H(hp,log,model_dir,  rules=['odr','odrd'], trial_list=trial_list_odr_odrd, recompute=recompute,)
    #compute_H(hp,log,model_dir,  rules=['overlap','zero_gap','gap'], trial_list=trial_list_antisacc, recompute=recompute,)

    print("Generate Info")
    generate_neuron_info(hp,model_dir,epochs=['stim1','delay1'],trial_list=trial_list_odr_odrd,rules=['odr','odrd'],recompute = False,)
    generate_neuron_info(hp,model_dir,epochs=['delay2'],trial_list=trial_list_odr_odrd,rules=['odrd'],recompute = False,)
    generate_neuron_info(hp,model_dir,epochs=['stim1'],trial_list=trial_list_antisacc,rules=['overlap','zero_gap','gap'],recompute = False,)

    print('\n\nPlot')
##########################################################################
##########################################################################
    tunning_analysis(hp,log,model_dir,'odr','stim1',trial_list_odr_odrd)
    tunning_analysis(hp,log,model_dir,'odr','delay1',trial_list_odr_odrd)
    tunning_analysis(hp,log,model_dir,'odrd','stim1',trial_list_odr_odrd)
    tunning_analysis(hp,log,model_dir,'odrd','delay2',trial_list_odr_odrd)
    tunning_analysis(hp,log,model_dir,'gap','stim1',trial_list_antisacc)
    tunning_analysis(hp,log,model_dir,'overlap','stim1',trial_list_antisacc)
    tunning_analysis(hp,log,model_dir,'zero_gap','stim1',trial_list_antisacc)

#######################################################
    plot_PSTH(hp, log, model_dir, 'odr', 'stim1', trial_list_odr_odrd, plot_oppo_dir = False)
    plot_PSTH(hp, log, model_dir, 'odr', 'delay1', trial_list_odr_odrd, plot_oppo_dir = False)
    plot_PSTH(hp, log, model_dir, 'odrd', 'stim1', trial_list_odr_odrd, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay1', trial_list_odr_odrd, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay2', trial_list_odr_odrd, plot_oppo_dir = True)

    plot_PSTH(hp, log, model_dir, 'gap', 'stim1', trial_list_antisacc, plot_oppo_dir = False)
    plot_PSTH(hp, log, model_dir, 'zero_gap', 'stim1', trial_list_antisacc, plot_oppo_dir = False)
    plot_PSTH(hp, log, model_dir, 'overlap', 'stim1', trial_list_antisacc, plot_oppo_dir = False)

##########################################################################
##########################################################################
    PSTH_log = gen_PSTH_log(hp,trial_list_all,model_dir,rule = 'odr',seltive_epoch = 'stim1')
    plot_epoch_mean_growth(hp,log, trial_list_all, model_dir, rule = 'odr', seltive_epoch = 'stim1', analy_epoch = 'stim1',PSTH_log = PSTH_log)
    plot_epoch_mean_growth(hp,log, trial_list_all, model_dir, rule = 'odr', seltive_epoch = 'stim1', analy_epoch = 'delay1',PSTH_log = PSTH_log)

####################################################### 
    plot_epoch_mean_growth(hp,log, trial_list_all, model_dir, rule = 'odr', seltive_epoch = 'delay1', analy_epoch = 'delay1',PSTH_log = PSTH_log)

#######################################################
    PSTH_log = gen_PSTH_log(hp,trial_list_all,model_dir,rule = 'odrd',seltive_epoch = 'stim1')
    plot_epoch_mean_growth(hp,log, trial_list_all, model_dir, rule = 'odrd', seltive_epoch = 'stim1', analy_epoch = 'stim1',PSTH_log = PSTH_log)
    plot_epoch_mean_growth(hp,log, trial_list_all, model_dir, rule = 'odrd', seltive_epoch = 'stim1', analy_epoch = 'delay2',PSTH_log = PSTH_log)


##########################################################################
##########################################################################
    #seldir_analysis(model_dir,'odr','stim1','odrd','stim1',trial_list_all,('exh_neurons','mix_neurons'))
    #seldir_analysis(model_dir,'odr','delay1','odrd','delay1',trial_list_all,('exh_neurons','mix_neurons'))
    #seldir_analysis(model_dir,'odr','delay1','odrd','delay2',trial_list_all,('exh_neurons','mix_neurons'))
    #seldir_analysis(model_dir,'odrd','delay1','odrd','delay2',trial_list_all,('exh_neurons','mix_neurons'))
    #seldir_analysis(model_dir,'odrd','stim1','odrd','delay2',trial_list_all,('exh_neurons','mix_neurons'))
    #seldir_analysis(model_dir,'overlap','stim1','odr','stim1',trial_list_all,('exh_neurons','mix_neurons'))

##########################################################################
##########################################################################
    #neuron_period_activity_analysis(hp,log, trial_list_odr_odrd, model_dir, rule = 'odr', seltive_epoch = 'delay1', analy_epoch = 'delay1')
    #neuron_period_activity_analysis(hp,log, trial_list_odr_odrd, model_dir, rule = 'odrd', seltive_epoch = 'delay2', analy_epoch = 'delay2')
    #neuron_period_activity_analysis(hp,log, trial_list_antisacc, model_dir, rule = 'overlap', seltive_epoch = 'stim1', analy_epoch = (1000,1250))
    #neuron_period_activity_analysis(hp,log, trial_list_antisacc, model_dir, rule = 'zero_gap', seltive_epoch = 'stim1', analy_epoch = (1000,1250))
    #neuron_period_activity_analysis(hp,log, trial_list_antisacc, model_dir, rule = 'gap', seltive_epoch = 'stim1', analy_epoch = (1100,1350))

##########################################################################
##########################################################################
    #odrd_distractor_analysis(hp,log,model_dir,trial_list=trial_list_odr_odrd)
    #saccade_distribut_analysis(hp,log,'overlap',model_dir,trial_list=trial_list_antisacc,)
    #saccade_distribut_analysis(hp,log,'gap',model_dir,trial_list=trial_list_antisacc,)
    #saccade_distribut_analysis(hp,log,'zero_gap',model_dir,trial_list=trial_list_antisacc,)

##########################################################################
##########################################################################
    sample_neuron_by_trial(hp,log,model_dir,'odr','stim1',[616960,628480],'mix_neurons')
    sample_neuron_by_trial(hp,log,model_dir,'odr','delay1',[616960,628480],'mix_neurons')
    sample_neuron_by_trial(hp,log,model_dir,'odrd','stim1',[616960,628480],'mix_neurons')
    sample_neuron_by_trial(hp,log,model_dir,'odrd','delay2',[616960,628480],'mix_neurons')

    #sample_neuron_by_trial(hp,log,model_dir,'odr','stim1',[616960,628480],'exh_neurons')
    #sample_neuron_by_trial(hp,log,model_dir,'odr','delay1',[616960,628480],'exh_neurons')
    #sample_neuron_by_trial(hp,log,model_dir,'odrd','stim1',[616960,628480],'exh_neurons')
    #sample_neuron_by_trial(hp,log,model_dir,'odrd','delay2',[616960,628480],'exh_neurons')

##########################################################################
##########################################################################
    Decoder_analysis(hp,model_dir,trial_num = log['trials'][-1],window_size=10,test_num=2)
    Decoder_analysis(hp,model_dir,trial_num = log['trials'][420],window_size=10,test_num=2)
    Decoder_analysis(hp,model_dir,trial_num = log['trials'][452],window_size=10,test_num=2)

    #Decoder_analysis(hp,model_dir,trial_num = log['trials'][-1],window_size=10,test_num=2,rule='odrd',)
    #Decoder_analysis(hp,model_dir,trial_num = log['trials'][468],window_size=10,test_num=2,rule='odrd',)
    #Decoder_analysis(hp,model_dir,trial_num = log['trials'][441],window_size=10,test_num=2,rule='odrd',)

##########################################################################
##########################################################################
    neuron_dPCA(hp,log,model_dir,trial_num = log['trials'][-1],rule='odr',task_mode='test',invert_tcomp_y=True,)
    neuron_dPCA(hp,log,model_dir,trial_num = log['trials'][452],rule='odr',task_mode='test',invert_tcomp_y=True,)
    neuron_dPCA(hp,log,model_dir,trial_num = log['trials'][420],rule='odr',task_mode='test',invert_tcomp_y=True,)

    #neuron_dPCA(hp,log,model_dir,trial_num = log['trials'][-1],rule='odrd',task_mode='test',appoint_loc_analysis=True, appoint_locs=[1,5],\
    #    invert_tcomp_y=True,invert_stcomp_y=True,tcomp_ylim=(-12,10),scomp_ylim=(-1,1),stcomp_ylim=(-15,15))#0.95898
    #neuron_dPCA(hp,log,model_dir,trial_num = log['trials'][468],rule='odrd',task_mode='test',appoint_loc_analysis=True, appoint_locs=[1,5],\
    #    invert_tcomp_y=True,invert_stcomp_y=True,tcomp_ylim=(-12,10),scomp_ylim=(-1,1),stcomp_ylim=(-15,15))#0.54102
    #neuron_dPCA(hp,log,model_dir,trial_num = log['trials'][441],rule='odrd',task_mode='test',appoint_loc_analysis=True, appoint_locs=[1,5],\
    #    invert_tcomp_y=True,invert_stcomp_y=True,tcomp_ylim=(-12,10),scomp_ylim=(-1,1),stcomp_ylim=(-15,15))#0.35352

    #neuron_dPCA(hp,log,model_dir,trial_num = log['trials'][80],rule='overlap',task_mode='test',appoint_loc_analysis=True, appoint_locs=[0,4],\
    #    invert_scomp_y=True,invert_stcomp_y=True,tcomp_ylim=(-4,8),scomp_ylim=(-1.5,1.5),stcomp_ylim=(-3,3))#1.0
    #neuron_dPCA(hp,log,model_dir,trial_num = log['trials'][42],rule='overlap',task_mode='test',appoint_loc_analysis=True, appoint_locs=[1,5],\
    #    invert_scomp_y=True,invert_stcomp_y=True,tcomp_ylim=(-4,8),scomp_ylim=(-1.5,1.5),stcomp_ylim=(-3,3))#0.57422
    #neuron_dPCA(hp,log,model_dir,trial_num = log['trials'][29],rule='overlap',task_mode='test',appoint_loc_analysis=True, appoint_locs=[1,5],\
    #    invert_scomp_y=True,invert_stcomp_y=True,tcomp_ylim=(-4,8),scomp_ylim=(-1.5,1.5),stcomp_ylim=(-3,3))#0.32813

