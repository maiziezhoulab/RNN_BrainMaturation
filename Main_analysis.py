from utils import tools
import os

from analysis.PSTH_print_basic_info import print_basic_info
from analysis.PSTH_compute_H import compute_H, gen_task_info
from analysis.PSTH_gen_neuron_info import generate_neuron_info

from analysis.PSTH_tunning_analysis import tunning_analysis
from analysis.PSTH_gen_PSTH_log import gen_PSTH_log
from analysis.PSTH_plot_period_ttest import plot_period_ttest_heatmap
from analysis.PSTH_plot_epoch_mean_growth import plot_epoch_mean_growth
from analysis.PSTH_seldir_analysis import seldir_analysis
from analysis.PSTH_plot_PSTH import plot_PSTH

from analysis.PSTH_neuron_period_activity_analysis import neuron_period_activity_analysis
from analysis.PSTH_odrd_distractor_analysis import odrd_distractor_analysis
from analysis.PSTH_saccade_distribut_analysis import saccade_distribut_analysis


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--modeldir', type=str, default='data/6tasks')
    parser.add_argument('--rules', type=str, default='None')
    parser.add_argument('--epochs', type=str, default='None')
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

    if args.epochs == 'None':
        epochs = None
    else:
        epochs = args.epochs.split(',')

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
    #trial_list = range(start,end+1,step)
    trial_list = range(0,62720,1280)

    print_basic_info(hp,log,model_dir,smooth_growth=True,smooth_window=5)
    #compute_H(hp,log,model_dir,  rules=rules, trial_list=trial_list, recompute=recompute,)
    print("compute H")
    #gen_task_info(hp,log,model_dir,['odr','odrd'],trial_list) #only generate task_info.pkl
    #compute_H(hp,log,model_dir,  rules=['odr','odrd'], trial_list=trial_list, recompute=recompute,)
    #compute_H(hp,log,model_dir,  rules=['odrd',], trial_list=range(520960,628480+1,1280), recompute=recompute,)
    #compute_H(hp,log,model_dir,  rules=['overlap','zero_gap','gap'], trial_list=trial_list, recompute=recompute,)

    print("Generate Info")
    #generate_neuron_info(hp,model_dir,epochs=['stim1','delay1'],trial_list=trial_list,rules=['odr','odrd'],)
    #generate_neuron_info(hp,model_dir,epochs=['delay2'],trial_list=trial_list,rules=['odrd'],)
    #generate_neuron_info(hp,model_dir,epochs=['stim1'],trial_list=trial_list,rules=['overlap','zero_gap','gap'],)

    print('\n\nPlot')
##########################################################################
##########################################################################
    #tunning_analysis(hp,log,model_dir,'odr','stim1',range(520960,628480+1,1280))
    #tunning_analysis(hp,log,model_dir,'odr','delay1',range(520960,628480+1,1280))
    #tunning_analysis(hp,log,model_dir,'odrd','stim1',range(520960,628480+1,1280))
    #tunning_analysis(hp,log,model_dir,'odrd','delay2',range(520960,628480+1,1280))
    #tunning_analysis(hp,log,model_dir,'gap','stim1',range(0,102400+1,1280))
    #tunning_analysis(hp,log,model_dir,'overlap','stim1',range(0,102400+1,1280))
    #tunning_analysis(hp,log,model_dir,'zero_gap','stim1',range(0,102400+1,1280))
#######################################################
    #plot_PSTH(hp, log, model_dir, 'odr', 'stim1', range(520960,628480+1,1280), plot_oppo_dir = False)
    #plot_PSTH(hp, log, model_dir, 'odr', 'delay1', range(520960,628480+1,1280), plot_oppo_dir = False)
    #plot_PSTH(hp, log, model_dir, 'odrd', 'stim1', range(520960,628480+1,1280), plot_oppo_dir = True)
    #plot_PSTH(hp, log, model_dir, 'odrd', 'delay1', range(520960,628480+1,1280), plot_oppo_dir = True)
    #plot_PSTH(hp, log, model_dir, 'odrd', 'delay2', range(520960,628480+1,1280), plot_oppo_dir = True)
##########################################################################
##########################################################################
    #PSTH_log = gen_PSTH_log(hp,trial_list,model_dir,rule = 'odr',seltive_epoch = 'stim1')
    #plot_period_ttest_heatmap(hp,log, trial_list, model_dir, rule = 'odr', seltive_epoch = 'stim1', analy_epoch = 'stim1',PSTH_log = PSTH_log)
    #plot_period_ttest_heatmap(hp,log, trial_list, model_dir, rule = 'odr', seltive_epoch = 'stim1', analy_epoch = 'delay1',PSTH_log = PSTH_log)

    #plot_epoch_mean_growth(hp,log, trial_list, model_dir, rule = 'odr', seltive_epoch = 'stim1', analy_epoch = 'stim1',PSTH_log = PSTH_log)
    #plot_epoch_mean_growth(hp,log, trial_list, model_dir, rule = 'odr', seltive_epoch = 'stim1', analy_epoch = 'delay1',PSTH_log = PSTH_log)
#######################################################
    #PSTH_log = gen_PSTH_log(hp,trial_list,model_dir,rule = 'odr',seltive_epoch = 'delay1')
    #plot_period_ttest_heatmap(hp,log, trial_list, model_dir, rule = 'odr', seltive_epoch = 'delay1', analy_epoch = 'delay1',PSTH_log = PSTH_log)
    
    #plot_epoch_mean_growth(hp,log, trial_list, model_dir, rule = 'odr', seltive_epoch = 'delay1', analy_epoch = 'delay1',PSTH_log = PSTH_log)

#######################################################
    #PSTH_log = gen_PSTH_log(hp,trial_list,model_dir,rule = 'odrd',seltive_epoch = 'stim1')
    #plot_period_ttest_heatmap(hp,log, trial_list, model_dir, rule = 'odrd', seltive_epoch = 'stim1', analy_epoch = 'stim1',PSTH_log = PSTH_log)  
    #plot_period_ttest_heatmap(hp,log, trial_list, model_dir, rule = 'odrd', seltive_epoch = 'stim1', analy_epoch = 'delay2',PSTH_log = PSTH_log)

    #plot_epoch_mean_growth(hp,log, trial_list, model_dir, rule = 'odrd', seltive_epoch = 'stim1', analy_epoch = 'stim1',PSTH_log = PSTH_log)
    #plot_epoch_mean_growth(hp,log, trial_list, model_dir, rule = 'odrd', seltive_epoch = 'stim1', analy_epoch = 'delay2',PSTH_log = PSTH_log)

#######################################################
    #plot_period_ttest_heatmap(hp,log, trial_list, model_dir, rule = 'overlap', seltive_epoch = 'stim1', analy_epoch = 'stim1')
    #plot_period_ttest_heatmap(hp,log, trial_list, model_dir, rule = 'zero_gap', seltive_epoch = 'stim1', analy_epoch = 'stim1')
    #plot_period_ttest_heatmap(hp,log, trial_list, model_dir, rule = 'gap', seltive_epoch = 'stim1', analy_epoch = 'stim1')


##########################################################################
##########################################################################
    #seldir_analysis(model_dir,'odr','stim1','odrd','stim1',trial_list,('exh_neurons','mix_neurons'))
    #seldir_analysis(model_dir,'odr','delay1','odrd','delay1',trial_list,('exh_neurons','mix_neurons'))
    #seldir_analysis(model_dir,'odr','delay1','odrd','delay2',trial_list,('exh_neurons','mix_neurons'))
    #seldir_analysis(model_dir,'odrd','delay1','odrd','delay2',trial_list,('exh_neurons','mix_neurons'))
    #seldir_analysis(model_dir,'odrd','stim1','odrd','delay2',trial_list,('exh_neurons','mix_neurons'))
    #seldir_analysis(model_dir,'overlap','stim1','odr','stim1',trial_list,('exh_neurons','mix_neurons'))

##########################################################################
##########################################################################
    #neuron_period_activity_analysis(hp,log, range(520960,628480+1,1280), model_dir, rule = 'odr', seltive_epoch = 'delay1', analy_epoch = 'delay1')
    #neuron_period_activity_analysis(hp,log, range(520960,628480+1,1280), model_dir, rule = 'odrd', seltive_epoch = 'delay2', analy_epoch = 'delay2')
    #neuron_period_activity_analysis(hp,log, range(0,102400+1,1280), model_dir, rule = 'overlap', seltive_epoch = 'stim1', analy_epoch = (1000,1250))
    #neuron_period_activity_analysis(hp,log, range(0,102400+1,1280), model_dir, rule = 'zero_gap', seltive_epoch = 'stim1', analy_epoch = (1000,1250))
    #neuron_period_activity_analysis(hp,log, range(0,102400+1,1280), model_dir, rule = 'gap', seltive_epoch = 'stim1', analy_epoch = (1100,1350))

##########################################################################
##########################################################################
    #odrd_distractor_analysis(hp,log,model_dir,trial_list=range(520960,628480+1,1280))
    #saccade_distribut_analysis(hp,log,'overlap',model_dir,trial_list=range(0,102400+1,1280),)
    #saccade_distribut_analysis(hp,log,'gap',model_dir,trial_list=range(0,102400+1,1280),)
    #saccade_distribut_analysis(hp,log,'zero_gap',model_dir,trial_list=range(0,102400+1,1280),)

##########################################################################
##########################################################################
    #sample_neuron_by_trial(hp,log,model_dir,'odr','stim1',[616960,628480],'mix_neurons')
    #sample_neuron_by_trial(hp,log,model_dir,'odr','delay1',[616960,628480],'mix_neurons')
    #sample_neuron_by_trial(hp,log,model_dir,'odrd','stim1',[616960,628480],'mix_neurons')
    #sample_neuron_by_trial(hp,log,model_dir,'odrd','delay2',[616960,628480],'mix_neurons')

    #sample_neuron_by_trial(hp,log,model_dir,'odr','stim1',[616960,628480],'exh_neurons')
    #sample_neuron_by_trial(hp,log,model_dir,'odr','delay1',[616960,628480],'exh_neurons')
    #sample_neuron_by_trial(hp,log,model_dir,'odrd','stim1',[616960,628480],'exh_neurons')
    #sample_neuron_by_trial(hp,log,model_dir,'odrd','delay2',[616960,628480],'exh_neurons')