#import tools
import os

from analysis.PSTH_print_basic_info import print_basic_info
from analysis.PSTH_compute_H import compute_H
from analysis.PSTH_gen_neuron_info import generate_neuron_info

from analysis.PSTH_plot_PSTH import plot_PSTH

from analysis.PSTH_odrd_distractor_analysis import odrd_distractor_analysis
from analysis.PSTH_saccade_distribut_analysis import saccade_distribut_analysis

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    '''
#####################################################
    model_dir = 'data/set1_exp1'
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)
    print_basic_info(hp,log,model_dir,smooth_growth=True,smooth_window=5)
    trial_list = range(820480,log['trials'][-1]+1,1280)
    recompute = False
    compute_H(hp,log,model_dir, rules=['odr','odrd'], trial_list=trial_list, recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['stim1','delay1'],trial_list=trial_list,rules=['odr','odrd'],recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['delay2'],trial_list=trial_list,rules=['odrd'],)
    plot_PSTH(hp, log, model_dir, 'odr', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odr', 'delay1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay2', trial_list, plot_oppo_dir = True)
#####################################################
    model_dir = 'data/set1_exp2'
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)
    print_basic_info(hp,log,model_dir,smooth_growth=True,smooth_window=5)
    trial_list = range(200960,log['trials'][-1]+1,1280)
    recompute = False
    compute_H(hp,log,model_dir, rules=['odr','odrd'], trial_list=trial_list, recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['stim1','delay1'],trial_list=trial_list,rules=['odr','odrd'],recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['delay2'],trial_list=trial_list,rules=['odrd'],)
    plot_PSTH(hp, log, model_dir, 'odr', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odr', 'delay1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay2', trial_list, plot_oppo_dir = True)
#####################################################
    model_dir = 'data/set1_exp3'
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)
    print_basic_info(hp,log,model_dir,smooth_growth=True,smooth_window=5)
    trial_list = range(149760,log['trials'][-1]+1,1280)
    recompute = False
    compute_H(hp,log,model_dir, rules=['odr','odrd'], trial_list=trial_list, recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['stim1','delay1'],trial_list=trial_list,rules=['odr','odrd'],recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['delay2'],trial_list=trial_list,rules=['odrd'],)
    plot_PSTH(hp, log, model_dir, 'odr', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odr', 'delay1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay2', trial_list, plot_oppo_dir = True)
#####################################################
    model_dir = 'data/set1_exp4'
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)
    print_basic_info(hp,log,model_dir,smooth_growth=True,smooth_window=5)
    trial_list = range(1100800,log['trials'][-1]+1,1280)
    recompute = False
    compute_H(hp,log,model_dir, rules=['odr','odrd'], trial_list=trial_list, recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['stim1','delay1'],trial_list=trial_list,rules=['odr','odrd'],recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['delay2'],trial_list=trial_list,rules=['odrd'],)
    plot_PSTH(hp, log, model_dir, 'odr', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odr', 'delay1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay2', trial_list, plot_oppo_dir = True)
#####################################################
    model_dir = 'data/set2_exp1'
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)
    print_basic_info(hp,log,model_dir,smooth_growth=True,smooth_window=5)
    trial_list = range(0,log['trials'][-1]+1,1280)
    recompute = False
    compute_H(hp,log,model_dir, rules=['odr','odrd'], trial_list=trial_list, recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['stim1','delay1'],trial_list=trial_list,rules=['odr','odrd'],recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['delay2'],trial_list=trial_list,rules=['odrd'],)
    plot_PSTH(hp, log, model_dir, 'odr', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odr', 'delay1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay2', trial_list, plot_oppo_dir = True)
#####################################################
    model_dir = 'data/set2_exp2'
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)
    print_basic_info(hp,log,model_dir,smooth_growth=True,smooth_window=5)
    trial_list = range(1100800,log['trials'][-1]+1,1280)
    recompute = False
    compute_H(hp,log,model_dir, rules=['odr','odrd'], trial_list=trial_list, recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['stim1','delay1'],trial_list=trial_list,rules=['odr','odrd'],recompute=recompute,norm=False)
    generate_neuron_info(hp,model_dir,epochs=['delay2'],trial_list=trial_list,rules=['odrd'],norm=False)
    plot_PSTH(hp, log, model_dir, 'odr', 'stim1', trial_list, plot_oppo_dir = True,norm=False)
    plot_PSTH(hp, log, model_dir, 'odr', 'delay1', trial_list, plot_oppo_dir = True,norm=False)
    plot_PSTH(hp, log, model_dir, 'odrd', 'stim1', trial_list, plot_oppo_dir = True,norm=False)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay1', trial_list, plot_oppo_dir = True,norm=False)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay2', trial_list, plot_oppo_dir = True,norm=False)
#####################################################
    model_dir = 'data/set2_exp3'
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)
    print_basic_info(hp,log,model_dir,smooth_growth=True,smooth_window=5)
    trial_list = range(2900480,log['trials'][-1]+1,1280)
    recompute = False
    compute_H(hp,log,model_dir, rules=['odr','odrd'], trial_list=trial_list, recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['stim1','delay1'],trial_list=trial_list,rules=['odr','odrd'],recompute=recompute,norm=False)
    generate_neuron_info(hp,model_dir,epochs=['delay2'],trial_list=trial_list,rules=['odrd'],norm=False)
    plot_PSTH(hp, log, model_dir, 'odr', 'stim1', trial_list, plot_oppo_dir = True,norm=False)
    plot_PSTH(hp, log, model_dir, 'odr', 'delay1', trial_list, plot_oppo_dir = True,norm=False)
    plot_PSTH(hp, log, model_dir, 'odrd', 'stim1', trial_list, plot_oppo_dir = True,norm=False)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay1', trial_list, plot_oppo_dir = True,norm=False)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay2', trial_list, plot_oppo_dir = True,norm=False)
#####################################################
    model_dir = 'data/set3_exp1'
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)
    print_basic_info(hp,log,model_dir,smooth_growth=True,smooth_window=5)
    trial_list = range(250880,log['trials'][-1]+1,1280)
    recompute = False
    compute_H(hp,log,model_dir, rules=['odr','odrd'], trial_list=trial_list, recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['stim1','delay1'],trial_list=trial_list,rules=['odr','odrd'],recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['delay2'],trial_list=trial_list,rules=['odrd'],)
    plot_PSTH(hp, log, model_dir, 'odr', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odr', 'delay1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay2', trial_list, plot_oppo_dir = True)
#####################################################
    model_dir = 'data/set3_exp2'
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)
    print_basic_info(hp,log,model_dir,smooth_growth=True,smooth_window=5)
    trial_list = range(261120,log['trials'][-1]+1,1280)
    recompute = False
    compute_H(hp,log,model_dir, rules=['odr','odrd'], trial_list=trial_list, recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['stim1','delay1'],trial_list=trial_list,rules=['odr','odrd'],recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['delay2'],trial_list=trial_list,rules=['odrd'],)
    plot_PSTH(hp, log, model_dir, 'odr', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odr', 'delay1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay2', trial_list, plot_oppo_dir = True)
#####################################################
    model_dir = 'data/set4_exp1' #NOTE: odr faster than odrd!
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)
    print_basic_info(hp,log,model_dir,smooth_growth=True,smooth_window=5)
    trial_list = range(0,log['trials'][-1]+1,1280)
    recompute = False
    compute_H(hp,log,model_dir, rules=['odr','odrd'], trial_list=trial_list, recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['stim1','delay1'],trial_list=trial_list,rules=['odr','odrd'],recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['delay2'],trial_list=trial_list,rules=['odrd'],)
    plot_PSTH(hp, log, model_dir, 'odr', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odr', 'delay1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay2', trial_list, plot_oppo_dir = True)
#####################################################
    model_dir = 'data/set4_exp2' #TRANING FAILED#
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)
    print_basic_info(hp,log,model_dir,smooth_growth=True,smooth_window=5)
    #trial_list = 
    #recompute = False
    #compute_H(hp,log,model_dir, rules=['odr','odrd'], trial_list=trial_list, recompute=recompute,)
    #generate_neuron_info(hp,model_dir,epochs=['stim1','delay1'],trial_list=trial_list,rules=['odr','odrd'],recompute=recompute,)
    #generate_neuron_info(hp,model_dir,epochs=['delay2'],trial_list=trial_list,rules=['odrd'],)
    #plot_PSTH(hp, log, model_dir, 'odr', 'stim1', trial_list, plot_oppo_dir = True)
    #plot_PSTH(hp, log, model_dir, 'odr', 'delay1', trial_list, plot_oppo_dir = True)
    #plot_PSTH(hp, log, model_dir, 'odrd', 'stim1', trial_list, plot_oppo_dir = True)
    #plot_PSTH(hp, log, model_dir, 'odrd', 'delay1', trial_list, plot_oppo_dir = True)
    #plot_PSTH(hp, log, model_dir, 'odrd', 'delay2', trial_list, plot_oppo_dir = True)
#####################################################
    model_dir = 'data/set4_exp3'#TRINING FAILED#
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)
    print_basic_info(hp,log,model_dir,smooth_growth=True,smooth_window=5)
    #trial_list = 
    #recompute = False
    #compute_H(hp,log,model_dir, rules=['odr','odrd'], trial_list=trial_list, recompute=recompute,)
    #generate_neuron_info(hp,model_dir,epochs=['stim1','delay1'],trial_list=trial_list,rules=['odr','odrd'],recompute=recompute,)
    #generate_neuron_info(hp,model_dir,epochs=['delay2'],trial_list=trial_list,rules=['odrd'],)
    #plot_PSTH(hp, log, model_dir, 'odr', 'stim1', trial_list, plot_oppo_dir = True)
    #plot_PSTH(hp, log, model_dir, 'odr', 'delay1', trial_list, plot_oppo_dir = True)
    #plot_PSTH(hp, log, model_dir, 'odrd', 'stim1', trial_list, plot_oppo_dir = True)
    #plot_PSTH(hp, log, model_dir, 'odrd', 'delay1', trial_list, plot_oppo_dir = True)
    #plot_PSTH(hp, log, model_dir, 'odrd', 'delay2', trial_list, plot_oppo_dir = True)
#####################################################
    model_dir = 'data/set5_exp1'
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)
    print_basic_info(hp,log,model_dir,smooth_growth=True,smooth_window=5)
    trial_list = range(720640,log['trials'][-1]+1,1280)
    recompute = False
    compute_H(hp,log,model_dir, rules=['odr','odrd'], trial_list=trial_list, recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['stim1','delay1'],trial_list=trial_list,rules=['odr','odrd'],recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['delay2'],trial_list=trial_list,rules=['odrd'],)
    plot_PSTH(hp, log, model_dir, 'odr', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odr', 'delay1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay2', trial_list, plot_oppo_dir = True)
    '''
#####################################################
    '''
    model_dir = 'data/set6_exp1'
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)
    print_basic_info(hp,log,model_dir,smooth_growth=True,smooth_window=5)
    trial_list = range(720640,log['trials'][-1]+1,1280)
    recompute = False
    compute_H(hp,log,model_dir, rules=['odr3000','odrd'], trial_list=trial_list, recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['stim1','delay1'],trial_list=trial_list,rules=['odr3000','odrd'],recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['delay2'],trial_list=trial_list,rules=['odrd'],)
    plot_PSTH(hp, log, model_dir, 'odr3000', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odr3000', 'delay1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay2', trial_list, plot_oppo_dir = True)

#####################################################
    model_dir = 'data/set6_exp2'
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)
    print_basic_info(hp,log,model_dir,smooth_growth=True,smooth_window=5)
    trial_list = range(720640,log['trials'][-1]+1,1280)
    recompute = False
    compute_H(hp,log,model_dir, rules=['odr6000','odrd'], trial_list=trial_list, recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['stim1','delay1'],trial_list=trial_list,rules=['odr6000','odrd'],recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['delay2'],trial_list=trial_list,rules=['odrd'],)
    plot_PSTH(hp, log, model_dir, 'odr6000', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odr6000', 'delay1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay2', trial_list, plot_oppo_dir = True)
    '''
#######################################################
    '''
    model_dir = 'data/indi_odrd'
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)
    print_basic_info(hp,log,model_dir,smooth_growth=True,smooth_window=5)
    
    trial_list = range(899840,log['trials'][-1]+1,1280)
    recompute = False
    compute_H(hp,log,model_dir, rules=['odrd'], trial_list=trial_list, recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['stim1','delay1','delay2'],trial_list=trial_list,rules=['odrd'],recompute=recompute,)
    plot_PSTH(hp, log, model_dir, 'odrd', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay2', trial_list, plot_oppo_dir = True)
    
    odrd_distractor_analysis(hp,log,model_dir,trial_list=trial_list)

#####################################################
    model_dir = 'data/indi_odr_odrd'
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)
    print_basic_info(hp,log,model_dir,smooth_growth=True,smooth_window=5)
    
    trial_list = range(2000640,log['trials'][-1]+1,1280)
    recompute = False
    compute_H(hp,log,model_dir, rules=['odr','odrd'], trial_list=trial_list, recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['stim1','delay1'],trial_list=trial_list,rules=['odr','odrd'],recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['delay2'],trial_list=trial_list,rules=['odrd'],)
    plot_PSTH(hp, log, model_dir, 'odr', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odr', 'delay1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay2', trial_list, plot_oppo_dir = True)
    
    odrd_distractor_analysis(hp,log,model_dir,trial_list=trial_list)
    saccade_distribut_analysis(hp,log,'odr',model_dir,trial_list=trial_list,)
#####################################################
    model_dir = 'data/indi_overlap'
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)
    print_basic_info(hp,log,model_dir,smooth_growth=True,smooth_window=5)
    
    trial_list = range(0,log['trials'][-1]+1,1280)
    recompute = False
    compute_H(hp,log,model_dir, rules=['overlap',], trial_list=trial_list, recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['stim1'],trial_list=trial_list,rules=['overlap',],recompute=recompute,)
    plot_PSTH(hp, log, model_dir, 'overlap', 'stim1', trial_list, plot_oppo_dir = True)

    saccade_distribut_analysis(hp,log,'overlap',model_dir,trial_list=trial_list,)

#####################################################
    model_dir = 'data/indi_gap'
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)
    print_basic_info(hp,log,model_dir,smooth_growth=True,smooth_window=5)
    
    trial_list = range(0,log['trials'][-1]+1,1280)
    recompute = False
    compute_H(hp,log,model_dir, rules=['gap'], trial_list=trial_list, recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['stim1',],trial_list=trial_list,rules=['gap',],recompute=recompute,)
    plot_PSTH(hp, log, model_dir, 'gap', 'stim1', trial_list, plot_oppo_dir = True)

    saccade_distribut_analysis(hp,log,'gap',model_dir,trial_list=trial_list,)

#####################################################
    model_dir = 'data/indi_zero_gap'
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)
    print_basic_info(hp,log,model_dir,smooth_growth=True,smooth_window=5)
    
    trial_list = range(0,log['trials'][-1]+1,1280)
    recompute = False
    compute_H(hp,log,model_dir, rules=['zero_gap'], trial_list=trial_list, recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['stim1',],trial_list=trial_list,rules=['zero_gap',],recompute=recompute,)
    plot_PSTH(hp, log, model_dir, 'zero_gap', 'stim1', trial_list, plot_oppo_dir = True)

    saccade_distribut_analysis(hp,log,'zero_gap',model_dir,trial_list=trial_list,)
    '''
    '''
#####################################################
    model_dir = 'data/set7_exp1'
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)
    print_basic_info(hp,log,model_dir,smooth_growth=True,smooth_window=5)
    
    trial_list = range(0,log['trials'][-1]+1,1280)
    recompute = False
    compute_H(hp,log,model_dir, rules=['odr','odrd'], trial_list=trial_list, recompute=recompute,)
    compute_H(hp,log,model_dir, rules=['overlap','zero_gap','gap'], trial_list=trial_list, recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['stim1',],trial_list=trial_list,rules=['odr','odrd','overlap','zero_gap','gap'],recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['delay1',],trial_list=trial_list,rules=['odr','odrd'],recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['delay2',],trial_list=trial_list,rules=['odrd'],)

    plot_PSTH(hp, log, model_dir, 'odr', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odr', 'delay1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay2', trial_list, plot_oppo_dir = True)

    plot_PSTH(hp, log, model_dir, 'overlap', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'zero_gap', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'gap', 'stim1', trial_list, plot_oppo_dir = True)

    saccade_distribut_analysis(hp,log,'zero_gap',model_dir,trial_list=trial_list,)
    saccade_distribut_analysis(hp,log,'gap',model_dir,trial_list=trial_list,)
    saccade_distribut_analysis(hp,log,'overlap',model_dir,trial_list=trial_list,)

    saccade_distribut_analysis(hp,log,'odr',model_dir,trial_list=trial_list,)
    odrd_distractor_analysis(hp,log,model_dir,trial_list=trial_list)
    
#####################################################
    model_dir = 'data/set7_exp2'
    hp = tools.load_hp(model_dir)
    log = tools.load_log(model_dir)
    print_basic_info(hp,log,model_dir,smooth_growth=True,smooth_window=5)
    
    trial_list = range(0,log['trials'][-1]+1,1280)
    recompute = False
    compute_H(hp,log,model_dir, rules=['odr','odrd'], trial_list=trial_list, recompute=recompute,)
    compute_H(hp,log,model_dir, rules=['overlap','zero_gap','gap'], trial_list=trial_list, recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['stim1',],trial_list=trial_list,rules=['odr','odrd','overlap','zero_gap','gap'],recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['delay1',],trial_list=trial_list,rules=['odr','odrd'],recompute=recompute,)
    generate_neuron_info(hp,model_dir,epochs=['delay2',],trial_list=trial_list,rules=['odrd'],)

    plot_PSTH(hp, log, model_dir, 'odr', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odr', 'delay1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'odrd', 'delay2', trial_list, plot_oppo_dir = True)

    plot_PSTH(hp, log, model_dir, 'overlap', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'zero_gap', 'stim1', trial_list, plot_oppo_dir = True)
    plot_PSTH(hp, log, model_dir, 'gap', 'stim1', trial_list, plot_oppo_dir = True)

    saccade_distribut_analysis(hp,log,'zero_gap',model_dir,trial_list=trial_list,)
    saccade_distribut_analysis(hp,log,'gap',model_dir,trial_list=trial_list,)
    saccade_distribut_analysis(hp,log,'overlap',model_dir,trial_list=trial_list,)

    saccade_distribut_analysis(hp,log,'odr',model_dir,trial_list=trial_list,)
    odrd_distractor_analysis(hp,log,model_dir,trial_list=trial_list)
    '''
    