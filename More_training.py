from training.train_PSTH import train
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

###########################################################################
#Indi_odr##################################################################
modeldir = 'data/indi_odr'

hp = {'activation': 'softplus',
    'n_rnn': 256,
    'mix_rule': True,
    'l1_h': 0.,
    'use_separate_input': False}

train(modeldir,
        seed=0,
        hp=hp,
        ruleset='all_new',
        rule_trains=['odr',],
        display_step=20)

#Indi_odrd##################################################################
modeldir = 'data/indi_odrd'

hp = {'activation': 'softplus',
    'n_rnn': 256,
    'mix_rule': True,
    'l1_h': 0.,
    'use_separate_input': False}

train(modeldir,
        seed=0,
        hp=hp,
        ruleset='all_new',
        rule_trains=['odrd',],
        display_step=20)

#Indi_odr_odrd##################################################################
modeldir = 'data/indi_odr_odrd'

hp = {'activation': 'softplus',
    'n_rnn': 256,
    'mix_rule': True,
    'l1_h': 0.,
    'use_separate_input': False}

train(modeldir,
        seed=0,
        hp=hp,
        ruleset='all_new',
        rule_trains=['odr','odrd'],
        display_step=20)

#Indi_gap##################################################################
modeldir = 'data/indi_gap'

hp = {'activation': 'softplus',
    'n_rnn': 256,
    'mix_rule': True,
    'l1_h': 0.,
    'use_separate_input': False}

train(modeldir,
        seed=0,
        hp=hp,
        ruleset='all_new',
        rule_trains=['gap',],
        display_step=20)

#Indi_zero_gap##################################################################
modeldir = 'data/indi_zero_gap'

hp = {'activation': 'softplus',
    'n_rnn': 256,
    'mix_rule': True,
    'l1_h': 0.,
    'use_separate_input': False}

train(modeldir,
        seed=0,
        hp=hp,
        ruleset='all_new',
        rule_trains=['zero_gap',],
        display_step=20)

#Indi_overlap##################################################################
modeldir = 'data/indi_overlap'

hp = {'activation': 'softplus',
    'n_rnn': 256,
    'mix_rule': True,
    'l1_h': 0.,
    'use_separate_input': False}

train(modeldir,
        seed=0,
        hp=hp,
        ruleset='all_new',
        rule_trains=['overlap',],
        display_step=20)

###############################################################################
#set1_exp1#####################################################################
modeldir = 'data/set1_exp1'

hp = {'activation': 'softplus',
    'n_rnn': 64,
    'mix_rule': True,
    'l1_h': 0.,
    'use_separate_input': False}

train(modeldir,
        seed=0,
        hp=hp,
        ruleset='all_new',
        rule_trains=["overlap", "zero_gap", "gap", "odr", "odrd", "gap500",],
        display_step=20)

#set1_exp2#####################################################################
modeldir = 'data/set1_exp2'

hp = {'activation': 'softplus',
    'n_rnn': 128,
    'mix_rule': True,
    'l1_h': 0.,
    'use_separate_input': False}

train(modeldir,
        seed=0,
        hp=hp,
        ruleset='all_new',
        rule_trains=["overlap", "zero_gap", "gap", "odr", "odrd", "gap500",],
        display_step=20)

#set1_exp3#####################################################################
modeldir = 'data/set1_exp3'

hp = {'activation': 'softplus',
    'n_rnn': 512,
    'mix_rule': True,
    'l1_h': 0.,
    'use_separate_input': False}

train(modeldir,
        seed=0,
        hp=hp,
        ruleset='all_new',
        rule_trains=["overlap", "zero_gap", "gap", "odr", "odrd", "gap500",],
        display_step=20)

#set1_exp4#####################################################################
modeldir = 'data/set1_exp4'

hp = {'activation': 'softplus',
    'n_rnn': 1024,
    'mix_rule': True,
    'l1_h': 0.,
    'use_separate_input': False}

train(modeldir,
        seed=0,
        hp=hp,
        ruleset='all_new',
        rule_trains=["overlap", "zero_gap", "gap", "odr", "odrd", "gap500",],
        display_step=20)

###############################################################################
#set2_exp1#####################################################################
modeldir = 'data/set2_exp1'

hp = {'activation': 'tanh',
    'n_rnn': 256,
    'mix_rule': True,
    'l1_h': 0.,
    'use_separate_input': False}

train(modeldir,
        seed=0,
        hp=hp,
        ruleset='all_new',
        rule_trains=["overlap", "zero_gap", "gap", "odr", "odrd", "gap500",],
        display_step=20)

#set2_exp2#####################################################################
modeldir = 'data/set2_exp2'

hp = {'activation': 'relu',
    'n_rnn': 256,
    'mix_rule': True,
    'l1_h': 0.,
    'use_separate_input': False}

train(modeldir,
        seed=0,
        hp=hp,
        ruleset='all_new',
        rule_trains=["overlap", "zero_gap", "gap", "odr", "odrd", "gap500",],
        display_step=20)

#set2_exp3#####################################################################
modeldir = 'data/set2_exp3'

hp = {'activation': 'elu',
        'n_rnn': 256,
        'mix_rule': True,
        'l1_h': 0.,
        'use_separate_input': False}

train(modeldir,
        seed=0,
        hp=hp,
        ruleset='all_new',
        rule_trains=["overlap", "zero_gap", "gap", "odr", "odrd", "gap500",],
        display_step=20)

#set2_exp4#####################################################################
modeldir = 'data/set2_exp4'

hp = {'activation': 'power',
    'n_rnn': 256,
    'mix_rule': True,
    'l1_h': 0.,
    'use_separate_input': False}

train(modeldir,
        seed=0,
        hp=hp,
        ruleset='all_new',
        rule_trains=["overlap", "zero_gap", "gap", "odr", "odrd", "gap500",],
        display_step=20)

###############################################################################
#set3_exp1#####################################################################
modeldir = 'data/set3_exp1'

hp = {'activation': 'softplus',
    'w_rec_init': 'diag',
    'n_rnn': 256,
    'mix_rule': True,
    'l1_h': 0.,
    'use_separate_input': False}

train(modeldir,
        seed=0,
        hp=hp,
        ruleset='all_new',
        rule_trains=["overlap", "zero_gap", "gap", "odr", "odrd", "gap500",],
        display_step=20)

#set3_exp2#####################################################################
modeldir = 'data/set3_exp2'

hp = {'activation': 'softplus',
    'w_rec_init': 'randgauss',
    'n_rnn': 256,
    'mix_rule': True,
    'l1_h': 0.,
    'use_separate_input': False}

train(modeldir,
        seed=0,
        hp=hp,
        ruleset='all_new',
        rule_trains=["overlap", "zero_gap", "gap", "odr", "odrd", "gap500",],
        display_step=20)

###############################################################################
#set4_exp1#####################################################################
modeldir = 'data/set4_exp1'

hp = {'activation': 'softplus',
    'learning_rate': 0.0001,
    'n_rnn': 256,
    'mix_rule': True,
    'l1_h': 0.,
    'use_separate_input': False}

train(modeldir,
        seed=0,
        hp=hp,
        ruleset='all_new',
        rule_trains=["overlap", "zero_gap", "gap", "odr", "odrd", "gap500",],
        display_step=20)

#set4_exp2#####################################################################
modeldir = 'data/set4_exp2'

hp = {'activation': 'softplus',
    'learning_rate': 0.01,
    'n_rnn': 256,
    'mix_rule': True,
    'l1_h': 0.,
    'use_separate_input': False}

train(modeldir,
        seed=0,
        hp=hp,
        ruleset='all_new',
        rule_trains=["overlap", "zero_gap", "gap", "odr", "odrd", "gap500",],
        display_step=20)

#set4_exp3#####################################################################
modeldir = 'data/set4_exp3'

hp = {'activation': 'softplus',
    'learning_rate': 0.1,
    'n_rnn': 256,
    'mix_rule': True,
    'l1_h': 0.,
    'use_separate_input': False}

train(modeldir,
        seed=0,
        hp=hp,
        ruleset='all_new',
        rule_trains=["overlap", "zero_gap", "gap", "odr", "odrd", "gap500",],
        display_step=20)

###############################################################################
#set5_exp1#####################################################################
modeldir = 'data/set5_exp1'

hp = {'activation': 'softplus',
    'n_rnn': 256,
    'mix_rule': True,
    'l1_h': 0.,
    'use_separate_input': False}
    
train(modeldir,
        seed=0,
        hp=hp,
        ruleset='mix_MoN_6tasks',
        rule_trains=['match_or_non','match_or_non_easy','overlap','zero_gap','gap','odr','odrd','gap500',],
        display_step=20)

#set5_exp2#####################################################################
modeldir = 'data/set5_exp2'

hp = {'activation': 'softplus',
    'n_rnn': 256,
    'mix_rule': True,
    'l1_h': 0.,
    'use_separate_input': False}
    
train(modeldir,
        seed=0,
        hp=hp,
        ruleset='mix_p_MoN_6tasks',
        rule_trains=['match_or_non','match_or_non_easy','match_or_non_passive','overlap','zero_gap','gap','odr','odrd','gap500',],
        display_step=20)

###############################################################################
#set6_exp1#####################################################################
modeldir = 'data/set6_exp1'

hp = {'activation': 'softplus',
    'n_rnn': 256,
    'mix_rule': True,
    'l1_h': 0.,
    'use_separate_input': False}

train(modeldir,
        seed=0,
        hp=hp,
        ruleset='all_new_odr3000',
        rule_trains=['overlap','zero_gap','gap','odr3000','odrd','gap500',],
        display_step=20)

#set6_exp2#####################################################################
modeldir = 'data/set6_exp2'

hp = {'activation': 'softplus',
    'n_rnn': 256,
    'mix_rule': True,
    'l1_h': 0.,
    'use_separate_input': False}
    
train(modeldir,
        seed=0,
        hp=hp,
        ruleset='all_new_odr6000',
        rule_trains=['overlap','zero_gap','gap','odr6000','odrd','gap500',],
        display_step=20)

#set6_exp3#####################################################################
modeldir = 'data/set6_exp3'

hp = {'activation': 'softplus',
    'n_rnn': 256,
    'mix_rule': True,
    'l1_h': 0.,
    'use_separate_input': False}
    
train(modeldir,
        seed=0,
        hp=hp,
        ruleset='all_new_odr15000',
        rule_trains=['overlap','zero_gap','gap','odr15000','odrd','gap500',],
        display_step=20)

###############################################################################
#set7_exp1#####################################################################
modeldir = 'data/set7_exp1'

hp = {'activation': 'softplus',
    'n_rnn': 256,
    'learning_rate': 0.001,
    'mix_rule': True,
    'l1_h': 0.,
    'use_separate_input': False,
    'rnn_type': 'LeakyGRU',}

train(modeldir,
        seed=0,
        hp=hp,
        ruleset='all_new',
        rule_trains=['overlap','zero_gap','gap','odr','odrd','gap500',],
        display_step=20)

#set7_exp2#####################################################################
modeldir = 'data/set7_exp2'

hp = {'activation': 'softplus',
    'n_rnn': 256,
    'learning_rate': 0.001,
    'mix_rule': True,
    'l1_h': 0.,
    'use_separate_input': False,
    'rnn_type': 'GRU',}

train(modeldir,
        seed=0,
        hp=hp,
        ruleset='all_new',
        rule_trains=['overlap','zero_gap','gap','odr','odrd','gap500',],
        display_step=20)