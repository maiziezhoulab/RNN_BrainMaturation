from training.train_PSTH import train

if __name__ == '__main__':
    import argparse
    import os
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--modeldir', type=str, default='data/6tasks')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    hp = {'activation': 'softplus',
          'n_rnn': 256,
          'learning_rate': 0.001,
          'mix_rule': True,
          'l1_h': 0.,
          'use_separate_input': False,
          'adult_target_perf': 0.95,
          'young_target_perf': 0.65,
          'infancy_target_perf': 0.35,}

    train(args.modeldir,
        seed=0,
        hp=hp,
        ruleset='all_new',
        rule_trains=['overlap','zero_gap','gap','odr','odrd','gap500',],
        display_step=20)
