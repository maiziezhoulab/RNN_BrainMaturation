# RNN_BrainMaturation

## Dependencies
matplotlib, statsmodels, scipy, pandas, Tensorflow 1.9 or higher (but not 2.X)

## Colab


## Training

*Training scripts (including train.py, task.py and network.py) are partly adapted from* <a href="https://github.com/gyyang/multitask">Multitask</a> 

We train RNNs to learn working memory tasks (ODR and ODRD) and anti-saccade tasks (Overlap, Zero-gap, and Gap/Gap500).

<p align="center">
	<img src="https://github.com/xinzhoucs/RNN_BrainMaturation/blob/master/example/Tasks.jpg"  width="783" height="282">
</p>

**Main_training.py** provides the main RNN models used in the paper, and trained models will be saved in *data/6tasks/*

### File Structure

After training, files in */data* would be structured as follows:
```
├─data
   └─6tasks
      ├─hp.json                       -----> Hyperparameters              
      ├─log.json                      -----> Training logs
      ├─0
      │  ├─checkpoint                      ┐
      │  ├─model.ckpt.data-00000-of-00001  |
      │  ├─model.ckpt.index                |——Model saved
      │  └─model.ckpt.meta                 ┘
      ├─1280                          -----> Number of trails trained when saving the model, also noted as "trial number".
      │  ├─checkpoint                        We use it to distinguish models at different training stage.
      │  ├─model.ckpt.data-00000-of-00001
      │  ├─model.ckpt.index
      │  └─model.ckpt.meta
      │ ...
```
## Analysis
Most of the analysis in the paper can be reproduced by **Main_analysis.py**. Simply uncommenmt corresponding lines and run the script.

### Analysis Function Instruction
**print_basic_info** would show the task performance during training and other basic information of the model, which can help you to decide which tasks(rules) and trials range(performance range) to analyze (corresponding to Fig.S7 in the paper).

<p align="center">
	<img src="https://github.com/maiziezhoulab/RNN_PFCmaturation/blob/master/example_pic/growth_of_performance.png"  width="800">
</p>

**compute_H/gen_task_info** both generate the information of tasks to be analyzed. *compute_H* would also save the activities of RNN units of the hidden layer as .pkl files to accelerate subsequent analysis procedure.

```
├─data
   └─6tasks
      ├─hp.json           
      ├─log.json
      ├─task_info.pkl            --->compute_H/gen_task_info
      ├─0
      │  ├─H_gap.pkl             ┐
      │  ├─H_odr.pkl             |
      │  ├─H_odrd.pkl            |-compute_H
      │  ├─H_overlap.pkl         |
      │  ├─H_zero_gap.pkl        ┘
      │  ├─checkpoint
      │  ├─model.ckpt.data-00000-of-00001
      │  ├─model.ckpt.index             
      │  └─model.ckpt.meta
      │ ...
```

**generate_neuron_info** analyzes the selectivities of RNN units and save them as .pkl files for further analysis.

```
├─data
   └─6tasks
      ├─hp.json           
      ├─log.json
      ├─task_info.pkl            --->compute_H/gen_task_info
      ├─0
      │  ├─H_gap.pkl             ┐
      │  ├─H_odr.pkl             |
      │  ├─H_odrd.pkl            |-compute_H
      │  ├─H_overlap.pkl         |
      │  ├─H_zero_gap.pkl        ┘
      │  ├─checkpoint
      │  ├─model.ckpt.data-00000-of-00001
      │  ├─model.ckpt.index             
      │  ├─model.ckpt.meta
      │  ├─neuron_info_gap_stim1.pkl      ┐
      │  ├─neuron_info_odrd_delay1.pkl    |
      │  ├─neuron_info_odrd_delay2.pkl    |
      │  ├─neuron_info_odrd_stim1.pkl     |
      │  ├─neuron_info_odr_delay1.pkl     |-generate_neuron_info
      │  ├─neuron_info_odr_stim1.pkl      |
      │  ├─neuron_info_overlap_stim1.pkl  |
      │  └─neuron_info_zero_gap_stim1.pkl ┘
      │ ...
```

**tunning_analysis**  plots tunning curves of RNN units (corresponding to Fig.S3 in the paper).

<p align="center">
	<img src="https://github.com/maiziezhoulab/RNN_PFCmaturation/blob/master/example_pic/odrd_stim1_(520960%2C%20628480)_step_1280_tuning_analysis.png"  width="800">
</p>

**plot_PSTH** plots mean rate of the RNN units responsive to the ODRD task, during three developmental stages (corresponding to Fig.4 in the paper).

<p align="center">
	<img src="https://github.com/maiziezhoulab/RNN_PFCmaturation/blob/master/example_pic/odrd_stim1_(520960%2C%20628480)_step_1280_PSTH.png"  width="800">
</p>

**plot_epoch_mean_growth** plots RNN activity during the course of training in the delay period (corresponding to Fig.S2B in the paper).

<p align="center">
	<img src="https://github.com/maiziezhoulab/RNN_PFCmaturation/blob/master/example_pic/delay1_epoch_mean_growth.png"  width="800">
</p>

**seldir_analysis** compares the RNN units' direction selectivity between two epochs from chosen tasks. Blue line represents the number of neurons that selective in both epochs. orange line shows the number of neurons which have same direction selectivity in these epochs, while the green line shows the number of neurons that behave differently on direction selectivity in these epochs. (Blue=Orange+Green) (corresponding to Fig.S4 in paper)

<p align="center">
	<img src="https://github.com/maiziezhoulab/RNN_PFCmaturation/blob/master/example_pic/odr_stim1_odrd_stim1exh_neurons_mix_neurons_seldir_analysis.png"  width="800">
</p>

**neuron_period_activity_analysis** plots the distribution of the neuron fire rate in a particular time period/epoch at different maturation stages (corresponding to Fig.S1 in paper).

<p align="center">
	<img src="https://github.com/maiziezhoulab/RNN_PFCmaturation/blob/master/example_pic/odrd_delay2_activity_oneway_anova_analysis.png"  width="800">
</p>



**sample_neuron_by_trial** plots activity of a single example unit in working memory task (corresponding to Fig.3 in the paper)

<p align="center">
	<img src="https://github.com/maiziezhoulab/RNN_PFCmaturation/blob/master/example_pic/115.png"  width="800">
</p>

### Figure File Structure

## More Training and Analysis

**More_training.py** contains a set of trainig examples with different hyperparameters (hp) or trained on different rules. Corresponding analysis code can be found in **More_training_analysis.py**. Please note that due to the difference in analysis procedure, analysis for MoN rules (match_or_non, match_or_non_easy and match_or_non_passive) is written in **MoN_analysis.py**. The training example set for MoN rules is *set5*
