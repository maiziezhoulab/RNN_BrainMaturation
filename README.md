# RNN_maturation

## Dependencies

matplotlib, statsmodels, scipy, pandas, Tensorflow 1.9 or higher (but not 2.X)

## Training

*Training scripts (including train.py, task.py and network.py) are partly adapted from* <a href="https://github.com/gyyang/multitask">Multitask</a> 

We train RNN to learn working memory task (ODR and ODRD) and anti-saccade task (Overlap, Zero-gap, and Gap/Gap500).

<p align="center">
	<img src="https://github.com/xinzhoucs/RNN_BrainMaturation/blob/master/example/Tasks.jpg"  width="783" height="282">
</p>

**Main_training.py** provides the main RNN model used in the paper (add link). Trained models would be saved in *data/6tasks/*

### File Structure

After training, files in /data would be structured as follows:
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
      ├─1280
      │  ├─checkpoint
      │  ├─model.ckpt.data-00000-of-00001
      │  ├─model.ckpt.index
      │  └─model.ckpt.meta
      │ ...
```
## Analysis
*All analysis results of main RNN model in the paper can be reproduced by **Main_analysis.py**. Simply uncommenmt corresponding lines and run the script.*

### Analysis Function Instruction
**print_basic_info** would show you the performance growth curve and other basic information of the model, which can hlep you to decide which rules and trial range to analyze.

**compute_H/gen_task_info** both generate the information of tasks to be analyzed. compute_H would also save the hidden layer response as .pkl files to accelerate subsquent analysis procedure, while gen_task_info only save task information to save up storage. 

**generate_neuron_info** analyzes the neuron selectivity and save it as .pkl files.

**tunning_analysis**  plots neuron tunning feature. (corresponding to Fig.X in paper)
(PIC)

**plot_PSTH** plots the population activity of responsive RNN units
(PIC)

**plot_epoch_mean_growth** 

**seldir_analysis**

**neuron_period_activity_analysis**

**odrd_distractor_analysis**

**saccade_distribut_analysis**

**sample_neuron_by_trial**

### File Structure

## More Training and Analysis

**More_training.py** contains a set of trainig examples with different hyperparameters (hp) or trained on different rules. Corresponding analysis code can be found in **More_training_analysis.py**. Please note that due to the difference in analysis procedure, analysis for MoN rules (match_or_non, match_or_non_easy and match_or_non_passive) is written in **MoN_analysis.py**. The training example set for MoN rules is *set5*
