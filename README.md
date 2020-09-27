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
      ├─1280                          -----> Number of trails trained when saving the model, also noted as "trial number".
      │  ├─checkpoint                        We use it to distinguish models at different training stage.
      │  ├─model.ckpt.data-00000-of-00001
      │  ├─model.ckpt.index
      │  └─model.ckpt.meta
      │ ...
```
## Analysis
*All analysis results of main RNN model in the paper can be reproduced by **Main_analysis.py**. Simply uncommenmt corresponding lines and run the script.*

### Analysis Function Instruction
**print_basic_info** would show you the performance growth curve and other basic information of the model, which can hlep you to decide which rules and trial range to analyze.

![图片alt](https://github.com/maiziezhoulab/RNN_PFCmaturation/blob/master/example_pic/growth_of_performance.png)

**compute_H/gen_task_info** both generate the information of tasks to be analyzed. compute_H would also save the hidden layer response as .pkl files to accelerate subsquent analysis procedure, while gen_task_info only save task information to save up storage. 

(FILE STRUCTURE)

**generate_neuron_info** analyzes the neuron selectivity and save it as .pkl files.

(FILE STRUCTURE)

**tunning_analysis**  plots neuron tunning feature. (corresponding to Fig.X in paper)

(Example PIC)

**plot_PSTH** plots the population activity of responsive RNN units (corresponding to Fig.4, Fig.6, Fig.S3 and Fig.S6 in paper)

(Example PIC)

**plot_epoch_mean_growth** plots the mean fire rate value change of a specified epoch in a task during training 

(Example PIC)

**seldir_analysis** compares the RNN units' direction selectivity between two epochs from chosen tasks. Blue line represents the number of neurons that selective in both epochs. orange line shows the number of neurons which have same direction selectivity in these epochs, while the green line shows the number of neurons that behave differently on direction selectivity in these epochs. (Blue=Orange+Green)

(Example PIC)

**neuron_period_activity_analysis** plots the distribution of the neuron fire rate in a particular time period/epoch at different maturation stages

(Example PIC)

**saccade_distribut_analysis** plots the output/saccade direction distribution of a particular task at each maturation stage

(Example PIC)

**odrd_distractor_analysis** plots the output/saccade direction distribution of odrd task with different cue-distractor distances at each maturation stage

(Example PIC)

**sample_neuron_by_trial** generates a set of pictures for each chosen model which describes every responsive unit's direction selectivity by polar map

(Example PIC)

### Figure File Structure

## More Training and Analysis

**More_training.py** contains a set of trainig examples with different hyperparameters (hp) or trained on different rules. Corresponding analysis code can be found in **More_training_analysis.py**. Please note that due to the difference in analysis procedure, analysis for MoN rules (match_or_non, match_or_non_easy and match_or_non_passive) is written in **MoN_analysis.py**. The training example set for MoN rules is *set5*
