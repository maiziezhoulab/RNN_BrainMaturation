# RNN_BrainMaturation

## Dependencies
matplotlib, statsmodels, scipy, pandas, Tensorflow 1.9 or higher (but not 2.X)


## Colab
Here we also provide a jupyter notebook **RNN_BrainMaturation.ipynb** that you can directly use in <a href="https://www.tutorialspoint.com/google_colab/google_colab_executing_external_python_files.htm">Google Colab</a> to train the models and perform some main analyses. 

*To use **RNN_BrainMaturation.ipynb**, download "RNN_BrainMaturation" via git clone and save it in the google drive.


## Pretrained model

Model used in paper can be downloaded from  <a href="https://drive.google.com/file/d/1l6Gg0vUnDVpGUhJAEG86UaMD1d3BHtWu/view?usp=sharing">here</a>

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

### Main Analysis Function Instruction
**print_basic_info** would show the task performance during training and other basic information of the model, which can help you to decide which tasks(rules) and trials range(performance range) to analyze (corresponding to Fig.S7 in the paper).

<p align="center">
	<img src="https://github.com/maiziezhoulab/RNN_PFCmaturation/blob/master/example_pic/growth_of_performance.png"  width="400">
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

**plot_PSTH** plots mean rate of the RNN units responsive to the ODRD task, during three developmental stages (corresponding to Fig.4E,F,G in the paper).

<p align="center">
	<img src="https://github.com/maiziezhoulab/RNN_PFCmaturation/blob/master/example_pic/odrd_stim1_(520960%2C%20628480)_step_1280_PSTH.png"  width="500">
</p>

**sample_neuron_by_trial** plots activity of a single example unit in working memory task (corresponding to Fig.3B in the paper, but not the same unit)

<p align="center">
	<img src="https://github.com/maiziezhoulab/RNN_PFCmaturation/blob/master/example_pic/115.png"  width="500">
</p>

**neuron_dPCA**  dPCA analysis performed for RNN units in the  mature networks (corresponding to Fig.5E in the paper).

<p align="center">
	<img src="https://github.com/maiziezhoulab/RNN_BrainMaturation/blob/master/example_pic/dPCA_rnn_odr_adult.png"  width="700">
</p>


**plot_epoch_mean_growth** plots RNN activity during the course of training in the delay period (corresponding to Fig.S2B in the paper).

<p align="center">
	<img src="https://github.com/maiziezhoulab/RNN_PFCmaturation/blob/master/example_pic/delay1_epoch_mean_growth.png"  width="400">
</p>

**tunning_analysis**  plots tunning curves of RNN units in the cue period (corresponding to Fig.S3B in the paper).

<p align="center">
	<img src="https://github.com/maiziezhoulab/RNN_PFCmaturation/blob/master/example_pic/odrd_stim1_(520960%2C%20628480)_step_1280_tuning_analysis.png"  width="500">
</p>

**Decoder_analysis**  Plots cross-temporal decoding accuracy in the ODR task for RNN data (corresponding to Fig.S5E in the paper).

<p align="center">
	<img src="https://github.com/maiziezhoulab/RNN_BrainMaturation/blob/master/example_pic/odr_628480_w200ms_s20ms.png"  width="300">
</p>



## Training with different hyper-parameters

**Training_with_diff_HyperParameters.py** contains a set of trainig examples with different hyperparameters (hp) or trained on different rules. Corresponding analysis code can be found in **Analysis_with_diff_HyperParameters.py**. 

## Cite the Work
Emergence of prefrontal neuron maturation properties by training recurrent neural networks in cognitive tasks. Y. H. Liu, J. Zhu, C. Constantinidis, X. Zhou. iScience (2021) 24(10):103178.


## Code DOI
https://doi.org/10.5281/zenodo.5518223

## Troubleshooting
#### Please submit issues on the github page for <a href="https://github.com/maiziezhoulab/RNN_BrainMaturation/issues">RNN_BrainMaturation</a>. 
