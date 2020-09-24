# RNN_maturation

Part of this code is adapted from the <a href="https://github.com/gyyang/multitask">Multitask</a> and <a href="https://github.com/nmasse/Short-term-plasticity-RNN">Short-term-plasticity-RNN</a>.

We train RNN to learn working memory task (ODR and ODRD) and anti-saccade task (Overlap, Zero-gap, and Gap).

<p align="center">
	<img src="https://github.com/xinzhoucs/RNN_BrainMaturation/blob/master/example/Tasks.jpg"  width="783" height="282">
</p>

## Dependencies

matplotlib, statsmodels, scipy, pandas, Tensorflow 1.9 or higher (but not 2.X)

## File Structure(Temporary)

```
├─data
   ├─6tasks
   │  ├─0
   │  ├─1280
   │  ├─2560
   │  ├─3840
   │  ...
```

## Training and Analysis

**Main_training.py** provides the main RNN model used in the paper (add link). Trained models would be saved in *data/6tasks/*

All analysis results of main RNN model in the paper can be reproduced by **Main_analysis.py**. Simply uncommenmt corresponding lines and run the script. 

### Analysis Function Instruction
**print_basic_info** would show you the performance growth curve and other basic information of the model, which can hlep you to decide which rules and trial range to analyze.

**compute_H/gen_task_info** both generate the information of tasks to be analyzed. compute_H would also save the hidden layer response as .pkl files to accelerate subsquent analysis procedure, while gen_task_info only save task information to save up storage. 

**generate_neuron_info** analyzes the neuron selectivity and save it as .pkl files.

**tunning_analysis**  plots neuron tunning feature. (corresponding to Fig.X in paper)
(PIC)

**

## More Training and Analysis

**More_training.py** contains a set of trainig examples with different hyperparameters (hp) or trained on different rules. Corresponding analysis code can be found in **More_training_analysis.py**. Please note that due to the difference in analysis procedure, analysis for MoN rules (match_or_non, match_or_non_easy and match_or_non_passive) is written in **MoN_analysis.py**. The training example set for MoN rules is *set5*
