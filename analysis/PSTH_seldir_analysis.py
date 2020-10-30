# basic packages #
import os
import numpy as np
import pickle

# plot #
import matplotlib.pyplot as plt

#
import pandas as pd

def seldir_analysis(
                    model_dir,
                    rule1,
                    epoch1,
                    rule2,
                    epoch2,
                    trial_list,
                    n_types,
                    ):
    
    plot_dic = dict()
    for item_ in ['inter','samedir','diffdir']:
        plot_dic[item_] = list()

    for trial_num in trial_list:
        read_name1 = model_dir+'/'+str(trial_num)+'/neuron_info_'+rule1+'_'+epoch1+'.pkl'
        read_name2 = model_dir+'/'+str(trial_num)+'/neuron_info_'+rule2+'_'+epoch2+'.pkl'

        with open(read_name1,'rb') as nf1:
            ninf1 = pickle.load(nf1)

        with open(read_name2,'rb') as nf2:
            ninf2 = pickle.load(nf2)

        n1 = list()
        n2 = list()
        for n_type in n_types:
            for x in ninf1[n_type]:
                n1.append(x[0])
            for x in ninf2[n_type]:
                n2.append(x[0])
        inter_n = list(set(n1)&set(n2))
        plot_dic['inter'].append(len(inter_n))


        n1_seldir = set()
        n2_seldir = set()
        for n_type in n_types:
            temp1 = set([(x[0],x[2]) for x in ninf1[n_type] if x[0] in inter_n])
            temp2 = set([(x[0],x[2]) for x in ninf2[n_type] if x[0] in inter_n])
            n1_seldir = n1_seldir.union(temp1)
            n2_seldir = n2_seldir.union(temp2)

        same_dir = len(n1_seldir&n2_seldir)
        diff_dir = len(n1_seldir.difference(n2_seldir))
        plot_dic['samedir'].append(same_dir)
        plot_dic['diffdir'].append(diff_dir)

    fig, ax = plt.subplots()
    ax.plot(trial_list,plot_dic['inter'],label = 'neuron intersection')
    ax.plot(trial_list,plot_dic['samedir'],label = 'same selective direction')
    ax.plot(trial_list,plot_dic['diffdir'],label = 'different selective direction')

    ax.legend()

    plt.savefig('figure/figure_'+model_dir.rstrip('/').split('/')[-1]+'/'+rule1+'_'+epoch1+'_'+rule2+'_'+epoch2+'_'.join(n_types)+'_seldir_analysis'+'.pdf')
    plt.savefig('figure/figure_'+model_dir.rstrip('/').split('/')[-1]+'/'+rule1+'_'+epoch1+'_'+rule2+'_'+epoch2+'_'.join(n_types)+'_seldir_analysis'+'.png')


    #save excel
    writer = pd.ExcelWriter('figure/figure_'+model_dir.rstrip('/').split('/')[-1]+'/'+rule1+'_'+epoch1+'_'+rule2+'_'+epoch2+'_'.join(n_types)+'_seldir_analysis'+'.xlsx')

    excel_data = list()
    col = ['inter','samedir','diffdir']
    row = [str(n) for n in trial_list]
    for item_ in col:
        excel_data.append(plot_dic[item_])

    excel_data = np.array(excel_data).T
    df = pd.DataFrame(data=excel_data,  columns=col, index=row)
    df.to_excel(writer)
