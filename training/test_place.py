"""
from matplotlib import pyplot as plt
import json

rules = ['odr500','odr750','odr1000','odr1100','odr1200','odr1300','odr1400','odr']

for rule in rules:
    fname = './data/'+rule+'_log.json'
    with open(fname, 'r') as f:
        log = json.load(f)

    fig_pref = plt.figure()
    plt.plot(log['trials'], log['perf_'+rule], label = rule)
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.title('Growth of Performance')
    plt.show()
"""
"""
import numpy as np 
a = np.array([[1,2,3],[4,5,6]])
b = a[:,1]
print(b.mean(axis=0))
"""
'''
from scipy.stats import ttest_rel
import numpy as np
x = np.array([20.5, 18.8, 19.8, 20.9, 21.5, 19.5, 21.0, 21.2])
y = np.array([17.7, 20.3, 20.0, 18.8, 19.0, 20.1, 20.0, 19.1])
#x = [20.5, 18.8, 19.8, 20.9, 21.5, 19.5, 21.0, 21.2]
#y = [17.7, 20.3, 20.0, 18.8, 19.0, 20.1, 20.0, 19.1]

# 配对样本t检验 
c = ttest_rel(x, y)
print(c)
'''
'''
h = dict()
if False or 1 not in h:
    print("OK")
'''
'''
a = [1,2,3]
b = [4,5,6]

a += b
print(a)
'''
'''
import numpy as np
a = [1,2,3,4]
b = [[1,1],[2,2]]
print(np.array(b).mean(axis=0))
print(np.array(a).mean(axis=0))
'''
'''
q = [1,1,1,2,2,2,3,3,3]
a = ['stim1','stim2','stim3','test','test']
b = ['stim1','key','key']
sta = len([ep for ep in a if 'stim' in ep])
stb = len([ep for ep in b if 'stim' in ep])
print(q[::sta])
print(q[::stb])
'''
'''
print(list(range(2,2)))
'''
'''
a = [1,2,3,4,5,6,7]
print(a[2-1::2])
'''
'''
import numpy as np
n_stim_loc, _ = batch_shape = 8, 16
batch_size = np.prod(batch_shape)
ind_stim_loc, _ = np.unravel_index(range(batch_size),batch_shape)
test = np.unravel_index(range(batch_size),(8*2,8))[0]%2
print('end')
'''
'''
a = [True,True,False,False]
b = [True,False,True,False]
c = [z[0] and not z[1] for z in zip(a,b)]
print(c)
'''
'''
a = [1,2,3,4]
print(a[0:])
'''
'''
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

a = dict()
a['test1'] = [1,2,3,4]
a['test2'] = [5,6,7,8]
a['test3'] = [9,10,11,12]

a_melt = dict()
a_melt['Maturation'] = list()
a_melt['Fire_rate'] = list()
for key,value in a.items():
    a_melt['Maturation'] += [key for i in range(len(value))]
    a_melt['Fire_rate'] += value

melted = pd.DataFrame(a_melt)
print(melted)
model = ols('Fire_rate~C(Maturation)',data=melted).fit()
anova_table = anova_lm(model, typ = 2)
print(anova_table)

print("\tP value:",anova_table['PR(>F)'][0])
print('group df:',anova_table['df'][0],'residual df:',anova_table['df'][1])
'''
'''
import sys
#import os
#print(os.getcwd())
#sys.path.append(os.getcwd())
sys.path.append('.')
#print(sys.path)
from utils import tools
'''