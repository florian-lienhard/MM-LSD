#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
from itertools import combinations,product


# In[2]:


#star = "teststar"
star = str(sys.argv[1])

stardir = './stars/'+star+'/'
inf_name = stardir + 'input.py'

exec(open(inf_name).read())


# In[3]:


params = pd.DataFrame()


listOLists =  [grid["max_nr_of_specs"],
               grid["velgridwidth"],
               grid["telluric_cut"],
               grid["mindepthparam"],
               grid["maxdepthparam"],
               grid["modelspecdeviationcut"],
               grid["exclwidelinesparam"],
               grid["telloption"],
               grid["erroption"],
               grid["rassoption"]]


# In[4]:


em = np.zeros((np.prod([len(l) for l in listOLists])))

count = 0

for keyword in grid.keys():
    params[keyword] = em    
    
for paramlist in product(*listOLists):

    for cn,keyword in enumerate(grid.keys()):
        if cn<len(listOLists):
            params[keyword][count] = paramlist[cn] 
    count+=1     
                                            
params.to_csv('./stars/'+star+'/params.csv',index=False)

print(params.shape,count)
count_later = np.copy(count)


# In[5]:


results = pd.DataFrame()

for keyword in grid.keys():
    results[keyword] = []

results["LSD RV std"] = []
results["LSD RV MAD"] = []
results["DRS RV std"] = []
results["DRS RV MAD"] = []
results["sigmafit_used"] = []
results["comp time"] = []

results.to_csv(resdir+f"results_{star}_{indic}.csv",index=False)


# In[27]:


if os.path.exists(rvresfile): 
    os.remove(rvresfile)
    os.remove(rverrresfile)
    os.remove(commonprofilefile)

