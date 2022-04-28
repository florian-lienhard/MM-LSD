#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import sys
import subprocess
import os
from pandas import read_csv
import pickle
import matplotlib.pyplot as plt
from matplotlib import rc,rcParams
from pandas import DataFrame

#tested on Ubuntu 18.04.6 LTS

rc('xtick',labelsize=15)
rc('ytick',labelsize=15)
rcParams['font.family'] = 'STIXGeneral'
rcParams['mathtext.fontset'] = 'stix'

plt.rcParams.update({
    "font.size":15,
    "figure.facecolor":  "white",  
    "axes.facecolor":    "white", 
    "savefig.facecolor": "white", 
})


# In[ ]:


# -------------------------------------------------------------
#           INITIALIZATION
# -------------------------------------------------------------

# Load the input file
star = str(sys.argv[1])
#star = "Sun"
#star = "Kepler-21"

#location of input file
stardir = './stars/'+star+'/'

# Create path to the input.py file
inf_name = stardir+'input.py'

# Did you create an input.py file?
if (not os.path.isfile(inf_name)):
    print('You have not created', inf_name)
    sys.exit()
    
#get all definitions in input.py
exec(open(inf_name).read())


# In[ ]:


# -------------------------------------------------------------
#           PREPROCESS SPECTRA AND SAVE RESULTS
# -------------------------------------------------------------

#THIS HAS TO BE RUN ONLY ONCE!

if preprocess:
    if not os.path.exists(rassine_loc+"/spectra_library/stars"):
        os.mkdir(rassine_loc+"/spectra_library/stars")
    
    #os.popen(f"python preprocess.py {star}").read()
    preprocess_command_run = subprocess.call([sys.executable, 'preprocess.py', star])
    
    if preprocess_command_run!=0:
        print("Some error. Run this file independently or copy the relevant jupyter notebook to check.")


# In[ ]:





# In[ ]:


# -------------------------------------------------------------
#           GENERATE GRID (PARAMETER COMBINATIONS)
# -------------------------------------------------------------


#deletes old pkl files!!

if generate_grid:
    #os.popen(f"python generate_grid.py {star}").read()
    generate_grid_command_run = subprocess.call([sys.executable, 'generate_grid.py', star])
    if generate_grid_command_run!=0:
        print("Some error. Run this file independently or copy the relevant jupyter notebook to check.")


# In[ ]:


# -------------------------------------------------------------
#           READ GRID
# -------------------------------------------------------------

parameter_grid = read_csv(stardir+'params.csv')


# In[ ]:


# -------------------------------------------------------------
#           RUN LSD ON GRID
# -------------------------------------------------------------

if run_on_grid:
    for grid_point in parameter_grid.index:
        #a = os.popen(f"python run_lsd.py {star} {grid_point}").read()
        command_run = subprocess.call([sys.executable, 'run_lsd.py', star, str(grid_point)])
        if command_run != 0:
            break
            print("Some error here.")
            


# In[ ]:


# -------------------------------------------------------------
#           LOAD RESULTS
# -------------------------------------------------------------

with open(rvresfile,"rb") as f:
        rv = pickle.load(f)
with open(rverrresfile,"rb") as f:
        rv_err = pickle.load(f)
with open(commonprofilefile,"rb") as f:
        commonprofiles = pickle.load(f)

grid_results_file = f"{resdir}results_{star}_{indic}.csv"
grid_results = read_csv(grid_results_file)

#info_file = read_csv(dirdir+"Info.csv")
#info_file.keys()

rv_ccf = rv["rv_ccf"]
rv_ccf_err = rv["rv_ccf_err"]
mjd = rv["mjd"]


# In[ ]:


for key in rv.keys():
    if key in parameter_grid.index:
        #if you want to use the common profiles: extract the indicators for all parameter combinations
        #or combine the common profiles from different parameter combinations
        #note that different parameter combinations lead to the inclusion of different line sets
        
        #velocity grid for parameter combination (same for all spectra)
        vel_key_i = commonprofiles[f"vel_{key}"]
        #common profiles for parameter combination
        Zs_key_i = commonprofiles[f"Z_{key}"]
        
        plt.figure(figsize=(10,5))
        plt.title(f"Common profiles for parameter combination {key}")
        for i in range(len(Zs_key_i)):
            plt.plot(vel_key_i,Zs_key_i[i],".-",label=f"spectrum {i}")
        plt.xlabel("vel [km/s]")
        plt.legend();
        break


# In[ ]:


# -------------------------------------------------------------
#           GENERATE MM-LSD RVS
# -------------------------------------------------------------

#empty containers
rv_mat = np.zeros((len(parameter_grid.index),len(rv_ccf)))
rv_err_mat = np.zeros((len(parameter_grid.index),len(rv_ccf)))

#get all time series (RVs & RV errors from different parameter combinations)
for key in rv.keys():
    if key in parameter_grid.index:
        rv_mat[key,:] = rv[key]
        rv_err_mat[key,:] = rv_err[key]
        
#RMS of the individual RV time series
rvstds = np.std(rv_mat,axis=1)

#find time series with lowest RMS
maxstd = rvstds[np.argsort(rvstds)[use_n_time_series]]
low_std_rvs = np.where(rvstds<maxstd)[0]

print(f"selected {len(low_std_rvs)} time series")

#mean-combine all RV time series
rv_all_mean = np.mean(rv_mat,axis=0)
rv_err_all_mean = np.mean(rv_err_mat,axis=0)

#mean-combine use_n_time_series RV time series with lowest RMS
rv_selection_mean = np.mean(rv_mat[low_std_rvs,:],axis=0)
rv_err_selection_mean = np.mean(rv_err_mat[low_std_rvs,:],axis=0)


# In[ ]:





# In[ ]:


# -------------------------------------------------------------
#           PLOT RESULTS
# -------------------------------------------------------------


plt.figure(figsize=(10,5))
plt.title("RV time series")

for i in np.arange(rv_mat.shape[0]):
    plt.plot(mjd,rv_mat[i,:],".",alpha=0.2)
plt.plot(mjd,rv_ccf,"D-",color="black",label="DRS CCF RVs")
plt.plot(mjd,rv_all_mean,"D-",color="blue",label=f"Mean {nr_of_combinations}/{nr_of_combinations} RVs")
plt.plot(mjd,rv_selection_mean,"D-",color="violet",label=f"Mean {use_n_time_series}/{nr_of_combinations} RVs")

plt.xlabel("MJD")
plt.ylabel("RV [m/s]")
plt.legend()
plt.savefig(resdir+f"RV_time_series.pdf");


# In[ ]:


plt.figure(figsize=(10,5))
plt.title("RV RMS Histogram")

plt.hist(rvstds,color="lightgray")
plt.axvline(np.std(rv_ccf),color="black",linewidth=3,label="DRS CCF")
plt.axvline(np.std(rv_all_mean),color="blue",linewidth=3,label=f"Mean {nr_of_combinations}/{nr_of_combinations} RVs")
plt.axvline(np.std(rv_selection_mean),color="violet",linewidth=3,label=f"Mean {use_n_time_series}/{nr_of_combinations} RVs")

plt.xlabel("RV RMS [m/s]")
plt.ylabel("Count")
plt.legend()
plt.savefig(resdir+f"RV_histogram.pdf");


# In[ ]:


results = DataFrame()
results["mjd"] = mjd
results["rv_all_mean"] = rv_all_mean
results["rv_err_all_mean"] = rv_err_all_mean
results["rv_selection_mean"] = rv_selection_mean
results["rv_err_selection_mean"] = rv_err_selection_mean

results["rv_ccf"] = rv_ccf
results["rv_ccf_err"] = rv_ccf_err

results.to_csv(resdir+f"RVs_{indic}.csv")


# In[ ]:


try:
    import seaborn as sns
    rc('text', usetex = True)

    fig,axes = plt.subplots(2,2,figsize=(8,5),sharey=True)


    meas = "LSD RV std"
    measdrs = "DRS RV std"

    colors = ["#27ae60","#2ecc71"]
    palette = sns.color_palette(colors)

    sns.violinplot(x="modelspecdeviationcut", y=meas, bw=.35,cut=0,data=grid_results,ax = axes[0,0],palette=palette)
    axes[0,0].axhline(grid_results[measdrs].iloc[0],color="gray")
    axes[0,0].set_xlabel("$\mathbf{\Gamma}$",fontsize=14)

    colors = ["#8e44ad", "#9b59b6"]
    palette = sns.color_palette(colors)

    sns.violinplot(x="maxdepthparam", y=meas, bw=.35,cut=0,data=grid_results,ax = axes[0,1],palette=palette)
    axes[0,1].axhline(grid_results[measdrs].iloc[0],color="gray")
    axes[0,1].set_xlabel("$\mathbf{\Xi}$",fontsize=14)

    colors = ["#f7dc6f","#f8c471","#f0b27a","#e59866"]
    palette = sns.color_palette(colors)

    sns.violinplot(x="velgridwidth", y=meas, bw=.35,cut=0,data=grid_results,ax = axes[1,0],palette=palette)
    axes[1,0].axhline(grid_results[measdrs].iloc[0],color="gray")
    axes[1,0].set_xlabel("$\mathbf{\Phi}$",fontsize=14)

    colors = [ "#2980b9" , "#3498db", "#3498db"]
    palette = sns.color_palette(colors)

    sns.violinplot(x="telluric_cut", y=meas, bw=.35,cut=0,data=grid_results,ax = axes[1,1],palette=palette)
    axes[1,1].axhline(grid_results[measdrs].iloc[0],color="gray")
    axes[1,1].set_xlabel("$\mathbf{\Theta}$",fontsize=14)

    axes[0,0].set_ylabel("")
    axes[0,1].set_ylabel("")
    axes[1,0].set_ylabel("")
    axes[1,1].set_ylabel("")

    fig.text(0.012, 0.5, r"RMS [m/s]", va='center', ha='center', rotation='vertical', fontsize=15)
    plt.tight_layout(pad=1.5)
    plt.savefig(resdir+f"Parameter_dependence.pdf")
except:
    pass


# In[ ]:


if preprocess:
    print("\npreprocessing:", ["done","errors"][preprocess_command_run])
if generate_grid:
    print("\ngenerate_grid:", ["done","errors"][generate_grid_command_run])
if run_on_grid:
    print("\nlsd run:",["done","errors"][command_run])

print(f"\nDone. Saved results in {resdir}")


# In[ ]:





# In[ ]:




