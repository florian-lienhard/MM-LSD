import os

star = "Sun"

#checklist for preprocessing:
#star name defined above
#rassine installed in rassine_loc
#s2d spectra, ccf files, s1d files in rawdir


#checklist for next steps:
#name vald file e.g. yourtarget.txt and put it into VALD_files folder
#star
#the steps after preprocessing are independent of the data format


from datetime import date
today = date.today()
#indic = str(today.year)+str(today.month)+str(today.day)
indic = "1"

# -------------------------------------------------------------
#           Settings for MM-LSD run (what should be done in this run?)
# -------------------------------------------------------------

#preprocess data (if not done already)
preprocess = True
#generate grid on which to evaluate LSD, delete old rv, rv_err pickle files
generate_grid = True
#run on this parameter combination grid
run_on_grid = True


# -------------------------------------------------------------
#           Folder locations
# -------------------------------------------------------------

#home
rt = "/home/fl386"
#folder where the input.py file is located. 
stardir = './stars/'+star+'/'
#results of MM-LSD will be saved here
resdir = stardir+f"results_{indic}/"



#location of RASSINE installation
rassine_loc = rt + "/Rassine_public-master"

#location of data
datadir = rt + "/MM-LSD-dev/data/"

#your targets data
maindir = datadir + star
#fits files in here
rawdir = maindir +"/data/"
#this will be produced, contains output from preprocess.py
dirdir = maindir+"/processed/"
#preprocess.py will save rassine output in this file. not needed anymore after initial run.
rassine_res = maindir + "/rassine_res/"

#make sure spectrum_name is set to cwd+'/spectra_library/stars/spec_toreduce.csv' in Rassine_config.py.

if not os.path.exists(resdir):
    os.mkdir(resdir)

#rvs and associated uncertainties from MM-LSD will be saved in these pickle files
rvresfile = resdir+"lsd_rv_"+star+f"_{indic}.pkl"
rverrresfile = resdir+"lsd_rv_err_"+star+f"_{indic}.pkl"  
commonprofilefile = resdir+"common_profile_"+star+f"_{indic}.pkl"  

# -------------------------------------------------------------
#           Data 
# -------------------------------------------------------------


#name of pipeline (e.g. HARPS-N new DRS (Dumusque et al, 2021)

#current version of "new drs" (April 2022)
pipname = "DRS_2.3.5"

#"old drs". older version has higher number
#pipname = "DRS_3.7"


# -------------------------------------------------------------
#           Preprocessing
# -------------------------------------------------------------

#only needed if you haven't already preprocessed your data (i.e. you have run this at least once)


#extract info from fits headers
extract_info = True
run_rassine = True
save_spectra_as_pkl = True
overlap_correction = True

#in case you want to reduce only the first x spectra, set number_of_spectra_to_reduce to x. Otherwise, set it to a number >= number of spectra.
number_of_spectra_to_reduce = 100000




#name of pipeline
if pipname == "DRS_3.7":
    pipeline_keyword = "DRS"
    sid ="TNG"    
if pipname == "DRS_2.3.5":
    #espresso pipeline
    pipeline_keyword = "QC"
    sid ="TNG"    
if pipname == "ESPRESSO":
    sid="ESO"


#velocity step for common profile approx velocity step between two adjacent pixels. Must be constant (assumption in cvmt function).
if pipname == "DRS_2.3.5" or pipname == "DRS_3.7":
    vStep = 0.82
else:
    vStep = 0.7


#------------------------------------------------------------------------------
#RV-EXTRACTION

#set to 1 to correct order overlap discrepancy. no major influence on RVs expected.
rassoption = 1

#set to 0 for flux weights
#set to 1 to use (almost) flat weights per absorption line. (that's inverse squared of the upper envelope of the error)
#set to 2 for uniform weights per order
erroption = 0

#set to 1 for removing all barycentric wavelengths ever affected by telluric lines. recommended.
telloption = 1

grid = {}

#run code on max max_nr_of_specs spectra
grid["max_nr_of_specs"]= [10000]
#velocity grid width parameter (FWHM * velgridwidth)
#dvel = np.round(vel_hwhm)*velgridwidth
#vel = np.arange(systemrv-dvel, systemrv+dvel, vStep)
grid["velgridwidth"] = [2.5,3.0,3.5,4.0]
#remove data affected by tellurics deeper than telluric_cut (0.1 = depth of telluric line = 90% transmission)
grid["telluric_cut"] = [0.2,0.1]
#minimal depth of included vald3 absorption lines
grid["mindepthparam"] = [0.1]
#maximal depth of included vald3 absorption lines
grid["maxdepthparam"] = [0.8,1.0]
#if absolute difference between spectrum and first convolution model greater than this: mask.
grid["modelspecdeviationcut"] = [0.5,1.0]
#exclude wide lines? 0 = no, 1 = yes. see run_lsd.
grid["exclwidelinesparam"] = [0]

grid["telloption"] = [telloption]
grid["erroption"] = [erroption]
grid["rassoption"] = [rassoption]

#mean-combine use_n_time_series many of the produced time series
#i.e. e.g. use the 16 timeseries with the lowest RMS (out of the 32 time series from the 32 parameter combinations)
use_n_time_series = 16




#what do you define as an outlier?
#remove rv if difference between median and rv is greater than this value
delta_rv_outlier = 200


#weighting scheme. letting the weights of the orders vary as defined in code is recommended.
weight_schemes = ["flux weight_can_vary"]
#alternatively:
#weight_schemes = ["flux weight_fixed_throughout_time_series"]

# -------------------------------------------------------------
#           Don't change these
# -------------------------------------------------------------



#compute nr of grid combinations
nr_of_combinations = 1
for key in grid.keys():
    nr_of_combinations*=len(grid[key])

assert use_n_time_series <= nr_of_combinations, "set use_n_time_series to value smaller than nr_of_combinations"


#remove data point if flux < excllower
#note that the spectra are between -1 and 0 (i.e. normalise to 1, then subtract 1)
excllower = -1.1
exclupper = 0.05
usetapas = True


#CONSTANTS
c = 299792.458
#------------------------------------------------------------------------------
