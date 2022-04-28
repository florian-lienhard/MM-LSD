from astropy.io import fits

import glob
import numpy as np
from scipy.interpolate import LSQUnivariateSpline
from scipy.signal import find_peaks

from pandas import read_pickle


def find_corresponding_s1d_and_ccf(e2dsfilename, maindir):
    date = e2dsfilename
    start = date.find("201")
    end = start + 10
    date_extended = date[start : end + 9]
    date = date[start:end]

    s1d_files = glob.glob(maindir + "/*" + date_extended + "*s1d_A.fits")
    ccf_files = glob.glob(maindir + "/*" + date_extended + "*ccf*A.fits")

    return s1d_files[0], ccf_files[0]

def jj(o, d, i):
    return o * (d + 1) + i


def upper_envelope2(x,y):
    #used to find the envelope of the uncertainties. See Lienhard et al. 2022: envelope weights.

    #run1 
    peaks = find_peaks(y,height=0.,distance = len(x)//100)[0]
    # t= knot positions
    spl = LSQUnivariateSpline(x=x[peaks], y=y[peaks], t=x[peaks][5::10][:-1])
    
    #run2
    peaks2 = peaks[y[peaks]/spl(x[peaks])<1.05]
    spl = LSQUnivariateSpline(x=x[peaks2], y=y[peaks2], t=x[peaks2][5::10][:-1])

    spres = spl(x)
    spres[spres<y]=y[spres<y]

    return spres


def open_pickle(filename):   
    if filename.split('.')[-1]=='p':
        a = read_pickle(filename)
        return a
    elif filename.split('.')[-1]=='fits':
        data = fits.getdata(filename)
        header = fits.getheader(filename)
        return data, header

def find_closest(x1,val):
    difference = np.abs(x1-val)
    return np.where(difference==np.min(difference))[0]

