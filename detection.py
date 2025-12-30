import os
import time
import math
import itertools
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip as sigc
from scipy.stats import t
from scipy.special import kl_div
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Logging Function
def logs():
    global tot_fcount
    global c
    global pw
    global start
    pw+=1
    os.system("clear")
    print("Process Watchdog:", pw, f"~{round((pw/(7+tot_fcount))*100, 2)}%", "\nFiles:", c)
    print("Total Runtime:", round((time.time()-start)/60, 2), "m")

# Reference/Training Frames
def get_bases():
    global c
    transits = []
    supports = []
    for transit in os.listdir("FITS/Transit"):
        transits.append(fits.getdata(f"FITS/Transit/{transit}").astype("<i4"))
        c+=1; logs()

    bases = 0
    for support in os.listdir("FITS/Baseline"):
        supports.append(fits.getdata(f"FITS/Baseline/{support}").astype("<i4"))
        c+=1; logs()

    # Averages
    avgt = np.sum(transits, axis=0)/len(transits)
    logs()
    print("AVGt DONE")
    avgs = np.sum(supports, axis=0)/len(supports)
    logs()
    print("AVGs DONE")

    return avgt, avgs

# Machine Training Algorithm
def train_data(tbase, bbase):
    # Get Global Vars
    global tot_fcount

    num = 0

    # Training Data
    for trainer in os.listdir("FITS/Training"):
        data = fits.getdata(f"FITS/Training/{trainer}").astype("<i4")
        
        # By KL Divergence
        transit_kld = kl_div(data[data != 0], tbase[tbase != 0])
        baseline_kld = kl_div(data[data != 0], bbase[bbase != 0])
        transit_kld[transit_kld > 1e308] = 0
        baseline_kld[baseline_kld > 1e308] = 0

        '''
        NEXT: WRITE CODE TO DETECT SATELLITE TRANSIT FROM KL_DIV DATA
        '''

# Timer/Progress Track
start = time.time()
c=0
pw=0

# Data Counts
transit_fcount = len([name for name in os.listdir('FITS/Transit/')])
baseline_fcount = len([name for name in os.listdir('FITS/Baseline/')])
unlabeled_fcount = len([name for name in os.listdir('FITS/Unlabeled/')])
train_fcount = len([name for name in os.listdir('FITS/Training')])
tot_fcount = transit_fcount+baseline_fcount+unlabeled_fcount+train_fcount

avgt, avgs = get_bases()
train_data(avgt, avgs)

