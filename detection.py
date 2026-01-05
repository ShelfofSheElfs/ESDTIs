"""
Time to beat: 0.72 min for 994 frames (about 0.225s per frame)
Accuracy to beat: 96.15% of all frames properly marked

Small Data Scientist Interruption:
    - This is trained on an incredibly insignificant number of frames
    - Selections from about 3 days of observations is 1% of 1/6 of CSS 
      observations and therefore insignificant on the whole
    - This will be amended with more data and when reviewed 
      alongside other simulation data
"""

import os
import time
import numpy as np
from astropy.io import fits
from scipy.special import kl_div
from matplotlib import pyplot as plt

# Logging Function
def logs(score):
    global tot_fcount
    global c
    global pw
    global start
    global fails
    global false_positives
    global false_negatives
    global passes
    global t_passes
    pw+=1
    os.system("clear")
    if score == False:
        print("Process Watchdog:", pw, f"~{round((pw/(7+tot_fcount))*100, 2)}%", "\nReq. Files:", tot_fcount-c)
        print("Total Runtime:", round((time.time()-start)/60, 2), "m")
    else:
        print(f"Total Passes   : {passes}")
        print(f"Total Failures : {fails}")
        print(f"Passed Transits: {t_passes}")
        print(f"Failed Transits: {false_negatives}")
        print(f"False Positives: {false_positives}")
        print(f"Accuracy Rate  : {passes/(passes+fails)}")
        print(f"Progress       : {(passes+fails)/tot_fcount}\n")
        print("Total Runtime   :", round((time.time()-start)/60, 2), "m")


def find_cache(fil):
    if os.path.exists(f"venv/{fil}cache.npy"):
        return np.load(f"venv/{fil}cache.npy")
    else:
        return False

# Reference/Training Frames
def get_bases():
    global c
    global tot_fcount
    transits = []
    supports = []

    try:
        if find_cache("transit") == False:
            for transit in os.listdir("FITS/Transit"):
                transits.append(fits.getdata(f"FITS/Transit/{transit}").astype("<i4"))
                c+=1; logs()
            avgt = np.sum(transits, axis=0)/len(transits)
            np.save("venv/avgtcache.npy", (np.sum(transits, axis=0)/len(transits)))
            np.save("venv/transitcache.npy", transits)
            logs(False)
            print("AVGt CACHE SAVED")
        else:
            transits = find_cache("transit")
            avgt = np.load("venv/avgtcache.npy")
            logs(False)
            print("AVGt CACHE PULLED")
    except ValueError:
        transits = np.load("venv/transitcache.npy")
        avgt = np.load("venv/avgtcache.npy")
        logs(False)
        print("AVGt CACHE PULLED")

    try:
        if find_cache("base") == False:
            for support in os.listdir("FITS/Baseline"):
                supports.append(fits.getdata(f"FITS/Baseline/{support}").astype("<i4"))
                c+=1; logs(False)
            avgs = np.sum(supports, axis=0)/len(supports)
            np.save("venv/avgscache.npy", (np.sum(supports, axis=0)/len(supports)))
            np.save("venv/basecache.npy", supports)
            logs(False)
            print("AVGs CACHE SAVED")
        else:
            supports = np.load("venv/basecache.npy")
            avgs = np.load("venv/avgscache.npy")
            logs(False)
            print("AVGs CACHE PULLED")
    except ValueError:
        supports = np.load("venv/basecache.npy")
        avgs = np.load("venv/avgscache.npy")
        logs(False)
        print("AVGs CACHE PULLED")

    return avgt, avgs

# Machine Training Algorithm
def train_data(tbase, bbase):
    # Get Global Vars
    global tot_fcount
    global start
    global fails
    global false_positives
    global false_negatives
    global passes
    global t_passes

    num = 0
    tbase = (tbase-np.min(tbase))/(np.max(tbase)-np.min(tbase))
    bbase = (bbase-np.min(bbase))/(np.max(bbase)-np.min(bbase))

    

    # Training Data
    for trainer in os.listdir("FITS/Training"):
        data = fits.getdata(f"FITS/Training/{trainer}").astype("<i4")
        
        data = (data-np.min(data))/(np.max(data)-np.min(data))

        # By KL Divergence
        transit_kld = kl_div(data, tbase)
        transit_kld = transit_kld[transit_kld != float('inf')]
        baseline_kld = kl_div(data, bbase)
        baseline_kld = baseline_kld[baseline_kld != float('inf')]

        values, bins, _ = plt.hist(transit_kld)
        tarea = np.sum(np.diff(bins)*values)

        values, bins, _ = plt.hist(baseline_kld)
        barea = np.sum(np.diff(bins)*values)

        if barea > tarea and trainer[0]=="T":
            t_passes+=1
            passes+=1
            logs(True)
        elif barea > tarea and trainer[0]!="T":
            false_positives+=1
            fails+=1
            logs(True)
        elif tarea > barea and trainer[0]=="T":
            false_negatives+=1
            fails+=1
            logs(True)
        else:
            passes+=1
            logs(True)

# Timer/Progress Track
start = time.time()
c=0
pw=0
fails = 0
false_positives = 0
false_negatives = 0
passes = 0
t_passes = 0

# Data Counts
train_fcount = len([name for name in os.listdir('FITS/Training')])
tot_fcount = train_fcount

avgt, avgs = get_bases()
train_data(avgt, avgs)
