# Imports
import os
import time
import math
import itertools
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip as sigc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Seaborn (sns)
sns.set_style("dark")

# Function Definitions

def logs():
    global tot_fcount
    global c
    global pw
    global start
    pw+=1
    os.system("clear")
    print("Process Watchdog:", pw, f"~{round((pw/(7+tot_fcount))*100, 2)}%", "\nFiles:", c)
    print("Total Runtime:", round((time.time()-start)/60, 2), "m")

# Timer/Progress Track
start = time.time()
c=0
pw=0

# Data Counts
transit_fcount = len([name for name in os.listdir('FITS/Transit/')])
baseline_fcount = len([name for name in os.listdir('FITS/Baseline/')])
unlabeled_fcount = len([name for name in os.listdir('FITS/Unlabeled/')])
tot_fcount = transit_fcount+baseline_fcount+unlabeled_fcount

# Transit Data Import
transits = []
supports = []
#for transit in os.listdir("FITS/Transit"):
#    transits.append(fits.getdata(f"FITS/Transit/{transit}").astype("<i4"))
#    c+=1; logs()

for support in os.listdir("FITS/Baseline"):
    supports.append(fits.getdata(f"FITS/Baseline/{support}").astype("<i4"))
    c+=1; logs()

# Averages
avgt = np.sum(transits, axis=0)/len(transits)
avgt = (avgt-np.min(avgt))/(np.max(avgt)-np.min(avgt))
avgs = np.sum(supports, axis=0)/len(supports)
avgs = (avgs-np.min(avgs))/(np.max(avgs)-np.min(avgs))
logs()

# Plot Setup
fig, axes = plt.subplots(1, 2, figsize=(10,7), dpi= 80)
axes[0].set_title("Histograms"); axes[1].set_title("KDEs") # Titles
axes[0].xaxis.set_major_formatter(ticker.EngFormatter()) # Value Clipping
axes[1].xaxis.set_major_formatter(ticker.EngFormatter()) # Value Clipping Pt. 2
axes[0].set_xlabel("FITS Brightness Value"); axes[1].set_xlabel("FITS Brightness Value") # X Labels
axes[0].set_ylabel("Count"); axes[1].set_ylabel("Count") # Y Labels
axes[0].set_yscale("symlog"); axes[1].set_yscale("symlog") # Scaling
logs()

# Histograms
#ht = sns.histplot(sigc(avgt.flatten(), sigma_lower=4, sigma_upper=999), color="orange", label="SAT", ax=axes[0], ); logs()
ha = sns.histplot(sigc(avgs.flatten(), sigma_lower=4, sigma_upper=999), color="dodgerblue", label="AVG", ax=axes[0]); logs()

# KDEs
#kdet = sns.kdeplot(sigc(avgt.flatten(), sigma_lower=4, sigma_upper=999), color="orange", bw_adjust=0.1, label="SAT KDE", ax=axes[1]); logs()
kdeh = sns.kdeplot(sigc(avgs.flatten(), sigma_lower=4, sigma_upper=999), color="dodgerblue", bw_adjust=0.1, label="AVG KDE", ax=axes[1]); logs()

plt.figlegend(loc="upper right")
logs()
plt.show()
