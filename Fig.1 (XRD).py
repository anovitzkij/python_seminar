# -*- coding: utf-8 -*-
"""
Python seminar. Friday, October 6, 2023. Andrei Novitskii
"""

# %% import

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.ticker

# %% global parameters

class MyLocator(matplotlib.ticker.AutoMinorLocator):
    def __init__(self, n=2): #this one determine the number of minor ticks (n-1)
        super().__init__(n=n)
matplotlib.ticker.AutoMinorLocator = MyLocator

plt.rcParams['font.size'] = 14 #similar font size for all
plt.rcParams['axes.linewidth'] = 1.0 #set the value of frame width globally 

# %% load data
#read data from excel sheet where each value is stored in a different sheet
# path = r'C:/...ZEM/...'
filename = r'DATA.xlsx'
# df = pd.read_excel(path+filename, sheet_name='XRD')
df = pd.read_excel(filename, sheet_name='XRD')

theta = ['t-7032-3', 't-7032-4','t-7032-5','t-7032-6','t-7032-1','t-7032-2']
intensity = ['I-7032-3','I-7032-4','I-7032-5','I-7032-6','I-7032-1','I-7032-2']

th = {}
ints = {}
ints_raw = {}

for i in range(len(theta)):
    th[i] = df[theta[i]].dropna()
    ints_raw[i] = df[intensity[i]].dropna()
    ints[i] = ints_raw[i]/max(ints_raw[i])-min(ints_raw[i]/max(ints_raw[i]))

hkl_SKD = df['CoSb3'].dropna()
hkl_InSb = df['InSb'].dropna()
hkl_Sb = df['Sb'].dropna()

# %% general figure parameters

colors = ['#66c2a5', '#66c2a5', '#fc8d62', '#fc8d62', '#8da0cb', '#8da0cb']
label = ['BMg','BMAg','MS','MSA','BM','BMA']
ms = 6
mw = 1

fig = plt.figure(1, figsize=(5,5), linewidth=5.0)
gs = gridspec.GridSpec(12,12)
gs.update(wspace=0.2, hspace=0.2)

xtr_subsplot= fig.add_subplot(gs[0:1,0:9])
for xc in hkl_SKD.values:
    plt.axvline(x=xc,color='grey')
plt.tick_params(left=False, bottom=False, right=False, top=False)
plt.tick_params(labelleft=False, labelbottom=False)
x_min = 20
x_max = 90
plt.xlim(x_min,x_max)

xtr_subsplot= fig.add_subplot(gs[0:1,9:12])
for xc in hkl_SKD.values:
    plt.axvline(x=xc,color='grey')
plt.tick_params(direction='in',bottom=False, right=True, top=False)
plt.tick_params(labelleft=False, labelbottom=False)
x_min1 = 21
x_max1 = 30
plt.xlim(x_min1,x_max1)

# %% plotting data for the first graph

delta = 1 #distance between pstterns
lw = 2 #line width

xtr_subsplot = fig.add_subplot(gs[1:12,0:9])

for i in range(len(theta)):
    plt.plot(th[i], ints[i]+delta*i, linestyle='-', label=label[i], color=colors[i],
             linewidth = lw)

plt.plot(28.7, delta*1+0.2, "v", label='Sb', color='k', markersize=ms) #symbol for secondary phase
plt.plot(40.1, delta*1+0.45, "v", label='Sb', color='k', markersize=ms)
plt.plot(41.95, delta*1+0.25, "v", label='Sb', color='k', markersize=ms)
plt.plot(23.79, delta*1+0.3, "d", label='InSb', color='k', markersize=ms)
plt.plot(39.33, delta*1+0.25, "d", label='InSb', color='k', markersize=ms)
plt.plot(46.48, delta*1+0.35, "d", label='InSb', color='k', markersize=ms)
plt.plot(23.79, delta*2+0.3, "d", label='InSb', color='k', markersize=ms)
plt.plot(39.33, delta*2+0.25, "d", label='InSb', color='k', markersize=ms)
plt.plot(46.48, delta*2+0.35, "d", label='InSb', color='k', markersize=ms)
plt.plot(23.5, delta*5+0.35, "d", label='InSb', color='k', markersize=ms)

plt.plot(23.5, delta*5+0.35, "d", label='InSb', color='#f14040', markersize=ms, markeredgewidth=mw, mfc='#fbc4c4')

# %% limits for the first graph
y_min = -0.15
y_max = delta*(len(theta)-1)+1.05

plt.xlim([x_min,x_max])
plt.ylim([y_min,y_max])

# %% ticks for the first graph
plt.minorticks_on()
# ticks parameters
plt.tick_params(direction='in',which='minor', length=3.5, bottom=False, top=False, left=False, right=False)
plt.tick_params(direction='in',which='major', length=5, bottom=True, top=True, left=False, right=False)
# ticks step on each axis
xticks = np.arange(x_min,x_max+0.01,20)
plt.xticks(xticks)
# labels and where it should be written
plt.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=False, labelsize=13)
plt.xlabel(r'2$\theta$ (degree)')
plt.ylabel(r'Intensity (arb. units)')

plt.text(x_min+(x_max-x_min)*0.03, y_max-(y_max-y_min)*0.03, '(a)', ha='left', va='top')

lft=1
up=0.25
plt.text(x_max-lft, ints[0][152]+up, 'BMG',va='bottom',ha='right')
plt.text(x_max-lft, ints[1][152]+delta+up, 'BMAG',va='bottom',ha='right')
plt.text(x_max-lft, ints[2][152]+delta*2+up, 'MS',va='bottom',ha='right')
plt.text(x_max-lft, ints[3][152]+delta*3+up, 'MSA',va='bottom',ha='right')
plt.text(x_max-lft, ints[4][152]+delta*4+up, 'BM',va='bottom',ha='right')
plt.text(x_max-lft, ints[5][152]+delta*5+up, 'BMA',va='bottom',ha='right')

#%% enlarged part
xtr_subsplot = fig.add_subplot(gs[1:12,9:12])
magn_coeff=6

for i in range(len(theta)):
    plt.plot(th[i], ints[i]+delta*i/magn_coeff+0.01, linestyle='-', label=label[i], color=colors[i],
             linewidth = lw)

plt.plot(28.69, delta*1/magn_coeff+0.4/magn_coeff, "v", label='Sb', color='k', markersize=ms) #symbol for secondary phase
plt.plot(23.5, delta*1/magn_coeff+0.6/magn_coeff, "d", label='InSb', color='k', markersize=ms)
plt.plot(23.5, delta*2/magn_coeff+0.6/magn_coeff, "d", label='InSb', color='k', markersize=ms)
plt.plot(23.5, delta*5/magn_coeff+0.55/magn_coeff, "d", label='InSb', color='k', markersize=ms)

# %% limits for the first graph
y_min = -0.01
y_max = 1.04
# y_max = delta*(len(theta)-1)+1.05

plt.xlim([x_min1,x_max1])
plt.ylim([y_min,y_max])

# %% ticks for the first graph
# plt.minorticks_on() # THEIR AMOUNT CAN BE SET MANUALLY AT IMPORT CODE SECTION
# ticks parameters
plt.tick_params(direction='in',which='minor', length=3.5, bottom=True, top=True, left=False, right=False)
plt.tick_params(direction='in',which='major', length=5, bottom=True, top=True, left=False, right=False)
# ticks step on each axis
xticks = np.arange(25,x_max1+0.01,5)
plt.xticks(xticks)
# labels and where it should be written
plt.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=False)
plt.xlabel(r'2$\theta$ (degree)')

# plt.text(x_max1+0.5, ints[0][152]/magn_coeff+0.005, 'BM',va='bottom',ha='left')
# plt.text(x_max1+0.5, ints[1][152]/magn_coeff+0.17, 'BMA',va='bottom',ha='left')
# plt.text(x_max1+0.5, ints[2][152]/magn_coeff+0.35, 'MS',va='center',ha='left')
# plt.text(x_max1+0.5, ints[3][152]/magn_coeff+0.52, 'MSA',va='center',ha='left')

plt.text(x_max1-(x_max1-x_min1)*0.03, y_max-(y_max-y_min)*0.03, '(b)', ha='right', va='top')

#plt.axvline(x=23.79, linestyle='dotted', color='black') #, zorder=1

# %% legend (will be placed to the 'best' location by default) for the first graph

# plt.legend(fontsize=14, loc='upper left', bbox_to_anchor=(1.04, 1), ncol=1) 
# plt.legend(ncol=2, columnspacing=0) 

# %% saving picture

plt.savefig('Fig.1.pdf', dpi=300, bbox_inches="tight")
