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
# import matplotlib.ticker as ticker
from numpy.polynomial import Polynomial
import sys
sys.path.append('C:/Google drive/Thermoelectricity/python figures')
import TEproperties_vs_SPBmodel as te
from matplotlib import container

# %% global parameters

class MyLocator(matplotlib.ticker.AutoMinorLocator):
    def __init__(self, n=2): #number of minor ticks here!
        super().__init__(n=n)
matplotlib.ticker.AutoMinorLocator = MyLocator

plt.rcParams['font.size']=14 #similar font size for all
plt.rcParams['axes.linewidth'] = 1.0 #set the value of frame width globally

# %% load data

filename = r'DATA.xlsx'
df = pd.read_excel(filename, sheet_name='TE')
ref = pd.read_excel(filename, sheet_name='refs')

T = df['T'].dropna()
sigma = ['s_CP','s_CP2','s_CP4','s_CP6']
alpha = ['a_CP','a_CP2','a_CP4','a_CP6']
kappa = ['k_CP','k_CP2','k_CP4','k_CP6']

s = {}
a = {}
k = {}

for i in range(len(sigma)):
    s[i] = df[sigma[i]].dropna()
    a[i] = df[alpha[i]].dropna()
    k[i] = df[kappa[i]].dropna()

T_ref = ['T_Tang (2018a)','T_Liu (2018)']
sigma_ref = ['sigma_Tang (2018a)','sigma_Liu (2018)']
alpha_ref = ['alpha_Tang (2018a)','alpha_Liu (2018)']
kappa_ref = ['kappa_Tang (2018a)','kappa_Liu (2018)']

T_refs = {}
a_refs = {}
s_refs = {}
k_refs = {}

for i in range(len(T_ref)):
    T_refs[i] = ref[T_ref[i]].dropna()
    a_refs[i] = ref[alpha_ref[i]].dropna()
    s_refs[i] = ref[sigma_ref[i]].dropna()
    k_refs[i] = ref[kappa_ref[i]].dropna()

# %% parameters of graphs & global visualization parameters

label = ['$x$ = 0','$x$ = 0.02','$x$ = 0.04','$x$ = 0.06']
lrefs = ['Tang et al.','Liu et al.']
marker = ['s','o','^','v','D']
markerr = ['*','P']
colors = ['#fec88c','#f0605d','#9e2e7e','#440f75','#000003']

ms = 10 #marker size
mw = 2 #markeredge width
ls ='-' # linestyle is how your points are connected
cs = 3 #capsize of errorbars
lw = 2 #linewidth
ee = 2 #show every ... of errorbar
lsr = [':','--']

s_err = 0.08
a_err = 0.10
k_err = 0.06
zt_err = 0.2

fig = plt.figure(1, figsize=(4,12), linewidth=5.0)
gs = gridspec.GridSpec(4,1)
gs.update(wspace=0.3, hspace=0.025)

# %% plotting data for the first graph

xtr_subsplot= fig.add_subplot(gs[0:1,0:1])
xtr_subsplot.set_yscale('log')
# xtr_subsplot.yaxis.set_major_formatter(ticker.FuncFormatter(myLogFormat))

plt.plot([], [], ' ', label="Sc$_{x}$O$_{1-x}$")

for i in range(len(T_ref)):
    coefs = Polynomial.fit(T_refs[i], s_refs[i],3).convert().coef
    interp_T = np.linspace(T_refs[i][0], T_refs[i][len(s_refs[i])-1], num=100)
    interp_s_refs = Polynomial(coefs)(interp_T)
    # plt.plot(interp_T, interp_s_refs, linestyle=lsr[i], color='#515151', linewidth = lw,label=lrefs[i], zorder=0)
    plt.plot(T_refs[i], s_refs[i], linestyle=' ', marker=markerr[i], label=lrefs[i], 
                  color='#515151', markersize=ms, markeredgewidth=mw, linewidth = lw, mfc='w', zorder=0)

for i in range(len(sigma)):
    coefs = Polynomial.fit(T, s[i],4).convert().coef
    interp_T = np.linspace(T[0], T[len(s[i])-1], num=100)
    interp_s_ts = Polynomial(coefs)(interp_T)
    z_order = 0+i
    plt.plot(interp_T, interp_s_ts, linestyle=ls, color=colors[i], linewidth = lw, zorder=z_order)
    plt.errorbar(T, s[i], yerr=s[i]*s_err, capsize=cs, linestyle=' ', marker=marker[i], 
                  label=label[i], errorevery = ee, zorder=z_order,
                   color=colors[i], markersize=ms, markeredgewidth=mw, linewidth = lw, mfc='w')

# %% limits for the first graph

y_min = 1e-4
y_max = 1000
x_min = 223
x_max = 1273

plt.xlim([x_min,x_max])
plt.ylim([y_min,y_max])

plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3.5, bottom=True, top=True, left=True, right=True)
plt.tick_params(direction='in',which='major', length=5, bottom=True, top=True, left=True, right=True)
yticks = [1e-3,1e-1,1e1,1e3]
xticks = np.arange(300, 1300, 200)
plt.xticks(xticks)
plt.yticks(yticks)
plt.tick_params(labelbottom=False, labeltop=False, labelright=False, labelleft=True, labelsize=13)
plt.ylabel(r'$\sigma$ ($\Omega^{-1}$ cm$^{-1}$)')
plt.text(x_min+(x_max-x_min)*0.05, y_min+(y_max-y_min)*0.5e-7, '(a)', ha='left', va='bottom')

# handles, labels = xtr_subsplot.get_legend_handles_labels() # get handles
# order = [0,1,2]

# plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], 
#             numpoints=1, labelspacing=0.3, columnspacing=0.1, loc='best',
#             ncol=1, fontsize=10)

handles, labels = xtr_subsplot.get_legend_handles_labels() # get handles
handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]

order = [0,3,4,5,6]

from matplotlib.legend_handler import HandlerLine2D

def updateline(handle, orig):
    handle.update_from(orig)
    handle.set_markersize(7)

leg1 = plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], 
            numpoints=1, labelspacing=0.3, columnspacing=0.2, loc='best',
            ncol=1, fontsize = 9.5, handletextpad=0.1,
            handler_map={plt.Line2D : HandlerLine2D(update_func = updateline)})

order = [1,2]

leg2 = plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], 
            numpoints=1, labelspacing=0.3, columnspacing=0.2, loc='lower center',
            ncol=1, fontsize = 9.5, handletextpad=0.1,
            handler_map={plt.Line2D : HandlerLine2D(update_func = updateline)})

plt.gca().add_artist(leg1)

# %% plotting data for the second graph
xtr_subsplot = fig.add_subplot(gs[1:2,0:1])

for i in range(len(T_ref)):
    coefs = Polynomial.fit(T_refs[i], a_refs[i],3).convert().coef
    interp_T = np.linspace(T_refs[i][0], T_refs[i][len(a_refs[i])-1], num=1000)
    interp_a_refs = Polynomial(coefs)(interp_T)
    plt.plot(interp_T, interp_a_refs, linestyle=lsr[i], color='#515151', linewidth = lw)
    # plt.plot(T_refs[i], a_refs[i], linestyle=' ', marker=markerr[i], label=lrefs[i], 
    #               color='#515151', markersize=ms, markeredgewidth=mw, linewidth = lw, mfc='w')

for i in range(len(alpha)):
    coefs = Polynomial.fit(T, a[i],3).convert().coef
    interp_T = np.linspace(T[0], T[len(a[i])-1], num=1000)
    interp_a_ts = Polynomial(coefs)(interp_T)
    z_order = 0+i
    plt.plot(interp_T, interp_a_ts, linestyle=ls, color=colors[i], linewidth = lw, zorder=z_order)
    plt.errorbar(T, a[i], yerr=a[i]*a_err, capsize=cs, linestyle=' ', marker=marker[i], label=label[i], 
                  color=colors[i], markersize=ms, markeredgewidth=mw,
                  zorder=z_order, linewidth = lw, mfc='w', errorevery=ee)

y_min = -500
y_max = -50

plt.xlim([x_min,x_max])
plt.ylim([y_min,y_max])

plt.minorticks_on() 
plt.tick_params(direction='in',which='minor', length=3.5, bottom=True, top=True, left=True, right=True)
plt.tick_params(direction='in',which='major', length=5, bottom=True, top=True, left=True, right=True)
yticks = np.arange(y_min,y_max, 200)
plt.xticks(xticks)
plt.yticks(yticks)
plt.tick_params(labelbottom=False, labeltop=False, labelright=False, labelleft=True, labelsize=13)
plt.ylabel(r'$\alpha$ ($\mathrm{\mu}$V K$^{-1}$)')
plt.text(x_max-(x_max-x_min)*0.05, y_max-(y_max-y_min)*0.05, '(b)', ha='right', va='top')
plt.xlabel(r'$T$ (K)')

# %% plotting data for the fourth graph
xtr_subsplot = fig.add_subplot(gs[2:3,0:1])

for i in range(len(T_ref)):
    coefs = Polynomial.fit(T_refs[i], k_refs[i],3).convert().coef
    interp_T = np.linspace(T_refs[i][0], T_refs[i][len(k_refs[i])-1], num=1000)
    interp_k_refs = Polynomial(coefs)(interp_T)
    plt.plot(interp_T, interp_k_refs, linestyle=lsr[i], color='#515151', linewidth = lw)
    plt.plot(T_refs[i], k_refs[i], linestyle=' ', marker=markerr[i], label=lrefs[i], 
                  color='#515151', markersize=ms, markeredgewidth=mw, linewidth = lw, mfc='w')

for i in range(len(kappa)):
    coefs = Polynomial.fit(T, k[i],3).convert().coef
    interp_T = np.linspace(T[0], T[len(k[i])-1], num=1000)
    interp_k_ts = Polynomial(coefs)(interp_T)
    z_order = 0+i
    plt.plot(interp_T, interp_k_ts, linestyle=ls, color=colors[i], linewidth = lw, zorder=z_order)
    plt.errorbar(T, k[i], yerr=k[i]*k_err, capsize=cs, errorevery=ee, linestyle=' ', 
                 marker=marker[i], label=label[i], color=colors[i], markersize=ms, 
                 markeredgewidth=mw, linewidth = lw, mfc='w', zorder=z_order)

y_min = 0
y_max = 45

plt.xlim([x_min,x_max])
plt.ylim([y_min,y_max])

plt.minorticks_on() # minor: THEIR AMOUNT CAN BE SET MANUALLY AT IMPORT CODE SECTION
plt.tick_params(direction='in',which='minor', length=3.5, bottom=True, top=True, left=True, right=True)
plt.tick_params(direction='in',which='major', length=5, bottom=True, top=True, left=True, right=True)
plt.xticks(xticks)
yticks = np.arange(y_min,y_max,10)
plt.yticks(yticks)
# labels and where it should be written
plt.tick_params(labelbottom=False, labeltop=False, labelright=False, labelleft=True, labelsize=13)
plt.ylabel(r'$\kappa_{tot}$ (W m$^{-1}$ K$^{-1}$)')
plt.text(x_min+(x_max-x_min)*0.05, y_min+(y_max-y_min)*0.05, '(c)', ha='left', va='bottom')

# %% plotting data for the fourth graph
xtr_subsplot = fig.add_subplot(gs[3:4,0:1])

for i in range(len(T_ref)):
    coefs = Polynomial.fit(T_refs[i], te.zT(T_refs[i],s_refs[i],a_refs[i],k_refs[i]),2).convert().coef
    interp_T = np.linspace(T_refs[i][0], T_refs[i][len(te.zT(T_refs[i],s_refs[i],a_refs[i],k_refs[i]))-1], num=1000)
    interp_zT_refs = Polynomial(coefs)(interp_T)
    plt.plot(interp_T, interp_zT_refs, linestyle=lsr[i], color='#515151', linewidth = lw)
    # plt.plot(T_refs[i], te.zT(T_refs[i],s_refs[i],a_refs[i],k_refs[i]), linestyle=' ', marker=marker[i], label=lrefs[i], 
    #               color='#515151', markersize=ms, markeredgewidth=mw, linewidth = lw, mfc='w')

for i in range(len(kappa)):
    coefs = Polynomial.fit(T, te.zT(T,s[i],a[i],k[i]),4).convert().coef
    interp_T = np.linspace(T[0], T[len(te.zT(T,s[i],a[i],k[i]))-1], num=1000)
    interp_zT_ts = Polynomial(coefs)(interp_T)
    z_order = 0+i
    plt.plot(interp_T, interp_zT_ts, linestyle=ls, color=colors[i], linewidth = lw, zorder=z_order)
    plt.errorbar(T, a[i]**2*s[i]*T/k[i]*1e-10, yerr=a[i]**2*s[i]*T/k[i]*1e-10*zt_err, zorder=z_order,
                 linestyle=' ', marker=marker[i], label=label[i], capsize=cs, errorevery=ee,
                  color=colors[i], markersize=ms, markeredgewidth=mw, linewidth = lw, mfc='w')

y_min = -0.025
y_max = 1.3

plt.xlim([x_min,x_max])
plt.ylim([y_min,y_max])

plt.minorticks_on() # minor: THEIR AMOUNT CAN BE SET MANUALLY AT IMPORT CODE SECTION
plt.tick_params(direction='in',which='minor', length=3.5, bottom=True, top=True, left=True, right=True)
plt.tick_params(direction='in',which='major', length=5, bottom=True, top=True, left=True, right=True)
plt.xticks(xticks)
yticks = np.arange(0,y_max,0.5)
plt.yticks(yticks)
# labels and where it should be written
plt.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True, labelsize=13)
plt.ylabel(r'$zT$')
plt.xlabel(r'$T$ (K)')
plt.text(x_min+(x_max-x_min)*0.05, y_max-(y_max-y_min)*0.05, '(d)', ha='left', va='top')

# %% saving picture

plt.savefig('Fig.3.pdf', dpi=300, bbox_inches="tight")
