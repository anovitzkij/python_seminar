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
from matplotlib.ticker import MultipleLocator

# %% global parameters

class MyLocator(matplotlib.ticker.AutoMinorLocator):
    def __init__(self, n=2): #number of minor ticks here!
        super().__init__(n=n)
matplotlib.ticker.AutoMinorLocator = MyLocator

plt.rcParams['font.size'] = 14 #similar font size for all
plt.rcParams['font.family'] = 'Helvetica' #similar font for all
plt.rcParams['axes.linewidth'] = 1.0 #set the value of frame width globally

def make_patch_spines_invisible(ax): #custom function to turn axes invisible
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

# %% load data
#read data from excel sheet where each value is stored in a different sheet
# path = r'ZEM/'
filename = r'DATA.xlsx'
# df = pd.read_excel(path+filename)
df = pd.read_excel(filename, sheet_name='SPS')

radius = 5 #radius of graphite die in [mm]
cutoff = 3003 #cut the unnecessary data

t = df['Time (s)'].iloc[:cutoff] #time in [s]
V = df['Voltage (V)'].iloc[:cutoff] #voltage in [V]
I = df['Current (A)'].iloc[:cutoff] #current in [A]
Tc = df['Temperature (C)'].iloc[:cutoff] #temperature in [C]
T = Tc + 273.15 #temperature in [K]
N = df['Pressure (kN)'].iloc[:cutoff] #pressure in [kN]
P = N*1000/(np.pi*(radius/1000)**2)/1000000 #pressure in [MPa]
Dm = df['Displace (mm)'].iloc[:cutoff] #displacement in [mm]
D = Dm/max(Dm) #relative displacement
Dr = df['Displace ratio (V/s)'].iloc[:cutoff] #displacement ration in [V/s]
vac = df['Vacuum (Pa)'].iloc[:cutoff] #vaccum level in [Pa]
t_prog = df['Prog. time (s)'].iloc[:cutoff] #programmed time in [s]
T_prog = df['Prog. temp (K)'].iloc[:cutoff] #programmed temperature in [K]

m = df['Mass (g)'].dropna() #mass of the sintered powder in [g]
dens = df['Densification (mm)'].dropna() #overall densification of the sample during sintering in [mm]
density = df['Density (g/cm3)'].dropna() #density of the obtained sample after sintering in [g/cm3]
theor_density = 9.56 #theoretical density for the material in [g/cm3]
rel_density = 100*density/theor_density

# %% general settings

lw = 2
x_min = 0
x_max = 1500

fig = plt.figure(1, figsize=(5,5), linewidth=5.0)
gs = gridspec.GridSpec(6,1)
gs.update(wspace=0.15, hspace=0.2)

# %% plotting data for the pressure

xtr_subsplot = fig.add_subplot(gs[0:1,0:1])
plt.plot(t, P, label='Pressure (MPa)', color='#b177de', linewidth = lw)

y_max = 70
y_min = 30
plt.xlim([x_min,x_max])
plt.ylim([y_min,y_max])
plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3.5, bottom=True, top=True, left=True, right=True)
plt.tick_params(direction='in',which='major', length=5, bottom=True, top=True, left=True, right=True)
xticks = np.arange(x_min,x_max+0.01,250)
yticks = np.arange(y_min,y_max+0.01,20)
plt.xticks(xticks)
plt.yticks(yticks)
plt.tick_params(labelbottom=False, labeltop=False, labelright=False, labelleft=True)
plt.ylabel(r'$P$ (MPa)')
plt.axhline(y=50, linestyle='dotted', color='gray', zorder=1)

# %% plotting temperature
xtr_subsplot = fig.add_subplot(gs[1:6,0:1])
plt.plot(t_prog, T_prog, label='Temperature program', color='k', linewidth = lw - 1, linestyle = '--')
plt.plot(t, T, label='Temperature, $T$', color='#f14040', linewidth = lw)

y_min = 273
y_max = 932

plt.text(x_max*1.2, y_max*1.075, '$D_\Sigma$ = %.2f mm / %d g\n$d$ = %.3f g/cm$^3$\n%.1f %%' % (dens,m,density,rel_density),
         ha='center', va='center', fontsize=10)

plt.xlim([x_min,x_max])
plt.ylim([y_min,y_max])
plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=3.5, bottom=True, top=True, left=True, right=True)
plt.tick_params(direction='in',which='major', length=5, bottom=True, top=True, left=True, right=True)
plt.tick_params(axis='y',which='both', colors='#f14040')
yticks = np.arange(300,y_max,100)
xtr_subsplot.spines['left'].set_color('#f14040')
plt.xticks(xticks)
plt.yticks(yticks)
plt.xlabel(r'$t$ (s)')
plt.ylabel(r'$T$ (K)', color='#f14040')

# %% right axis for displacement

ax2 = xtr_subsplot.twinx()
y2, = ax2.plot(t, D, label='Relative displacement, $D$', color='#1a6fdf', lw = 1.5, ls = ':')
make_patch_spines_invisible(ax2)
ax2.spines["right"].set_visible(True)
y_min=-0.2
y_max=1.1
ax2.set_ylim(y_min,y_max)
ax2.yaxis.set_ticks(np.arange(0,y_max,0.2))
minorLocatory = MultipleLocator(0.1)
ax2.yaxis.set_minor_locator(minorLocatory)
ax2.tick_params(direction='in',which='minor', length=3.5, colors='#1a6fdf')
ax2.tick_params(direction='in',which='major', length=5, colors='#1a6fdf')
ax2.spines['right'].set_color('#1a6fdf')
ax2.set_ylabel(r'$D$ (arb. units)', color='#1a6fdf')

# %% right axis for displacement ratio outside main plot

ax3 = xtr_subsplot.twinx()
ax3.spines["right"].set_position(("axes", 1.2))
make_patch_spines_invisible(ax3)
ax3.spines["right"].set_visible(True)
y3, = ax3.plot(t, Dr, label='Displacement ratio, $D_r$', color='#37ad6b', lw = 1.5)
y_min=-0.35
y_max=-y_min
ax3.set_ylim(y_min,y_max)
ax3.yaxis.set_ticks(np.arange(-0.3,0.31,0.1))
# minorLocatory = MultipleLocator(0.1)
# ax3.yaxis.set_minor_locator(minorLocatory)
ax3.tick_params(direction='in',which='minor', length=3.5, colors='#37ad6b')
ax3.tick_params(direction='in',which='major', length=5, colors='#37ad6b')
ax3.spines['right'].set_color('#37ad6b')
ax3.set_ylabel(r'$D_r$ (V/s)', color='#37ad6b')

#%% one legend for the data from different subplots

hy1, ly1 = xtr_subsplot.get_legend_handles_labels()
hy2, ly2 = ax2.get_legend_handles_labels()
hy3, ly3 = ax3.get_legend_handles_labels()
plt.legend(hy1+hy2+hy3, ly1+ly2+ly3, ncol=1, loc='lower right', fontsize=10) #columnspacing=0.05 labelspacing=0.05 handletextpad=0.05

# %% saving figure

plt.savefig('Fig.2.pdf', dpi=300, bbox_inches="tight")