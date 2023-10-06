# -*- coding: utf-8 -*-
"""
Python seminar. Friday, October 6, 2023. Andrei Novitskii
"""

# %% import

import pandas as pd
# import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
# import seaborn as sns
import matplotlib.ticker
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

# %% global parameters

class MyLocator(matplotlib.ticker.AutoMinorLocator):
    def __init__(self, n=2): #number of minor ticks here!
        super().__init__(n=n)
matplotlib.ticker.AutoMinorLocator = MyLocator

plt.rcParams['font.size'] = 14 #similar font size for all
plt.rcParams['axes.linewidth'] = 1.0 #set the value of frame width globally

# %% load data
#read data from excel sheet where each value is stored in a different sheet
# path = r'ZEM/'
filename = r'DATA.xlsx'
# df = pd.read_excel(path+filename)
df = pd.read_excel(filename, sheet_name='DLS')

s_sample1 = (df['Sample1_size']/1000).dropna() #nm to mkm
i_sample1 = df['Sample1_intensity'].dropna()
s_sample2 = (df['Sample2_size']/1000).dropna()
i_sample2 = df['Sample2_intensity'].dropna()
s_sample3 = (df['Sample3_size']/1000).dropna()
i_sample3 = df['Sample3_intensity'].dropna()

#fitting function and some parameters
def logNormal(x, y0, A, sigma, mu):
    return y0+A/(np.sqrt(2*np.pi)*sigma*x)*np.exp(-(np.log(x)-np.log(mu))**2/(2*sigma**2)) # eq1 from 10.1007/s12598-017-0926-5
initialParameters = np.array([1.0, 1.0, 1.0, 1.0]) # initial parameter values - must be within bounds
lowerBounds = (0, 0, 0, 0) # bounds on parameters - initial parameters must be within these
upperBounds = (np.inf, np.inf, np.inf, np.inf)
parameterBounds = [lowerBounds, upperBounds]

fittedParameters1, pcov = curve_fit(logNormal, s_sample1, i_sample1, initialParameters, bounds = parameterBounds)
fittedParameters2, pcov = curve_fit(logNormal, s_sample2, i_sample2, initialParameters, bounds = parameterBounds)
fittedParameters3, pcov = curve_fit(logNormal, s_sample3, i_sample3, initialParameters, bounds = parameterBounds)

y01, A1, sigma1, mu1 = fittedParameters1
y02, A2, sigma2, mu2 = fittedParameters2
y03, A3, sigma3, mu3 = fittedParameters3

xPlotData1 = np.linspace(0, 1, 100)
xPlotData2 = np.linspace(0.2, 2, 100)
xPlotData3 = np.linspace(0.5, 2.5, 100)

y_plot1 = logNormal(xPlotData1, y01, A1, sigma1, mu1)
y_plot2 = logNormal(xPlotData2, y02, A2, sigma2, mu2)
y_plot3 = logNormal(xPlotData3, y03, A3, sigma3, mu3)

# %% plotting data for the phase ticks

# colors from seaborn palette 'rocket', number is the requaered colors
# colors=sns.color_palette("rocket",5)
# or you can use your own palette 
colors = ['#b2df8a', '#1f78b4', '#fc8d59']

fig = plt.figure(1, figsize=(5, 5), linewidth=5.0)
gs = gridspec.GridSpec(10,1)
gs.update(wspace=0.2, hspace=0.2)

x_min = 0.1
x_max = 5

# %% plotting data for the first graph

xtr_subsplot = fig.add_subplot(gs[0:10,0:1])
xtr_subsplot.set_xscale('log')
xtr_subsplot.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:g}'.format(x)))

w = 0.145 #width of bars

plt.bar(s_sample3, i_sample3, width=w*s_sample3, align='center', color=colors[2], zorder = 2, alpha = 0.75, label='USP', edgecolor='k')
# plt.plot(xPlotData3, y_plot3, color='k', linewidth=1,zorder = 2) #you can compare how Gauss fit the data (obviously wrong)
plt.plot(xPlotData3, y_plot3, color='k', linewidth=1,zorder = 2)
plt.bar(s_sample2, i_sample2, width=w*s_sample2, align='center', color=colors[1], zorder = 2, alpha = 0.75, label='CP', edgecolor='k')
# plt.plot(s_sample2, i_sample2, color=colors[1], zorder = 1)
# plt.plot(xPlotData2, y_plot2, color='k', linewidth=1,zorder = 2)
plt.plot(xPlotData2, y_plot2, color='k', linewidth=1,zorder = 2)
plt.bar(s_sample1, i_sample1, width=w*s_sample1, align='center', color=colors[0], zorder = 2, alpha = 0.75, label='CS', edgecolor='k')
# plt.plot(s_sample1, i_sample1, color=colors[0], zorder = 1)
#plt.plot(xPlotData1, y_plot1, color='k', linewidth=1,zorder = 2)
plt.plot(xPlotData1, y_plot1, color='k', linestyle='-', linewidth=1,zorder = 2)

plt.legend(labelspacing = 0.25)

res1 = "$\mu_{\mathrm{CS}}$ = %.3f $\mathrm{\mu}$m" % (mu1)
plt.text(0.125, 20, res1)
res2 = "$\mu_{\mathrm{CP}}$ = %.3f $\mathrm{\mu}$m" % (mu2)
plt.text(0.125, 22.5, res2)
res3 = "$\mu_{\mathrm{USP}}$ = %.3f $\mathrm{\mu}$m" % (mu3)
plt.text(0.125, 25, res3)

# %% limits for the first graph
y_min = 0
y_max = 30

plt.xlim([x_min,x_max])
plt.ylim([y_min,y_max])

# %% ticks for the first graph
plt.minorticks_on() # minor: THEIR AMOUNT CAN BE SET MANUALLY AT IMPORT CODE SECTION
# ticks parameters
plt.tick_params(direction='in',which='minor', length=3.5, bottom=True, top=True, left=True, right=False, zorder=2)
plt.tick_params(direction='in',which='major', length=5, bottom=True, top=True, left=True, right=False, zorder=2)
# ticks step on each axis
xticks = [0.1, 0.5, 1, 2.5, 5]
yticks = np.arange(5,y_max+1,5)
plt.xticks(xticks)
plt.yticks(yticks)
# labels and where it should be written
plt.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True)
plt.ylabel(r'Number of particles (%)')
plt.xlabel(r'Size ($\mathrm{\mu}$m)')

# %% saving figure

plt.savefig('Fig.4.pdf', dpi=300, bbox_inches="tight")
