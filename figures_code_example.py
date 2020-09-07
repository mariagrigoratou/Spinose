# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import string
from matplotlib import colors
import time
from scipy.integrate import odeint
from scipy.interpolate import spline
import pandas as pd
import csv
from matplotlib import style
#style.use('ggplot')
style.use('seaborn')


df = pd.read_csv('"add the directory and the name of the csv file".csv')
df.replace('DEAD', 1e-04, inplace=True)
#df['%foram'].values[df['%foram'].values > 1e+00] = 1e+00
df['%foram'].values[df['%foram'].values < 1e-04] = 1e-04
df['%foram'] = df['%foram'].astype(float)
# setup the plot
#fig, ax = plt.subplots(3,3, figsize=(6,6))
## sharex='col', sharey='row' are used to premoved inner labels on the grid to make the plot cleaner.
x =np.zeros((3,3)) 
fig, x = plt.subplots(3,3, sharex='col', sharey='row', figsize=(12,12))

### adjust the space between the plots
fig.subplots_adjust(hspace=0.2, wspace=0.2)

###### define the colormap   ######
#cmap = plt.cm.YlGnBu_r
cmap = plt.cm.coolwarm
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
## force the first color entry to be grey
#cmaplist[0] = ('w')
# create the new map
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)


### define the data

## 30oC oligotrophic - subplot ax1 ###

# Growth rate for plausible (GrR)
GrR_O30 = df.loc[, 'gf']
#Mortality rate for plausible (MrR)              
MrR_O30 = df.loc[, 'mf']
# % biomass for plausible (BR)
BR_O30 = df.loc[, '%foram']
BR_O30 =  np.log10(BR_O30)

# # Growth rate for low_biomass (GrS)
# GrS_O30 =
# #Mortality rate for low_biomass (MrS)             
# MrS_O30 =
# # % biomass for low_biomass (BS)
# #BS_O30 = [np.array([])]
# BS_O30 = 

# Growth rate for the other senarios (Gr)
Gr_O30 = df.loc[, 'gf']
#Mortality rate for the other senarios(Mr)             
Mr_O30 = df.loc[, 'mf']
# % biomass for the other senarios (B)
B_O30 = df.loc[, '%foram']
B_O30 = np.log10(B_O30)
# make the scatter
plt.subplot(x[0,2])
plausible = plt.scatter(MrR_O30,GrR_O30,c = BR_O30, s = 200, cmap=cmap, marker = "*",vmax= 1.0e+00, vmin = -4.0e+00, edgecolors='black')  
low_biomass = plt.scatter(MrS_O30, GrS_O30, c = BS_O30, s=100, cmap=cmap, marker = 'v', vmax= 1.0e+00, vmin = -4.0e+00,edgecolors='black')
Total = plt.scatter(Mr_O30, Gr_O30, c = B_O30, s=100, cmap=cmap, marker = '.', vmax= 1.0e+00, vmin = -4.0e+00, edgecolors='black')
plt.xlim(-1, 100)
plt.ylim(-1, 100)
plt.title('30$^\circ$C Oligotrophic', fontsize=12)


### END OF 30oC OLIGOTROPHIC ###

## 30oC MESOTROPHIC - subplot ax2 ####

# Growth rate for plausible (GrR)
GrR_M30 = df.loc[, 'gf']
#Mortality rate for plausible (MrR)              
MrR_M30 = df.loc[, 'mf']
# % biomass for plausible (BR)
BR_M30 = df.loc[, '%foram']
BR_M30 = np.log10(BR_M30)

# Growth rate for low_biomass (GrS)
GrS_M30 = df.loc[, 'gf']
#Mortality rate for low_biomass (MrS)             
MrS_M30 = df.loc[, 'mf']
# % biomass for low_biomass (BS)
BS_M30 = df.loc[, '%foram']
BS_M30 = np.log10(BS_M30)

# Growth rate for the other senarios (Gr)
Gr_M30 = df.loc[, 'gf']
#Mortality rate for the other senarios(Mr)             
Mr_M30= df.loc[, 'mf']
# % biomass for the other senarios (B)
B_M30 = df.loc[, '%foram']
B_M30 = np.log10(B_M30)

# make the scatter
plt.subplot(x[1,2])
plausible = plt.scatter(MrR_M30,GrR_M30,c = BR_M30, s = 200, cmap=cmap, marker = "*",vmax= 1.0e+00, vmin = -4.0e+00, edgecolors='black')  
low_biomass = plt.scatter(MrS_M30, GrS_M30, c = BS_M30, s=100, cmap=cmap, marker = 'v', vmax= 1.0e+00, vmin = -4.0e+00,edgecolors='black')
Total = plt.scatter(Mr_M30, Gr_M30, c = B_M30, s=100, cmap=cmap, marker = '.', vmax= 1.0e+00, vmin = -4.0e+00, edgecolors='black')
plt.xlim(-1, 100)
plt.ylim(-1, 100)
plt.title('30$^\circ$C Mesotrophic', fontsize=12)


### END OF 30oC MESOTROPHIC ###

## 30oC EUTROPHIC - subplot ax3 ####

# Growth rate for plausible (GrR)
GrR_E30 = df.loc [, 'gf']
#Mortality rate for plausible (MrR)              
MrR_E30 = df.loc[, 'mf']
# % biomass for plausible (BR)
BR_E30 = df.loc [, '%foram'] 
BR_E30 = np.log10(BR_E30)

# # Growth rate for low_biomass (GrS)
GrS_E30 = df.loc[, 'gf']
# #Mortality rate for low_biomass (MrS)             
MrS_E30 = df.loc[, 'mf']
# # % biomass for low_biomass (BS)
BS_E30 = df.loc [, '%foram'] 
BS_E30 = np.log10(BS_E30)

# Growth rate for the other senarios (Gr)
Gr_E30 = df.loc[, 'gf']
#Mortality rate for the other senarios(Mr)             
Mr_E30 = df.loc[, 'mf']
# % biomass for the other senarios (B)
B_E30 = df.loc[, '%foram']
B_E30 = np.log10(B_E30)

# make the scatter
plt.subplot(x[2,2])
plausible = plt.scatter(MrR_E30,GrR_E30,c = BR_E30, s = 200, cmap=cmap, marker = "*",vmax= 1.0e+00, vmin = -4.0e+00, edgecolors='black')  
low_biomass = plt.scatter(MrS_E30, GrS_E30, c = BS_E30, s=100, cmap=cmap, marker = 'v', vmax= 1.0e+00, vmin = -4.0e+00,edgecolors='black')
Total = plt.scatter(Mr_E30, Gr_E30, c = B_E30, s=100, cmap=cmap, marker = '.', vmax= 1.0e+00, vmin = -4.0e+00, edgecolors='black')
plt.xlim(-1, 100)
plt.ylim(-1, 100)
plt.title('30$^\circ$C Eutrophic', fontsize=12)

# plt.text(30.0,90.0,'$\sigma$ = 0.6',              
           # #frameon=False, 
           # fontsize=14)
           
fig.legend((plausible, low_biomass, Total),("plausible", "low biomass","other"),
           scatterpoints=1, 
           loc = 'upper right', 
           ncol=1,
           frameon=False, 
           fontsize=13)  
      
### END OF 30oC EUTROPHIC ###

## 20oC oligotrophic - subplot ax4 ###

# Growth rate for plausible (GrR)
GrR_O20 = df.loc[, 'gf']
#Mortality rate for plausible (MrR)              
MrR_O20 = df.loc[, 'mf']
# % biomass for plausible (BR)
BR_O20 = df.loc[, '%foram']
BR_O20 = np.log10(BR_O20)

# # Growth rate for low_biomass (GrS)
# GrS_O20 =
# #Mortality rate for low_biomass (MrS)             
# MrS_O20 = 
# # % biomass for low_biomass (BS)
# BS_O20 = 
# BS_O20 = 

# Growth rate for the other senarios (Gr)
Gr_O20 = df.loc[, 'gf']
#Mortality rate for the other senarios(Mr)             
Mr_O20 = df.loc[, 'mf']
# % biomass for the other senarios (B)
B_O20 = df.loc[, '%foram']
B_O20 = np.log10(B_O20)

# make the scatter
plt.subplot(x[0,1])
plausible = plt.scatter(MrR_O20,GrR_O20,c = BR_O20, s = 200, cmap=cmap, marker = "*",vmax= 1.0e+00, vmin = -4.0e+00, edgecolors='black')  
low_biomass = plt.scatter(MrS_O20, GrS_O20, c = BS_O20, s=100, cmap=cmap, marker = 'v', vmax= 1.0e+00, vmin = -4.0e+00,edgecolors='black')
Total = plt.scatter(Mr_O20, Gr_O20, c = B_O20, s=100, cmap=cmap, marker = '.', vmax= 1.0e+00, vmin = -4.0e+00, edgecolors='black')
plt.xlim(-1, 100)
plt.ylim(-1, 100)
plt.title('20$^\circ$C Oligotrophic' , fontsize=12)

# plt.text(30.0,90.0,'$\sigma$ = 0.8',              
           # #frameon=False, 
           # fontsize=14)

### END OF 20oC OLIGOTROPHIC ###

## 20oC MESOTROPHIC - subplot ax5####

# Growth rate for plausible (GrR)
GrR_M20 = df.loc[, 'gf']
#Mortality rate for plausible (MrR)              
MrR_M20 = df.loc[, 'mf']
# % biomass for plausible (BR)
BR_M20 = df.loc[, '%foram']
BR_M20 = np.log10(BR_M20)

# # Growth rate for low_biomass (GrS)
# GrS_M20 = df.loc[, 'gf']
# #Mortality rate for low_biomass (MrS)             
# MrS_M20 = df.loc[, 'mf']
# # % biomass for low_biomass (BS)
# #BS_M20 = [np.array([5.47E-03])]
# BS_M20 = df.loc[, '%forams_total_zoo']
# BS_M20 = np.log10(BS_M20)

# Growth rate for the other senarios (Gr)
Gr_M20 = df.loc[, 'gf']
#Mortality rate for the other senarios(Mr)             
Mr_M20 = df.loc[, 'mf']
# % biomass for the other senarios (B)
B_M20 = df.loc[, '%foram']
B_M20 = np.log10(B_M20)
# make the scatter
plt.subplot(x[1,1]) 
plausible = plt.scatter(MrR_M20,GrR_M20,c = BR_M20, s = 200, cmap=cmap, marker = "*",vmax= 1.0e+00, vmin = -4.0e+00, edgecolors='black')  
low_biomass = plt.scatter(MrS_M20, GrS_M20, c = BS_M20, s=100, cmap=cmap, marker = 'v', vmax= 1.0e+00, vmin = -4.0e+00,edgecolors='black')
Total = plt.scatter(Mr_M20, Gr_M20, c = B_M20, s=100, cmap=cmap, marker = '.', vmax= 1.0e+00, vmin = -4.0e+00, edgecolors='black')
plt.xlim(-1, 100)
plt.ylim(-1, 100)
plt.title('20$^\circ$C Mesotrophic', fontsize=12)

# plt.text(30.0,90.0,'$\sigma$ = 0.6',              
           # #frameon=False, 
           # fontsize=14)
           
### END OF 20oC MESOTROPHIC ###

## 20oC EUTROPHIC - subplot ax6 ####

# Growth rate for plausible (GrR)
GrR_E20 = df.loc [0:0, 'gf']
#Mortality rate for plausible (MrR)              
MrR_E20 = df.loc [0:0, 'mf']
# % biomass for plausible (BR)
BR_E20 = df.loc [0:0, '%foram']

# # Growth rate for low_biomass (GrS)
# GrS_E20 = df.loc [, 'gf']
# #Mortality rate for low_biomass (MrS)             
# MrS_E20 = df.loc[, 'mf']
# # % biomass for low_biomass (BS)
# BS_E20 = df.loc[, '%forams_total_zoo']
# BS_E20 = np.log10(BS_E20)

# Growth rate for the other senarios (Gr)
Gr_E20 = df.loc[, 'gf']
#Mortality rate for the other senarios(Mr)             
Mr_E20= df.loc[, 'mf']
# % biomass for the other senarios (B)
B_E20 = df.loc[, '%foram']
B_E20 = np.log10(B_E20)


# make the scatter
plt.subplot(x[2,1])
plausible = plt.scatter(MrR_E20,GrR_E20,c = BR_E20, s = 200, cmap=cmap, marker = "*",vmax= 1.0e+00, vmin = -4.0e+00, edgecolors='black')  
low_biomass = plt.scatter(MrS_E20, GrS_E20, c = BS_E20, s=100, cmap=cmap, marker = 'v', vmax= 1.0e+00, vmin = -4.0e+00,edgecolors='black')
Total = plt.scatter(Mr_E20, Gr_E20, c = B_E20, s=100, cmap=cmap, marker = '.', vmax= 1.0e+00, vmin = -4.0e+00, edgecolors='black')
plt.xlim(-1, 100)
plt.ylim(-1, 100)
plt.title('20$^\circ$C Eutrophic' , fontsize=12)
fig.legend((plausible, low_biomass, Total),("plausible", "low biomass","other"),
           scatterpoints=1, 
           loc = 'upper right', 
           ncol=1,
           frameon=False, 
           fontsize=13)  
### END OF 20oC EUTROHIC ###
 
## 10oC oligotrophic - subplot ax7 ###

# # Growth rate for plausible (GrR)
# GrR_O10 =
# #Mortality rate for plausible (MrR)              
# MrR_O10 =
# # % biomass for plausible (BR)
# BR_O10 = 

# # Growth rate for low_biomass (GrS)
# GrS_O10 =
# #Mortality rate for low_biomass (MrS)             
# MrS_O10 =
# # % biomass for low_biomass (BS)
# BS_O10 = 

# #Growth rate for the other senarios (Gr)
# Gr_O10 = df.loc[,'gf']
# #Mortality rate for the other senarios(Mr)             
# Mr_O10 = df.loc[,'mf']
# #% biomass for the other senarios (B)
# B_O10 = df.loc[,'%foram']
# B_O10 = np.log10(B_O10)
# make the scatter
plt.subplot(x[0,0])
plausible = plt.scatter(MrR_O10,GrR_O10,c = BR_O10, s = 200, cmap=cmap, marker = "*",vmax= 1.0e+00, vmin = -4.0e+00, edgecolors='black')  
low_biomass = plt.scatter(MrS_O10, GrS_O10, c = BS_O10, s=100, cmap=cmap, marker = 'v', vmax= 1.0e+00, vmin = -4.0e+00,edgecolors='black')
Total = plt.scatter(Mr_O10, Gr_O10, c = B_O10, s=100, cmap=cmap, marker = '.', vmax= 1.0e+00, vmin = -4.0e+00, edgecolors='black')
plt.xlim(-1, 100)
plt.ylim(-1, 100)
plt.title('10$^\circ$C Oligotrophic', fontsize=12)

### END OF 10oC OLIGOTROPHIC ###

## 10oC MESOTROPHIC - subplot ax8 ####

# Growth rate for plausible (GrR)
GrR_M10 = df.loc[,'gf']
#Mortality rate for plausible (MrR)              
MrR_M10 = df.loc[,'mf']
# % biomass for plausible (BR)
BR_M10 = df.loc[,'%foram']

# Growth rate for low_biomass (GrS)
GrS_M10 = df.loc[,'gf']
#Mortality rate for low_biomass (MrS)             
MrS_M10 =  df.loc[,'mf']
# % biomass for low_biomass (BS)
BS_M10 = df.loc[,'%foram']
BS_M10 = np.log10(BS_M10)

# Growth rate for the other senarios (Gr)
Gr_M10 = df.loc[,'gf']
#Mortality rate for the other senarios(Mr)             
Mr_M10 = df.loc[,'mf']
# % biomass for the other senarios (B)
B_M10 = df.loc[,'%foram']
B_M10 = np.log10(B_M10)

# make the scatter
plt.subplot(x[1,0])
plausible = plt.scatter(MrR_M10,GrR_M10,c = BR_M10, s = 200, cmap=cmap, marker = "*",vmax= 1.0e+00, vmin = -4.0e+00, edgecolors='black')  
low_biomass = plt.scatter(MrS_M10, GrS_M10, c = BS_M10, s=100, cmap=cmap, marker = 'v', vmax= 1.0e+00, vmin = -4.0e+00,edgecolors='black')
Total = plt.scatter(Mr_M10, Gr_M10, c = B_M10, s=100, cmap=cmap, marker = '.', vmax= 1.0e+00, vmin = -4.0e+00, edgecolors='black')
plt.xlim(-1, 100)
plt.ylim(-1, 100)
plt.title('10$^\circ$C Mesotrophic', fontsize=12)

# plt.text(30.0,90.0,'$\sigma$ = 0.6',              
           # #frameon=False, 
           # fontsize=14)
### END OF 10oC MESOTROPHIC ###

## 10oC EUTROPHIC - subplot ax9 ####

# Growth rate for plausible (GrR)
GrR_E10 =
#Mortality rate for plausible (MrR)              
MrR_E10 = 
# % biomass for plausible (BR)
BR_E10 = 

# Growth rate for low_biomass (GrS)
GrS_E10 =df.loc[,'gf'] 
#Mortality rate for low_biomass (MrS)             
MrS_E10 = df.loc[, 'mf']
# % biomass for low_biomass (BS)
BS_E10 = df.loc [,'%forams']
BS_E10 = np.log10(BS_E10)#log value

# Growth rate for the other senarios (Gr)
Gr_E10 =df.loc[,'gf']
#Mortality rate for the other senarios(Mr)             
Mr_E10 = df.loc[,'mf']
# % biomass for the other senarios (B)
B_E10 = df.loc[,'%foram']
B_E10 = np.log10(B_E10) # log 


# make the scatter
plt.subplot(x[2,0])
plausible = plt.scatter(MrR_E10,GrR_E10,c = BR_E10, s = 200, cmap=cmap, marker = "*",vmax= 1.0e+00, vmin = -4.0e+00, edgecolors='black')  
low_biomass = plt.scatter(MrS_E10, GrS_E10, c = BS_E10, s=100, cmap=cmap, marker = 'v', vmax= 1.0e+00, vmin = -4.0e+00,edgecolors='black')
Total = plt.scatter(Mr_E10, Gr_E10, c = B_E10, s=100, cmap=cmap, marker = '.', vmax= 1.0e+00, vmin = -4.0e+00, edgecolors='black')
plt.xlim(-1, 100)
plt.ylim(-1, 100)
plt.title('10$^\circ$C Eutrophic',fontsize=12)
# plt.text(30.0,90.0,'$\sigma$ = 0.6',              
           # #frameon=False, 
           # fontsize=14)
### END OF 10oC EUTROHIC ###

#
#adjust the subpolts and the colour bar
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.83, 0.1, 0.020, 0.8]) #plt.axes((left, bottom, width, height), facecolor='w')
plt.colorbar(cax=cax)

fig.text(0.455, 0.98,
        'Spinose Carnivorous, 1:2, $\sigma$ = 0.5',
        ha='center', fontsize = 18)


## subtitles xaxis yaxis   

fig.text(0.5, 0.05,
        '%Decrease of mortality rate (cal$_m$$_o$$_r$$_t$)',
        ha='center', fontsize = 17)
                      
fig.text(0.080, 0.55,
        '%Decrease of growth rate (cal$_c$$_o$$_s$$_t$)',
        ha='center', rotation = 'vertical', fontsize = 17)

fig.text(0.455, 0.95,
        'Temperature',
        ha='center', fontsize = 17)

fig.text(0.03, 0.54,
        'Nutrient Concentrarion ',
        ha='center', rotation = 'vertical', fontsize = 17)

fig.text(0.93, 0.55,
        '%Biomass (log$_1$$_0$)',
        ha='center', rotation = 'vertical', fontsize = 17)
plt.show()
