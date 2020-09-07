"""
Model code for the study "Investigating the effect of spines and diet on planktonic foraminifera’s biogeography with a trait-based ecosystem model"
by Maria Grigoratou, Fanny M. Monteiro, Andy Ridgwell and Daniela N. Schmidt

code authors: Maria Grigoratou
contact author: Maria Grigoratou, mgrigoratou@gmri.org
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time
from scipy.integrate import odeint
import pandas as pd
from openpyxl import Workbook
import xlsxwriter
from xlsxwriter.utility import xl_rowcol_to_cell
#
##phytoplankton diameter
#D=np.zeros([1,50])
#D[0,0]=0.6204
#for n in range(1,25):
#    D[0,n]=D[0,n-1]*np.power(2,1.0/3.0)
###zooplankton diameter
#for n in range(0,25):
#    a=1024*(4.0/3.0*np.pi*(D[0,n]*0.5)**3) # 
#    D[0,n+25]=2.0*((3.0*a/(4.0*np.pi))**(1.0/3.0))



##diameters 
D = np.array([6.20400000e-01,   7.81655019e-01,   9.84823613e-01,   1.24080000e+00,
    1.56331004e+00,   1.96964723e+00,   2.48160000e+00,   3.12662008e+00,
    3.93929445e+00,   4.96320000e+00,   6.25324015e+00,   7.87858890e+00,
    9.92640000e+00,   1.25064803e+01,   1.57571778e+01,   1.98528000e+01,
    2.50129606e+01,   3.15143556e+01,   3.97056000e+01,   5.00259212e+01,
    6.30287112e+01,   7.94112000e+01,   1.00051842e+02,   1.26057422e+02,
    1.58822400e+02,   6.25324015e+00,   7.87858890e+00,   9.92640000e+00,
    1.25064803e+01,   1.57571778e+01,   1.98528000e+01,   2.50129606e+01,
    3.15143556e+01,   3.97056000e+01,   5.00259212e+01,   6.30287112e+01,
    7.94112000e+01,   1.00051842e+02,   1.26057422e+02,   1.58822400e+02,
    2.00103685e+02,   2.52114845e+02,   3.17644800e+02,   4.00207370e+02,
    5.04229690e+02,   6.35289600e+02,   8.00414740e+02,   1.00845938e+03,
    1.27057920e+03,   1.60082948e+03, 1.58822400e+02, 0.0], ndmin =2) # the size for foram adult  equals to 1.58822400e+02


nsize=D.size 
plankton_type=np.zeros([1,nsize])
plankton_type[0,0:25]=1 #phytoplankton
plankton_type[0,25:50]=2 #zooplankton 
plankton_type[0,50:51]=3 # foram
n_iter= 3 
N0_range=np.linspace(1.0, 5.0, n_iter) #Deep Nutrient concentration
equilibrium=np.zeros([n_iter,nsize])

model=Plankton_Size_Chemostat(10000,10000,N0_range[0],D,plankton_type) # run for 10,000 days


for n in range(0,n_iter):    
    print('N0 =',N0_range[n])
    model.No=N0_range[n]
    model.pred_model='foodweb'
    model.thita_opt=D[0,25]/D[0,0]
    model.run()
    output=model.output_B
    N= model.output_N
    t= model.output_time
    equilibrium[n,:]=np.mean(output[10000-1000-1:10000-1,:],axis=0)
    tmp=equilibrium[n,:]
    #tmp[tmp<1e-4]=1e-4
    model.B_init_value=output[0,:]
    print(model.run_status)
    print(model.run_time)

cumsum=np.cumsum(equilibrium,axis=1)

### plots ###

plt.figure(1)
for n in range(0,int(nsize/2)):
    plt.plot(np.sum(equilibrium,axis=1),cumsum[:,n],'g')  
	
plt.plot(np.sum(equilibrium,axis=1),cumsum[:,nsize/2:-1],'r')

ZP_ratio=np.zeros([1,n_iter])
for n in range(0,n_iter):
    ZP_ratio[0,n]= np.sum(equilibrium[n,nsize/2:-1])/ np.sum(equilibrium[n,0:nsize/2])

##plot through time
plt.figure(2)
for n in range(0,int(nsize/2)):
    plt.plot(t,output[:,n],'g')
    plt.xlabel('Times (days)')
    plt.ylabel('phytoplankton biomass (mmolN m-3)')
plt.legend()
#plt.show()

plt.figure(3)
plt.plot(t,output[:,nsize/2:-1],'r')
plt.xlabel('Times (days)')
plt.ylabel('zooplankton biomass (mmolN m-3)')
plt.legend()
#plt.show()

plt.figure(4)
plt.plot(t,output[:,50],'b')
plt.xlabel('Times (days)')
plt.ylabel('forams biomass (mmolN m-3)')
plt.legend()
#plt.show()

plt.figure(5)
plt.plot(t, N, 'b')
plt.xlabel('Times (days)')
plt.ylabel('Nutrient concentration (mmolN m-3)')
plt.legend()
#plt.show()


####Transfer data from DataFrame to excel

#create DataFrame

groups= pd.DataFrame(model.PFT_F, model.PFT_Z) 
diameter = pd.DataFrame(model.D[0])
volume = pd.DataFrame(model.V[0])
#
pcmax = pd.DataFrame(model.pcmax_a[0])
mortality = pd.DataFrame(model.m[0])
mumax = pd.DataFrame(model.mumax[0])
kN = pd.DataFrame(model.kN[0])
growth = pd.DataFrame(model.g[0])
gf = pd.DataFrame(model.gf[0])
f = pd.DataFrame(model.f)


#create excel filE
results = pd.ExcelWriter('"Add the directory and excel file name".xlsx', engine='xlsxwriter')

## Convert the dataframe to an XlsxWriter Excel object.
groups.to_excel(results, sheet_name='plankton_info',header=True,float_format="%10.7f", startcol=0, startrow=0)
diameter.to_excel(results, sheet_name='plankton_info',header=True,float_format="%10.7f", startcol=2, startrow=0)
volume.to_excel(results, sheet_name='plankton_info',header=True,float_format="%10.7f", startcol=4, startrow=0)
pcmax.to_excel(results, sheet_name='plankton_info',header=True,float_format="%10.7f", startcol=6, startrow=0)
mortality.to_excel(results, sheet_name='plankton_info',header=True,float_format="%10.7f", startcol=8, startrow=0)
mumax.to_excel(results, sheet_name='plankton_info',header=True,float_format="%10.7f", startcol=10, startrow=0)
kN.to_excel(results, sheet_name='plankton_info',header=True,float_format="%10.7f", startcol=12, startrow=0)
growth.to_excel(results, sheet_name='plankton_info',header=True,float_format="%10.7f", startcol=14, startrow=0)
gf.to_excel(results, sheet_name='plankton_info',header=True,float_format="%10.7f", startcol=16, startrow=0)
f.to_excel(results, sheet_name='plankton_info',header=True,float_format="%10.7f", startcol=18, startrow=0)


# write a title to excel shell
worksheet = results.sheets['plankton_info']
worksheet.write_string(0, 0, 'plankton group')
worksheet.write_string(0, 1, 'plankton type')
worksheet.write_string(0, 2, 'plankton group')
worksheet.write_string(0, 3, 'diameter microm')
worksheet.write_string(0, 4, 'plankton group')
worksheet.write_string(0, 5, 'volume')
worksheet.write_string(0, 6, 'plankton group')
worksheet.write_string(0, 7, 'pcmax')
worksheet.write_string(0, 8, 'plankton group')
worksheet.write_string(0, 9, 'mortality')
worksheet.write_string(0, 10, 'plankton group')
worksheet.write_string(0, 11, 'mumax')
worksheet.write_string(0, 12, 'plankton group')
worksheet.write_string(0, 13, 'kN')
worksheet.write_string(0, 14, 'plankton group')
worksheet.write_string(0, 15, 'growth rate')
worksheet.write_string(0, 16, 'plankton group')
worksheet.write_string(0, 17, 'gf')
worksheet.write_string(0, 18, 'plankton group')
worksheet.write_string(0, 19, 'f')


### second sheet NPZ info ###
N_info = pd.DataFrame(N) 
PZ_info = pd.DataFrame(output)

### third sheet ### 
N_range = pd.DataFrame(N0_range)
equilibrium_info = pd.DataFrame(equilibrium)
N_range.to_excel(results, sheet_name='equilibrium',header=True,float_format="%10.7f", startcol=0, startrow=0)
equilibrium_info.to_excel(results, sheet_name='equilibrium',header=True,float_format="%10.20f", startcol=2, startrow=0)

### write a title to excel shell ###
worksheet = results.sheets['equilibrium']
worksheet.write_string(0, 1, 'N_range')
worksheet.write_string(0, 2, 'Biomass')

### N_range = pd.DataFrame(N0_range) ###
ZP_ratio = pd.DataFrame(ZP_ratio)
ZP_ratio.to_excel(results, sheet_name='ZP_ratio',header=True,float_format="%10.7f", startcol=0, startrow=0)

### write a title to excel shell ###
worksheet = results.sheets['ZP_ratio']

graze= pd.DataFrame(model.graze_dt)
graze.to_excel(results, sheet_name='graze',header=True,float_format="%10.20f", startcol=0, startrow=0)
# write a title to excel shell
worksheet = results.sheets['graze']

### Close the Pandas Excel writer and output the Excel file. ###
results.save()

##end
