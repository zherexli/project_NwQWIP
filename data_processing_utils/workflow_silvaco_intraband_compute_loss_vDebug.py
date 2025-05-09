#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from glob import glob
import os
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("-i","--xGa_index_CL",type=int,default=0,help="Give the index (starting from 0) of the xGa array")
args=parser.parse_args()

def convert_colormap(value,start,end):
    #return plt.cm.cool((value-start*1.0)/(end*1.0-start*1.0))
    return plt.cm.rainbow((value-start*1.0)/(end*1.0-start*1.0))
#end def

# load experimental data
data_exp_src = np.genfromtxt("intraband_responsivity_bias_6V_108K.txt",skip_header=1)
wavelength_exp_full = data_exp_src[:,0]/1000.0 # unit: um
data_exp_full = data_exp_src[:,1]/data_exp_src[:,1].max()
# we will crop the experimental data as well. 
# discard the data below 3000nm
wavelength_exp = wavelength_exp_full[wavelength_exp_full>2.95]
data_exp = data_exp_full[wavelength_exp_full>2.95]

# load the extracted simulation data
# The simulation data has the following format:
# headers
# first data row: 0, 0, simulated_wavelength in um
# second row: following the order: [xGa, dEcPer, normalised PL]
wavelength_data_sim = np.genfromtxt("all_simulation_data_processed_Wavelength.txt",skip_header=1)
data_sim = np.genfromtxt("all_simulation_data_processed_GainTMe.txt",skip_header=1)
#data_sim = np.genfromtxt("all_simulation_data_processed_SponEmiTMe.txt",skip_header=1)
#-------------------------------------------------------------------------------------
# the data dimensions come from the Silvaco simulation setup
num_dEc = 19
num_xGa = 57
xGa_design = np.linspace(0.203,0.707,num=num_xGa)
idx_xGa = np.isclose(data_sim[:,0],xGa_design[args.xGa_index_CL])
data_sim_local = data_sim[idx_xGa,:]
wavelength_sim_local = wavelength_data_sim[idx_xGa,:]

#set up plot parameters
data_color_plt = np.arange(data_sim_local.shape[0])
norm_color_plt = mpl.colors.Normalize(vmin=data_sim_local[:,1].min(),vmax=data_sim_local[:,1].max())
sm_color_plt = plt.cm.ScalarMappable(cmap='rainbow',norm=norm_color_plt)
sm_color_plt.set_array([])

fig1=plt.figure(1)
fig1.subplots_adjust(right=0.8)
cb1_ax = fig1.add_axes([0.85,0.03,0.015,0.85]) #put data < breakdown at the bottome
ax1=fig1.add_subplot(111)
fig2=plt.figure(2)
fig2.subplots_adjust(right=0.8)
cb2_ax = fig2.add_axes([0.85,0.03,0.015,0.85]) #put data < breakdown at the bottome
ax2=fig2.add_subplot(111)
for ii in np.arange(data_sim_local.shape[0]):
    #
    ax1.set_title("xGa = %.3g"%(xGa_design[args.xGa_index_CL]),fontsize=16)
    ax1.plot(wavelength_sim_local[ii,2:],data_sim_local[ii,2:],linestyle='solid',linewidth=1,
             color=convert_colormap(data_color_plt[ii],data_color_plt[0],data_color_plt[-1]),
             label='dEcPer=%.2g'%(data_sim_local[ii,1]))

    ax2.set_title("xGa = %.3g"%(xGa_design[args.xGa_index_CL]),fontsize=16)
    ax2.plot(wavelength_sim_local[ii,2:-1],np.diff(data_sim_local[ii,2:]),linestyle='solid',linewidth=1,
             color=convert_colormap(data_color_plt[ii],data_color_plt[0],data_color_plt[-1]),
             label='dEcPer=%.2g'%(data_sim_local[ii,1]))

    gain_slope_temp = np.diff(data_sim_local[ii,2:])
    if gain_slope_temp.max()*gain_slope_temp.min()>0:
        print(f"dEc = {data_sim_local[ii,1]} has a monotonic trend")
    else:
        print(f"dEc = {data_sim_local[ii,1]} has at least one peak")
    #end if
#end for

ax1.plot(wavelength_exp_full,data_exp_full,'ko',markersize=5,label='Experiment')
cb1=fig1.colorbar(sm_color_plt,cax=cb1_ax,ticks=data_sim_local[:,1],
                   format='%.2g')
cb1.set_label(label='dEc in Percentage [%]',fontsize=16)
ax1.tick_params(axis='both',which='both',direction='in',labelsize=14)
ax1.set_xlim([1.95,6.05])
ax1.set_ylim([-0.05,1.05])
ax1.set_xlabel(r"Wavelength ($\mu$m)",fontsize=16)
ax1.set_ylabel(r"Responsivity (Normalised)",fontsize=16)

cb2=fig2.colorbar(sm_color_plt,cax=cb2_ax,ticks=data_sim_local[:,1],
                   format='%.2g')
cb2.set_label(label='dEc in Percentage [%]',fontsize=16)
ax2.tick_params(axis='both',which='both',direction='in',labelsize=14)
ax2.set_xlim([1.95,6.05])
#ax2.set_ylim([-0.05,1.05])
ax2.set_xlabel(r"Wavelength ($\mu$m)",fontsize=16)
ax2.set_ylabel(r"Gradient",fontsize=16)



plt.show()


