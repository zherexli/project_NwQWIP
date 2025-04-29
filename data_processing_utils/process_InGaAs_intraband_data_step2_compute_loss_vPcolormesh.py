#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os

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
#data_sim = np.genfromtxt("all_simulation_data_processed_GainTMe.txt",skip_header=1)
data_sim = np.genfromtxt("all_simulation_data_processed_SponEmiTMe.txt",skip_header=1)
wavelength_sim = data_sim[0,2:]; #unit:[um]
#NOTE: remove the first row of the simulated data that contains wavelength informaiton
data_sim = data_sim[1:,:]
#-------------------------------------------------------------------------------------
# the data dimensions come from the Silvaco simulation setup
num_dEc = 19
num_xGa = 57

# set up the output file name for the computed loss 
#output_file_name   = "all_loss_data_processed_GainTMe.txt"
output_file_name   = "all_loss_data_processed_SponEmiTMe.txt"
#output_file_sorted_name   = "all_loss_sorted_data_processed_GainTMe.txt"
output_file_sorted_name   = "all_loss_sorted_data_processed_SponEmiTMe.txt"
output_file_header = "xGa    dEcper  MSE"
if os.path.exists(output_file_name):
    print(f"**Info: loading the computed loss file: {output_file_name}.")
    loss_all = np.genfromtxt(output_file_name,skip_header=1)
    loss_all_sorted = loss_all.copy()
    loss_all_sorted = loss_all_sorted[loss_all_sorted[:,2].argsort(),:]
    np.savetxt(output_file_sorted_name,loss_all_sorted,fmt='%.6g',header=output_file_header)
else:
    # calcualte the loss for each case
    loss_all = np.empty((0,),dtype=np.float64)
    for ii in np.arange(data_sim.shape[0]):
        print(f"**Info: processing data {ii+1} out of {data_sim.shape[0]}...")
        data_sim_temp = data_sim[ii,2:]
        data_sim_temp[np.isnan(data_sim_temp)]=0.0
        PL_sim_interp = np.interp(wavelength_exp,wavelength_sim,data_sim_temp)
        loss_temp = (PL_sim_interp-data_exp)**2
        loss_temp = loss_temp.sum()
        loss_all = np.hstack((loss_all,data_sim[ii,:2],loss_temp))
    #end for
    loss_all.shape = (int(loss_all.size/3),3)
    # sort out the xGa and dEc orders
    xGa_sim_range=np.unique(loss_all[:,0])
    dEc_sim_range=np.unique(loss_all[:,1])
    loss_all = loss_all[loss_all[:,0].argsort(),:]
    for xGa_sim_temp in xGa_sim_range:
        xGa_sim_temp_idx= np.isclose(loss_all[:,0],xGa_sim_temp)
        loss_all_part = loss_all[xGa_sim_temp_idx,:]
        dEc_sim_temp_idx_sorted = np.argsort(loss_all_part[:,1])
        loss_all[xGa_sim_temp_idx,:] = loss_all_part[dEc_sim_temp_idx_sorted,:]
    #end for
    np.savetxt(output_file_name,loss_all,fmt='%.6g',header=output_file_header)
#end if

## quick scatter-plot check
#fig3=plt.figure(3)
#ax3=fig3.add_subplot(111)
##NOTE: normalise the loss so it is convenient to compare it with that of interband PL loss
#loss_norm = loss_all[:,2]/loss_all[:,2].max()
#loss_scatter_plt=ax3.scatter(loss_all[:,0],loss_all[:,1],c=loss_norm,alpha=0.8,edgecolors='none',\
#                             vmin=0,vmax=1)
#cbar = fig3.colorbar(loss_scatter_plt,ax=ax3)
#cbar.set_label("Error function (normalised)")
#ax3.set_xlim([0,1])
#ax3.set_ylim([0,1])
#ax3.set_xlabel(r"$x$ for In$_{1-x}$Ga$_x$As",fontsize=16)
#ax3.set_ylabel(r"$\Delta$E$_c$ as a fraction",fontsize=16)
#ax3.tick_params(axis='both',which='both',direction='in',labelsize=16)
### Create a custom position for the colorbar
##cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
##cbar = fig.colorbar(scatter, cax=cax)
##cbar.set_label('Values')

## prepare the plot
xGa_grid = loss_all[:,0]
dEc_grid = loss_all[:,1]
##NOTE: normalise the loss so it is convenient to compare it with that of interband PL loss
loss_grid = loss_all[:,2]/loss_all[:,2].max()

xGa_grid = np.reshape(xGa_grid,(num_dEc,num_xGa),order='F')
dEc_grid = np.reshape(dEc_grid,(num_dEc,num_xGa),order='F')
loss_grid = np.reshape(loss_grid,(num_dEc,num_xGa),order='F')

fig1=plt.figure(1)
ax1 = fig1.add_subplot(121)
pc1=ax1.pcolormesh(xGa_grid,dEc_grid,loss_grid)
#ax1.set_aspect('equal',adjustable='box')
ax1.set_xlabel(r"$x$ for In$_{1-x}$Ga$_x$As",fontsize=16)
ax1.set_ylabel(r"$\Delta$E$_c$ as a fraction",fontsize=16)
ax1.tick_params(axis='both',which='both',direction='in',labelsize=16)
ax1.set_xlim([0,1])
ax1.set_ylim([0,1])
#cb1=fig1.colorbar(pc1,ticks=np.linspace(Eg_nm.min(),Eg_nm.max(),num=7),format='%i')
cb1=fig1.colorbar(pc1)
cb1.set_label(label='Errors',fontsize=18)


ax2 = fig1.add_subplot(122)
# NOTE: find the best fit first
idx_sorted_loss = 0
fit_best_idx  = np.isclose(data_sim[:,0],loss_all_sorted[idx_sorted_loss,0]) & np.isclose(data_sim[:,1],loss_all_sorted[idx_sorted_loss,1])
fit_best_data = data_sim[fit_best_idx,2:]
fit_best_data = np.ravel(fit_best_data)
## extra information:
### Method 4: For comparing entire coordinate pairs
##coordinates = np.array([x0, y0])
### This checks if any row's first two columns match the coordinates
##matches = np.where(np.all(DATA[:, :2] == coordinates, axis=1))[0]
##print("Matching row indices:", matches)
#--------------------------------------
ax2.plot(wavelength_exp_full,data_exp_full,'ro',markersize=5,label='Experiment')
#ax2.plot(wavelength_exp,data_exp,'ro',markersize=5,label='Experiment')
ax2.plot(wavelength_sim,fit_best_data,'k-',linewidth=2.5,label='Best fit: Sim')
ax2.legend(loc='best',fontsize=15)
ax2.tick_params(axis='both',which='both',direction='in',labelsize=16)
ax2.set_xlim([1.95,6.05])
ax2.set_ylim([-0.05,1.05])
ax2.set_xlabel(r"Wavelength ($\mu$m)",fontsize=16)
ax2.set_ylabel(r"Responsivity (Normalised)",fontsize=16)

plt.show()
