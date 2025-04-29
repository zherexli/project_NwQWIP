#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os

# load experimental data
data_exp = np.genfromtxt("src_pl_spec_InGaAs_Temperature_300K.txt",skip_header=1)
wavelength_exp = data_exp[:,0]/1000.0 # unit: um
data_exp[:,1] = data_exp[:,1]/data_exp[:,1].max()

# load the extracted simulation data
# The simulation data has the following format:
# headers
# first data row: 0, 0, simulated_wavelength in um
# second row: following the order: [xGa, dEcPer, normalised PL]
data_sim = np.genfromtxt("all_simulation_data_processed.txt",skip_header=1)
wavelength_sim = data_sim[0,2:]; #unit:[um]
#NOTE: remove the first row of the simulated data that contains wavelength informaiton
data_sim = data_sim[1:,:]
#-------------------------------------------------------------------------------------
# the data dimensions come from the Silvaco simulation setup
num_dEc = 19
num_xGa = 99

# set up the output file name for the computed loss 
output_file_name   = "all_loss_data_processed.txt"
output_file_header = "xGa    dEcper  MSE"
if os.path.exists(output_file_name):
    print(f"**Info: loading the computed loss file: {output_file_name}.")
    loss_all = np.genfromtxt(output_file_name,skip_header=1)
    loss_all_sorted = loss_all.copy()
    loss_all_sorted = loss_all_sorted[loss_all_sorted[:,2].argsort(),:]
    np.savetxt("all_loss_sorted_data_processed.txt",loss_all_sorted,fmt='%.6g',header=output_file_header)

else:
    # calcualte the loss for each case
    loss_all = np.empty((0,),dtype=np.float64)
    for ii in np.arange(data_sim.shape[0]):
        print(f"**Info: processing data {ii+1} out of {data_sim.shape[0]}...")
        data_sim_temp = data_sim[ii,2:]
        data_sim_temp[np.isnan(data_sim_temp)]=0.0
        PL_sim_interp = np.interp(wavelength_exp,wavelength_sim,data_sim_temp)
        loss_temp = (PL_sim_interp-data_exp[:,1])**2
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

# prepare the plot
xGa_grid = loss_all[:,0]
dEc_grid = loss_all[:,1]
loss_grid = loss_all[:,2]

xGa_grid = np.reshape(xGa_grid,(num_dEc,num_xGa),order='F')
dEc_grid = np.reshape(dEc_grid,(num_dEc,num_xGa),order='F')
loss_grid = np.reshape(loss_grid,(num_dEc,num_xGa),order='F')
#NOTE: let's normalise the loss so that it is convenient to compare it with that of intraband transitions
loss_grid = loss_grid/loss_grid.max()

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
fit_best_idx  = np.isclose(data_sim[:,0],loss_all_sorted[0,0]) & np.isclose(data_sim[:,1],loss_all_sorted[0,1])
fit_best_data = data_sim[fit_best_idx,2:]
fit_best_data = np.ravel(fit_best_data)
## extra information:
### Method 4: For comparing entire coordinate pairs
##coordinates = np.array([x0, y0])
### This checks if any row's first two columns match the coordinates
##matches = np.where(np.all(DATA[:, :2] == coordinates, axis=1))[0]
##print("Matching row indices:", matches)
#--------------------------------------
ax2.plot(wavelength_exp,data_exp[:,1],'ro',markersize=5,label='Experiment')
ax2.plot(wavelength_sim,fit_best_data,'k-',linewidth=2.5,label='Best fit: Sim')
ax2.legend(loc='best',fontsize=15)
ax2.tick_params(axis='both',which='both',direction='in',labelsize=16)
ax2.set_xlim([0.95,1.85])
ax2.set_ylim([-0.05,1.05])
ax2.set_xlabel(r"Wavelength ($\mu$m)",fontsize=16)
ax2.set_ylabel(r"PL (Normalised)",fontsize=16)

plt.show()
