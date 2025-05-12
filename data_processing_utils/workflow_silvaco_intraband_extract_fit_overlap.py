#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

# load the computed loss (MSE) from the interband and intraband transisions.
# For the simulated interband data, we adopted the simualted spontaneous emission rate at 300 K
# for the simualted intraband data, we adopted the simulated absorption coefficients (TM Gain) at 300 K
# NOTE that the computed MSEs may not be normalised. We need to normalise them for the subsequent comparison.
# Both sorted loss data has the format as follows:
## [xGa; dEcper;  MSE]
loss_data_interband = np.genfromtxt("all_loss_sorted_data_processed_interband_300K.txt",skip_header=1)
loss_data_intraband = np.genfromtxt("all_loss_sorted_data_processed_intraband_300K_GainTMe.txt",skip_header=1)
# normalise the loaded MSEs
loss_data_interband[:,2] = loss_data_interband[:,2]/loss_data_interband[:,2].max()
loss_data_intraband[:,2] = loss_data_intraband[:,2]/loss_data_intraband[:,2].max()

# set the MSE threshold value for finding the overlapped region
# The routine will find the overlapped ['xGa'; 'dEcPer'] pairs given the threshold value below
loss_threshold = 0.05
loss_interband_red = loss_data_interband[loss_data_interband[:,2]<=loss_threshold,:]
loss_intraband_red = loss_data_intraband[loss_data_intraband[:,2]<=loss_threshold,:]

# find the overlap
loss_interband_overlapped = np.empty((0,),dtype=np.float64)
loss_intraband_overlapped = np.empty((0,),dtype=np.float64)
if loss_intraband_red.shape[0]<=loss_interband_red.shape[0]:
    for ii in range(loss_intraband_red.shape[0]):
        overlap_idx_temp = np.isclose(loss_interband_red[:,0],loss_intraband_red[ii,0]) & np.isclose(loss_interband_red[:,1],loss_intraband_red[ii,1])
        if overlap_idx_temp.sum()==True:
            loss_interband_overlapped = np.hstack((loss_interband_overlapped,loss_interband_red[overlap_idx_temp,:].flatten()))
            loss_intraband_overlapped = np.hstack((loss_intraband_overlapped,loss_intraband_red[ii,:].flatten()))
        #end if
    #end for
    #NOTE: in this case, the data 'loss_intraband_overlapped' is natuarally sorted ascendingly so you don't need to do anything. 
    loss_interband_overlapped = loss_interband_overlapped.reshape(-1,3)
    loss_intraband_overlapped = loss_intraband_overlapped.reshape(-1,3)
else:
    for ii in range(loss_interband_red.shape[0]):
        overlap_idx_temp = np.isclose(loss_interband_red[ii,0],loss_intraband_red[:,0]) & np.isclose(loss_interband_red[ii,1],loss_intraband_red[:,1])
        if overlap_idx_temp.sum()==True:
            loss_interband_overlapped = np.hstack((loss_interband_overlapped,loss_interband_red[ii,:].flatten()))
            loss_intraband_overlapped = np.hstack((loss_intraband_overlapped,loss_intraband_red[overlap_idx_temp,:].flatten()))
        #end if
    #end for
    #NOTE: in this case, the data 'loss_interband_overlapped' is naturally sorted ascendingly. 
    #      BUT, we only want the best fit for the intraband transition, not interband transition.
    #      So, we need to sort the data 'loss_intraband_overlapped' in an ascending order.
    loss_interband_overlapped = loss_interband_overlapped.reshape(-1,3)
    loss_intraband_overlapped = loss_intraband_overlapped.reshape(-1,3)
    loss_intraband_overlapped = loss_intraband_overlapped[loss_intraband_overlapped[:,-1].argsort(),:]
    #TODO: maybe you can sort out the interband transition data according to the same order.
#end if

if loss_intraband_overlapped.size==0:
    print("**Info: NO overlapped MSE region is found! The loss threshold value is too small!")
else:
    #save the data
    output_fname_interband = "loss_interband_threshold_"+str(loss_threshold)+".txt"
    output_fname_intraband = "loss_intraband_threshold_"+str(loss_threshold)+".txt"
    output_headers = "xGa dEcPer MSE"
    np.savetxt(output_fname_interband,loss_interband_overlapped,fmt='%.6g',header=output_headers)
    np.savetxt(output_fname_intraband,loss_intraband_overlapped,fmt='%.6g',header=output_headers)
    print(f"**Info: save the data: {output_fname_interband}.")
    print(f"**Info: save the data: {output_fname_intraband}.")
    #pass
#end if

# plot plan: 2x2
# top row, interband, intraband, with highlighted data within the loss_threshold
# bottom row, interband exp vs. sim, intraband exp vs. sim, with the selected idx and normliased MSE value, respectively. 
# NOTE: always set the best fit index based on the fitting of the inTRAband transition, not the interband transition. 

# Load the experimental data
exp_interband = np.genfromtxt("src_pl_spec_InGaAs_Temperature_300K.txt",skip_header=1)
exp_interband[:,0] = exp_interband[:,0]/1000.0 #unit: um
exp_interband[:,1] = np.abs(exp_interband[:,1])
exp_interband[:,1] = exp_interband[:,1]/exp_interband[:,1].max()
exp_intraband = np.genfromtxt("intraband_responsivity_bias_6V_108K.txt",skip_header=1)
exp_intraband[:,0] = exp_intraband[:,0]/1000.0 #unit: um
exp_intraband[:,1] = np.abs(exp_intraband[:,1])
exp_intraband[:,1] = exp_intraband[:,1]/exp_intraband[:,1].max()
# Load the src simualted data for plots
# For interband simulation data, the data format is: 
# xGa dEcPer  PL_normalised
sim_src_interband = np.genfromtxt("all_simulation_data_processed_interband_PL_300K.txt",skip_header=1)
sim_src_interband_wave = np.genfromtxt("all_simulation_data_processed_interband_Wavelength.txt",skip_header=1)
# For intraband simulation data, the data format is: 
# xGa dEcPer  AbsCoeff_normalised
sim_src_intraband = np.genfromtxt("all_simulation_data_processed_intraband_GainTMe_300K.txt",skip_header=1)
sim_src_intraband_wave = np.genfromtxt("all_simulation_data_processed_intraband_Wavelength.txt",skip_header=1)
# Load the source loss data 
# 1. interband data
loss_src_interband = np.genfromtxt("all_loss_data_processed_interband_PL_300K.txt",skip_header=1)
loss_grid_interband = loss_src_interband[:,2]/loss_src_interband[:,2].max()
# prepare the plot using the source loss data
num_dEc_interband = 19 
num_xGa_interband = 99
xGa_grid_interband = loss_src_interband[:,0]
dEc_grid_interband = loss_src_interband[:,1]
xGa_grid_interband = np.reshape(xGa_grid_interband,(num_dEc_interband,num_xGa_interband),order='F')
dEc_grid_interband = np.reshape(dEc_grid_interband,(num_dEc_interband,num_xGa_interband),order='F')
loss_grid_interband = np.reshape(loss_grid_interband,(num_dEc_interband,num_xGa_interband),order='F')
# 2. intraband data
loss_src_intraband = np.genfromtxt("all_loss_data_processed_intraband_GainTMe_300K.txt",skip_header=1)
loss_grid_intraband = loss_src_intraband[:,2]/loss_src_intraband[:,2].max()
# prepare the plot using the source loss data
num_dEc_intraband = 19
num_xGa_intraband = 57
xGa_grid_intraband = loss_src_intraband[:,0]
dEc_grid_intraband = loss_src_intraband[:,1]
xGa_grid_intraband = np.reshape(xGa_grid_intraband,(num_dEc_intraband,num_xGa_intraband),order='F')
dEc_grid_intraband = np.reshape(dEc_grid_intraband,(num_dEc_intraband,num_xGa_intraband),order='F')
loss_grid_intraband = np.reshape(loss_grid_intraband,(num_dEc_intraband,num_xGa_intraband),order='F')
# -------------------------------------------------------------------------------------------------------#

fig1=plt.figure(1)
ax1_1 = fig1.add_subplot(121)
pc1_1=ax1_1.pcolormesh(xGa_grid_interband,dEc_grid_interband,loss_grid_interband,vmin=0,vmax=1,cmap='GnBu_r')
#ax1.set_aspect('equal',adjustable='box')
ax1_1.set_xlabel(r"$x$ for In$_{1-x}$Ga$_x$As",fontsize=16)
ax1_1.set_ylabel(r"$\Delta$E$_c$ as a fraction",fontsize=16)
ax1_1.tick_params(axis='both',which='both',direction='in',labelsize=16)
ax1_1.set_xlim([0,1])
ax1_1.set_ylim([0,1])
#cb1=fig1.colorbar(pc1,ticks=np.linspace(Eg_nm.min(),Eg_nm.max(),num=7),format='%i')
cb1_1=fig1.colorbar(pc1_1,ticks=[0.0,0.2,0.4,0.6,0.8,1.0])
cb1_1.set_label(label='Mean Squared Errors',fontsize=18)
#NOTE: for the scatter plot, we always use the best fit from the intraband data
sc11=ax1_1.scatter(loss_intraband_overlapped[:,0],loss_intraband_overlapped[:,1],s=80,\
                   alpha=1,linewidth=1.5,edgecolors='red',facecolor='none')


ax1_2 = fig1.add_subplot(122)
pc1_2=ax1_2.pcolormesh(xGa_grid_intraband,dEc_grid_intraband,loss_grid_intraband,vmin=0,vmax=1,cmap='PuRd_r')
#ax1.set_aspect('equal',adjustable='box')
ax1_2.set_xlabel(r"$x$ for In$_{1-x}$Ga$_x$As",fontsize=16)
ax1_2.set_ylabel(r"$\Delta$E$_c$ as a fraction",fontsize=16)
ax1_2.tick_params(axis='both',which='both',direction='in',labelsize=16)
ax1_2.set_xlim([0,1])
ax1_2.set_ylim([0,1])
#cb1=fig1.colorbar(pc1,ticks=np.linspace(Eg_nm.min(),Eg_nm.max(),num=7),format='%i')
cb1_2=fig1.colorbar(pc1_2,ticks=[0.0,0.2,0.4,0.6,0.8,1.0])
cb1_2.set_label(label='Mean Squared Errors',fontsize=18)
sc12=ax1_2.scatter(loss_intraband_overlapped[:,0],loss_intraband_overlapped[:,1],s=80,\
                   alpha=1,linewidth=1.5,edgecolors='red',facecolor='none')

fig2=plt.figure(2)
plt_fit_best_idx = 0
fit_best_idx_interband  = np.isclose(sim_src_interband[:,0],loss_intraband_overlapped[plt_fit_best_idx,0]) & np.isclose(sim_src_interband[:,1],loss_intraband_overlapped[plt_fit_best_idx,1])
fit_best_data_interband = sim_src_interband[fit_best_idx_interband,2:]
fit_best_data_interband = np.ravel(fit_best_data_interband)

fit_best_data_interband_wave = sim_src_interband_wave[fit_best_idx_interband,2:]
fit_best_data_interband_wave = np.ravel(fit_best_data_interband_wave)

fit_best_idx_intraband  = np.isclose(sim_src_intraband[:,0],loss_intraband_overlapped[plt_fit_best_idx,0]) & np.isclose(sim_src_intraband[:,1],loss_intraband_overlapped[plt_fit_best_idx,1])
fit_best_data_intraband = sim_src_intraband[fit_best_idx_intraband,2:]
fit_best_data_intraband = np.ravel(fit_best_data_intraband)

fit_best_data_intraband_wave = sim_src_intraband_wave[fit_best_idx_intraband,2:]
fit_best_data_intraband_wave = np.ravel(fit_best_data_intraband_wave)

ax2_1 = fig2.add_subplot(121)
ax2_1.plot(exp_interband[:,0],exp_interband[:,1],'ro',markersize=5,label='Experiment')
#ax2.plot(wavelength_exp,data_exp,'ro',markersize=5,label='Experiment')
ax2_1.plot(fit_best_data_interband_wave,fit_best_data_interband,'k-',linewidth=2.5,label='Best fit: Sim')
ax2_1.legend(loc='best',fontsize=15)
ax2_1.tick_params(axis='both',which='both',direction='in',labelsize=16)
ax2_1.set_xlim([0.95,1.85])
ax2_1.set_ylim([-0.05,1.05])
ax2_1.set_xlabel(r"Wavelength ($\mu$m)",fontsize=16)
ax2_1.set_ylabel(r"PL (Normalised)",fontsize=16)

ax2_2 = fig2.add_subplot(122)
ax2_2.plot(exp_intraband[:,0],exp_intraband[:,1],'ro',markersize=5,label='Experiment')
#ax2.plot(wavelength_exp,data_exp,'ro',markersize=5,label='Experiment')
ax2_2.plot(fit_best_data_intraband_wave,fit_best_data_intraband,'k-',linewidth=2.5,label='Best fit: Sim')
ax2_2.legend(loc='best',fontsize=15)
ax2_2.tick_params(axis='both',which='both',direction='in',labelsize=16)
ax2_2.set_xlim([1.95,6.05])
ax2_2.set_ylim([-0.05,1.05])
ax2_2.set_xlabel(r"Wavelength ($\mu$m)",fontsize=16)
ax2_2.set_ylabel(r"Responsivity (Normalised)",fontsize=16)


plt.show()
