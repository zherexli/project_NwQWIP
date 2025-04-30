#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

### load experimental data
##data_exp = np.genfromtxt("src_pl_spec_InGaAs_Temperature_300K.txt",skip_header=1)

# preprocess the simulated data
# simulated data pattern:
# "extracted_xGa_xx_dEcPer_xx.dat"
# NOTE: In these data sets, it was the "Gain" of (TM) electrons that was extracted, which is 
# the negative of the absorption coefficient!
sim_files_all = glob("extracted_intraband_xGa_*_dEcPer_*.dat")

#first, extract the sim_wavelength
sim_file_temp = np.genfromtxt(sim_files_all[0],skip_header=1)
sim_wavelength = sim_file_temp[:,0] # unit: [um]
#NOTE: the src silvaco *.dat file has the wavelength in the reverse order, i.e., it starts from 1.8 um down to 1.0 um.
sim_wavelength = np.flipud(sim_wavelength)

sim_data_processed = np.empty((0,),dtype=np.float64)
for sim_file in sim_files_all:
    print(f"**Info: process data: {sim_file}")
    xGa_temp = sim_file[sim_file.find("xGa_")+len("xGa_"):sim_file.find("_dEcPer")]
    xGa_temp = float(xGa_temp)
    dEcPer_temp = sim_file[sim_file.find("dEcPer_")+len("dEcPer_"):sim_file.find(".dat")]
    dEcPer_temp = float(dEcPer_temp)
    sim_data_temp = np.genfromtxt(sim_file,skip_header=1)
    # NOTE: what we get is the gain, different from absorption coefficients by a negative sign
    sim_data_temp = np.abs(sim_data_temp[:,1])
    sim_data_temp = np.flipud(sim_data_temp)
    sim_data_processed = np.hstack((sim_data_processed,xGa_temp,dEcPer_temp,sim_data_temp/sim_data_temp.max()))
#end for
sim_data_processed.shape = (int(sim_data_processed.size/(sim_data_temp.size+2)),sim_data_temp.size+2)
sim_data_processed = np.vstack((np.hstack((0,0,sim_wavelength)),sim_data_processed))
# save the processed simulation data
outfile_name="all_simulation_data_processed_GainTMe.txt"
outfile_header="xGa dEcPer  GainTMe_normalised"
np.savetxt(outfile_name,sim_data_processed,fmt='%.6g',header=outfile_header)

# test plot
#plt.plot(sim_data_processed[0,2:],sim_data_processed[1,2:])
#plt.plot(sim_data_processed[0,2:],sim_data_processed[2,2:])
#plt.plot(sim_data_processed[0,2:],sim_data_processed[3,2:])
#plt.show()

