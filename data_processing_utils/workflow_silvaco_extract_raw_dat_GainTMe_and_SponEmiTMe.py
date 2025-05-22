#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

### load experimental data
##data_exp = np.genfromtxt("src_pl_spec_InGaAs_Temperature_300K.txt",skip_header=1)

# preprocess the simulated data
# simulated data pattern:
# "extracted_xGa_xx_dEcPer_xx.dat"
# NOTE: In each data set, it contains both the "Gain" of (TM) electrons and spontaneous emission rate for TM electrons, where is former is the negative of the absorption coefficient and don't forget to take the absolute value of it!
sim_files_all = glob("extracted_intraband_xGa_*_dEcPer_*.dat")
# each *dat file has the following data format:
# xGa dEcPer "Wavelength" "Gain (TM, electrons) (well 1) (1/cm)" "Spontaneous emission (TM, electrons) (well 1) (1/(s cm3 eV))"

sim_data_processed_wavelength = np.empty((0,),dtype=np.float64) # for wavelength
sim_data_processed_Gain       = np.empty((0,),dtype=np.float64) # for Gain TM
sim_data_processed_Spon       = np.empty((0,),dtype=np.float64) # for spontaneous emission rate
sim_data_processed_GainPeak   = np.empty((0,),dtype=np.float64) # for peak Gain TM
for sim_file in sim_files_all:
    print(f"**Info: process data: {sim_file}")
    xGa_temp = sim_file[sim_file.find("xGa_")+len("xGa_"):sim_file.find("_dEcPer")]
    xGa_temp = float(xGa_temp)
    dEcPer_temp = sim_file[sim_file.find("dEcPer_")+len("dEcPer_"):sim_file.find(".dat")]
    dEcPer_temp = float(dEcPer_temp)
    sim_data_all = np.genfromtxt(sim_file,skip_header=1)

    sim_wavelength = sim_data_all[:,0] # unit: [um]
    #NOTE: the src silvaco *.dat file has the wavelength in the reverse order, i.e., it starts from 1.8 um down to 1.0 um.
    sim_wavelength = np.flipud(sim_wavelength)
    sim_data_processed_wavelength = np.hstack((sim_data_processed_wavelength,xGa_temp,dEcPer_temp,sim_wavelength))

    # NOTE: the electron gain is different from absorption coefficients by a negative sign
    sim_data_temp = np.abs(sim_data_all[:,1])
    sim_data_temp = np.flipud(sim_data_temp)
    sim_data_processed_Gain = np.hstack((sim_data_processed_Gain,xGa_temp,dEcPer_temp,sim_data_temp/sim_data_temp.max()))
    # also extract the peak Gain value and its corresponding wavelength
    peak_gain_idx=np.argmax(sim_data_temp)
    sim_data_processed_GainPeak = np.hstack((sim_data_processed_GainPeak,xGa_temp,dEcPer_temp,\
                                             sim_wavelength[peak_gain_idx],sim_data_temp[peak_gain_idx]))

    sim_data_temp = np.abs(sim_data_all[:,2])
    sim_data_temp = np.flipud(sim_data_temp)
    sim_data_processed_Spon = np.hstack((sim_data_processed_Spon,xGa_temp,dEcPer_temp,sim_data_temp/sim_data_temp.max()))
#end for
sim_data_processed_wavelength.shape = (-1,sim_wavelength.size+2)
sim_data_processed_Gain.shape = (-1,sim_data_temp.size+2)
sim_data_processed_GainPeak.shape = (-1,4)
#sim_data_processed_Gain = np.vstack((np.hstack((0,0,sim_wavelength)),sim_data_processed_Gain))
sim_data_processed_Spon.shape = (-1,sim_data_temp.size+2)
#sim_data_processed_Spon = np.vstack((np.hstack((0,0,sim_wavelength)),sim_data_processed_Spon))
# save the processed simulation data
outfile_name_wavelength="all_simulation_data_processed_Wavelength.txt"
outfile_name_Gain="all_simulation_data_processed_GainTMe.txt"
outfile_name_Spon="all_simulation_data_processed_SponEmiTMe.txt"
outfile_name_GainPeak="all_simulation_data_processed_PeakGainTMe.txt"
outfile_header_wavelength="xGa dEcPer  wavelength[um]"
outfile_header_Gain="xGa dEcPer  AbsCoeff_normalised"
outfile_header_Spon="xGa dEcPer  PL_normalised"
outfile_header_GainPeak="xGa dEcPer  Wavelength_at_peak_gain[um] peak_TMGain[1/cm]"
np.savetxt(outfile_name_wavelength,sim_data_processed_wavelength,fmt='%.6g',header=outfile_header_wavelength)
np.savetxt(outfile_name_Gain,sim_data_processed_Gain,fmt='%.6g',header=outfile_header_Gain)
np.savetxt(outfile_name_Spon,sim_data_processed_Spon,fmt='%.6g',header=outfile_header_Spon)
np.savetxt(outfile_name_GainPeak,sim_data_processed_GainPeak,fmt='%.6g',header=outfile_header_GainPeak)

# test plot
#plt.plot(sim_data_processed[0,2:],sim_data_processed[1,2:])
#plt.plot(sim_data_processed[0,2:],sim_data_processed[2,2:])
#plt.plot(sim_data_processed[0,2:],sim_data_processed[3,2:])
#plt.show()

