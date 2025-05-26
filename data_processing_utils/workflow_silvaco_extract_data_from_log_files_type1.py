#!/usr/bin/python

import numpy as np
from glob import glob

log_file_all = glob("intraband_NwQWIP_bias_0V_300K_xGa_*_dEcPer_*.log")
num_of_log_header_lines = 25
#NOTE: for each silvaco log file, the data structure is:
# 1 Gain (TM, electrons) (well 1) (1/cm) 
# 2 Gain (TM, holes) (well 1) (1/cm) 
# 3 Radiative Capture Spectral Density (TM, electrons) (well 1) (1/(s eV)) 
# 4 Radiative Capture Spectral Density (TM, holes) (well 1) (1/(s eV)) 
# 5 Spontaneous emission (TM, electrons) (well 1) (1/(s cm3 eV)) 
# 6 Spontaneous emission (TM, holes)  (well 1) (1/(s cm3 eV)) 
# 7 Flux Spectral Density (well 1)  (1/(s cm eV)) 
# 8 Power Spectral Density (well 1)  (W/(cm eV)) 
# --------------------------------------------------#
# The 0th column should be removed
# The 1st column is the energy spectrum
# The 2nd column is the corresponding wavelength
# The data starts from the 3rd column
# --------------------------------------------------#

# set up the output data information
output_file_header="wavelength[um]  spontaneous_emission_TM_electron[1/s/cm3/eV]"

for log_file_temp in log_file_all:
    log_data_temp = np.genfromtxt(log_file_temp,skip_header=num_of_log_header_lines)
    # extract the numeric information on xGa and dEcPer
    xGa_temp = log_file_temp[log_file_temp.find("xGa_")+len("xGa_"):log_file_temp.find("_dEcPer")]
    #xGa_temp = float(xGa_temp)
    dEcPer_temp = log_file_temp[log_file_temp.find("dEcPer_")+len("dEcPer_"):log_file_temp.find(".log")]
    #dEcPer_temp = float(dEcPer_temp)
    # extract the desired data
    # we first need to remove the original 0th column
    log_data_temp  = log_data_temp[:,1:]
    output_data_temp = np.column_stack((log_data_temp[:,1],log_data_temp[:,-4]))
    output_filename_temp = "extracted_intraband_SponEmiTMe_xGa_"+xGa_temp+"_dEcPer_"+dEcPer_temp+".dat"
    np.savetxt(output_filename_temp,output_data_temp,fmt='%.6g',header=output_file_header)
    print(f"**Info: extract the data from {log_file_temp}.")
#end for


