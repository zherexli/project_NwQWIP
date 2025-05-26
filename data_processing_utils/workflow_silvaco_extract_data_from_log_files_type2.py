#!/usr/bin/python

import numpy as np
from glob import glob

# An exmaple name: 
# "NwQWIP1D_RadialQWs_noOP_LightSource_xo_0.5_yo_-0.5_SpecResp_no_qwip.log"
log_file_all = glob("NwQWIP1D_RadialQWs_noOP_LightSource_xo_*_yo_*_SpecResp_no_qwip.log")
num_of_log_header_lines = 20
num_of_log_var = 17
#NOTE: for each silvaco log type 2 file, the number of columns is equal to the number of variables the log file contains.
#      the number of rows is the number of data points being simulated.
# The current type 2 log file has 17 variables, whose names you can read by using TonyPlot to check. 
# For our current interest, we need to extract the following variables:
# "Optical Wavelength" and "Available photo current", 
# which are in the 6th and 4th columns, respectively.
# (NOTE: index always starts from 0)
# --------------------------------------------------#
# The 0th column of the raw log file contains a letter, which will be removed by np.genfromtxt
# --------------------------------------------------#

# set up the output data information
output_file_header="wavelength[um]  available photocurrent[A]"
idx_in_log_wavelength = 6
idx_in_log_avail_photocurr = 4

for log_file_temp in log_file_all:
    # NOTE: the 'usecols' in 'np.genfromtxt()' removes the first column.
    log_data_temp = np.genfromtxt(log_file_temp,skip_header=num_of_log_header_lines,usecols=range(1,num_of_log_var+1))
    # extract the numeric information on xGa and dEcPer
    xo_temp = log_file_temp[log_file_temp.find("xo_")+len("xo_"):log_file_temp.find("_yo")]
    #xo_temp = float(xo_temp)
    yo_temp = log_file_temp[log_file_temp.find("yo_")+len("yo_"):log_file_temp.find("_SpecResp")]
    #yo_temp = float(yo_temp)
    # extract the desired data
    output_data_temp = np.column_stack((log_data_temp[:,idx_in_log_wavelength],log_data_temp[:,idx_in_log_avail_photocurr]))
    output_filename_temp = "extracted_noOP_LightSource_xo_"+xo_temp+"_yo_"+yo_temp+"_AvaPhoCur_no_qwip.dat"
    np.savetxt(output_filename_temp,output_data_temp,fmt='%.6g',header=output_file_header)
    print(f"**Info: extract the data from {log_file_temp}.")
#end for


