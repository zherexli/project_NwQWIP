#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline

exp_data = np.genfromtxt("src_exp_photocurrent_77K.txt",skip_header=1)
exp_data_x = exp_data[:,0]
exp_data_y = exp_data[:,1]

# method 1: use scipy spline
#spline_lam=0.1
spline_lam=None
spline = make_smoothing_spline(exp_data_x, exp_data_y, lam=spline_lam)
y_smooth = spline(exp_data_x)

# method 2: use numpy windowed smoothing
window_len=65 # should be an odd number
window=np.hanning(window_len)
y_smooth_2 = np.convolve(window/window.sum(),exp_data_y,mode='same')

#np.savetxt("src_exp_photocurrent_77K_smooth_spline_lam_0p1.txt",np.column_stack((exp_data_x,y_smooth)),\
#           fmt='%.6g',header="wavelength[um] photocurrent[A]")
np.savetxt("src_exp_photocurrent_77K_smooth_spline_lam_default.txt",np.column_stack((exp_data_x,y_smooth)),\
           fmt='%.6g',header="wavelength[um] photocurrent[A]")
np.savetxt("src_exp_photocurrent_77K_smooth_win_hanning.txt",np.column_stack((exp_data_x,y_smooth_2)),\
           fmt='%.6g',header="wavelength[um] photocurrent[A]")


fig1=plt.figure(1)
ax1=fig1.add_subplot(121)
ax1.set_title(f"Spline function lam = {spline_lam}",fontsize=16)
ax1.plot(exp_data_x,exp_data_y,'ko',markersize=3,label='original')
ax1.plot(exp_data_x,y_smooth,'r-',linewidth=3,label='smoothed')
ax1.legend(loc='best',fontsize=15)
ax1.tick_params(axis='both',which='both',direction='in',labelsize=16)
ax1.set_xlim([3,5])
#ax1.set_ylim([-0.05,1.05])
ax1.set_xlabel(r"Wavelength ($\mu$m)",fontsize=16)
ax1.set_ylabel(r"Photocurrent",fontsize=16)

ax2=fig1.add_subplot(122)
ax2.set_title(f"Window size = {window_len}",fontsize=16)
ax2.plot(exp_data_x,exp_data_y,'ko',markersize=3,label='original')
ax2.plot(exp_data_x,y_smooth_2,'r-',linewidth=3,label='smoothed')
ax2.legend(loc='best',fontsize=15)
ax2.tick_params(axis='both',which='both',direction='in',labelsize=16)
ax2.set_xlim([3,5])
#ax1.set_ylim([-0.05,1.05])
ax2.set_xlabel(r"Wavelength ($\mu$m)",fontsize=16)
ax2.set_ylabel(r"Photocurrent",fontsize=16)

plt.show()
