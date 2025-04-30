#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib as mpl
def convert_colormap_v1(value,start,end):
    return plt.cm.plasma((value-start*1.0)/(end*1.0-start*1.0))
#end def

# Define the Gaussian function
def gaussian(x, amplitude, mean, standard_deviation):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * standard_deviation ** 2))
#end def

# Define the Varnish formula
def varnish(x, Eg0, alpha, beta):
    return Eg0-alpha*x**2/(x+beta)
#end def

# Read InGa PL spectrum 
pl_InP_77K  = np.genfromtxt("pl_spec_InP_Temperature_77K.txt",skip_header=1)
pl_InP_97K  = np.genfromtxt("pl_spec_InP_Temperature_108K.txt",skip_header=1)
pl_InP_173K = np.genfromtxt("pl_spec_InP_Temperature_175K.txt",skip_header=1)
pl_InP_223K = np.genfromtxt("pl_spec_InP_Temperature_225K.txt",skip_header=1)
pl_InP_273K = np.genfromtxt("pl_spec_InP_Temperature_275K.txt",skip_header=1)
pl_InP_298K = np.genfromtxt("pl_spec_InP_Temperature_300K.txt",skip_header=1)

pl_InP_77K[:,1]  = np.abs(pl_InP_77K[:,1]) 
pl_InP_97K[:,1]  = np.abs(pl_InP_97K[:,1]) 
pl_InP_173K[:,1] = np.abs(pl_InP_173K[:,1]) 
pl_InP_223K[:,1] = np.abs(pl_InP_223K[:,1]) 
pl_InP_273K[:,1] = np.abs(pl_InP_273K[:,1]) 
pl_InP_298K[:,1] = np.abs(pl_InP_298K[:,1]) 

# Normalise the PL spectrum for the curve fitting
pl_InP_77K_norm  = pl_InP_77K[:,1]/pl_InP_77K[:,1].max()
pl_InP_97K_norm  = pl_InP_97K[:,1]/pl_InP_97K[:,1].max()  
pl_InP_173K_norm = pl_InP_173K[:,1]/pl_InP_173K[:,1].max() 
pl_InP_223K_norm = pl_InP_223K[:,1]/pl_InP_223K[:,1].max() 
pl_InP_273K_norm = pl_InP_273K[:,1]/pl_InP_273K[:,1].max() 
pl_InP_298K_norm = pl_InP_298K[:,1]/pl_InP_298K[:,1].max() 

# set the wavelength range [nm]
wavelength_min=700
wavelength_max=1100
wavelength_fine = np.linspace(wavelength_min,wavelength_max,num=200)

# Fit the PL spectrum 
# T = 77 K
# Perform the curve fit
popt_77K, pcov_77K = curve_fit(gaussian, pl_InP_77K[:,0], pl_InP_77K_norm, p0=[1, 900, 100])
# popt contains the best fit values for the parameters of the Gaussian function
amplitude_77K, mean_77K, std_77K = popt_77K
pl_InP_77K_fit = gaussian(wavelength_fine,amplitude_77K,mean_77K,std_77K)

# T = 97 K
# Perform the curve fit
popt_97K, pcov_97K = curve_fit(gaussian, pl_InP_97K[:,0], pl_InP_97K_norm, p0=[1, 900, 100])
# popt contains the best fit values for the parameters of the Gaussian function
amplitude_97K, mean_97K, std_97K = popt_97K
pl_InP_97K_fit = gaussian(wavelength_fine,amplitude_97K,mean_97K,std_97K)

# T = 173 K
# Perform the curve fit
popt_173K, pcov_173K = curve_fit(gaussian, pl_InP_173K[:,0], pl_InP_173K_norm, p0=[1, 900, 100])
# popt contains the best fit values for the parameters of the Gaussian function
amplitude_173K, mean_173K, std_173K = popt_173K
pl_InP_173K_fit = gaussian(wavelength_fine,amplitude_173K,mean_173K,std_173K)

# T = 223 K
# Perform the curve fit
popt_223K, pcov_223K = curve_fit(gaussian, pl_InP_223K[:,0], pl_InP_223K_norm, p0=[1, 900, 100])
# popt contains the best fit values for the parameters of the Gaussian function
amplitude_223K, mean_223K, std_223K = popt_223K
pl_InP_223K_fit = gaussian(wavelength_fine,amplitude_223K,mean_223K,std_223K)

# T = 273 K
# Perform the curve fit
popt_273K, pcov_273K = curve_fit(gaussian, pl_InP_273K[:,0], pl_InP_273K_norm, p0=[1, 900, 100])
# popt contains the best fit values for the parameters of the Gaussian function
amplitude_273K, mean_273K, std_273K = popt_273K
pl_InP_273K_fit = gaussian(wavelength_fine,amplitude_273K,mean_273K,std_273K)

# T = 298 K
# Perform the curve fit
popt_298K, pcov_298K = curve_fit(gaussian, pl_InP_298K[:,0], pl_InP_298K_norm, p0=[1, 900, 100])
# popt contains the best fit values for the parameters of the Gaussian function
amplitude_298K, mean_298K, std_298K = popt_298K
pl_InP_298K_fit = gaussian(wavelength_fine,amplitude_298K,mean_298K,std_298K)


# Fit the extracted peak PL spectrum to Varnish bandgap empirical equation
nm_to_ev_const = 1239.8
Eg_77K_eV  = nm_to_ev_const/mean_77K
Eg_97K_eV  = nm_to_ev_const/mean_97K
Eg_173K_eV = nm_to_ev_const/mean_173K
Eg_223K_eV = nm_to_ev_const/mean_223K
Eg_273K_eV = nm_to_ev_const/mean_273K
Eg_298K_eV = nm_to_ev_const/mean_298K
TL_exp = np.array([77,108,175,225,275,300])
Eg_exp = np.array([Eg_77K_eV,Eg_97K_eV,Eg_173K_eV,Eg_223K_eV,Eg_273K_eV,Eg_298K_eV])

# Set up the Varnish bandgap empirical equation
Eg0_InP = 1.4236 # eV
alpha_InP = 0.363e-3 # eV/K
beta_InP = 162.0 # K
TL_Varnish = np.linspace(77.0,298,num=100)
Eg_Varnish_raw = Eg0_InP - alpha_InP*TL_Varnish**2/(TL_Varnish+beta_InP)

popt_Eg, pcov_Eg = curve_fit(varnish,TL_exp,Eg_exp,p0=[Eg0_InP,alpha_InP,beta_InP])
Eg0_InP_fit,alpha_InP_fit,beta_InP_fit = popt_Eg
Eg_Varnish_fit = varnish(TL_Varnish,Eg0_InP_fit,alpha_InP_fit,beta_InP_fit)
print("**Results: fitted InP's Eg0   = %.6g [eV]"%(Eg0_InP_fit))
print("**Results: fitted InP's alpha = %.6g [eV/K]"%(alpha_InP_fit))
print("**Results: fitted InP's beta  = %.6g [K]"%(beta_InP_fit))
np.savetxt("InP_fitted_Varnish_params.txt",np.array([Eg0_InP_fit,alpha_InP_fit,beta_InP_fit]),fmt='%.6g',\
           header="Eg0[eV]  alpha[eV/K]  beta[K]")

output_bandgap_exp = np.column_stack((TL_exp,Eg_exp))
output_bandgap_exp = np.vstack((output_bandgap_exp,np.array([300,varnish(300,Eg0_InP_fit,alpha_InP_fit,beta_InP_fit)])))
np.savetxt("InP_fitted_bandgap_from_PL.txt",output_bandgap_exp,fmt='%.6g',\
           header="Temperature[K]  Fitted_bandgap[eV]")


# Plot the fitted PL spectrum one by one:
fig1=plt.figure(1)

ax1=fig1.add_subplot(2,3,1)
ax1.set_title("T = 77 K",fontsize=17)
ax1.plot(pl_InP_77K[:,0],pl_InP_77K_norm,'ko',fillstyle='none',label='experiment')
ax1.plot(wavelength_fine,pl_InP_77K_fit,'k-',linewidth=2,label='fit')

ax2=fig1.add_subplot(2,3,2)
ax2.set_title("T = 108 K",fontsize=17)
ax2.plot(pl_InP_97K[:,0],pl_InP_97K_norm,'ko',fillstyle='none',label='experiment')
ax2.plot(wavelength_fine,pl_InP_97K_fit,'k-',linewidth=2,label='fit')

ax3=fig1.add_subplot(2,3,3)
ax3.set_title("T = 175 K",fontsize=17)
ax3.plot(pl_InP_173K[:,0],pl_InP_173K_norm,'ko',fillstyle='none',label='experiment')
ax3.plot(wavelength_fine,pl_InP_173K_fit,'k-',linewidth=2,label='fit')

ax4=fig1.add_subplot(2,3,4)
ax4.set_title("T = 225 K",fontsize=17)
ax4.plot(pl_InP_223K[:,0],pl_InP_223K_norm,'ko',fillstyle='none',label='experiment')
ax4.plot(wavelength_fine,pl_InP_223K_fit,'k-',linewidth=2,label='fit')

ax5=fig1.add_subplot(2,3,5)
ax5.set_title("T = 275 K",fontsize=17)
ax5.plot(pl_InP_273K[:,0],pl_InP_273K_norm,'ko',fillstyle='none',label='experiment')
ax5.plot(wavelength_fine,pl_InP_273K_fit,'k-',linewidth=2,label='fit')

ax6=fig1.add_subplot(2,3,6)
ax6.set_title("T = 300 K",fontsize=17)
ax6.plot(pl_InP_298K[:,0],pl_InP_298K_norm,'ko',fillstyle='none',label='experiment')
ax6.plot(wavelength_fine,pl_InP_298K_fit,'k-',linewidth=2,label='fit')

ax1.tick_params(axis='both',which='both',labelsize=20,direction='in')
ax1.set_xlim([800,1100])
ax2.tick_params(axis='both',which='both',labelsize=20,direction='in')
ax2.set_xlim([800,1100])
ax3.tick_params(axis='both',which='both',labelsize=20,direction='in')
ax3.set_xlim([800,1100])
ax4.tick_params(axis='both',which='both',labelsize=20,direction='in')
ax4.set_xlim([800,1100])
ax5.tick_params(axis='both',which='both',labelsize=20,direction='in')
ax5.set_xlim([800,1100])
ax6.tick_params(axis='both',which='both',labelsize=20,direction='in')
ax6.set_xlim([800,1100])

ax4.set_xlabel(r'Wavelength [nm]',fontsize=25)
ax5.set_xlabel(r'Wavelength [nm]',fontsize=25)
ax6.set_xlabel(r'Wavelength [nm]',fontsize=25)
ax1.set_ylabel(r'PL Intensity [a.u.]',fontsize=25)
ax4.set_ylabel(r'PL Intensity [a.u.]',fontsize=25)


fig2=plt.figure(2)
ax2=fig2.add_subplot(111)
ax2.plot(TL_exp,Eg_exp,'ro',fillstyle='none',markersize=8,label='experimentally fitted')
ax2.plot(TL_Varnish,Eg_Varnish_raw,'k-',linewidth=2,label='Raw Varnish law')
ax2.plot(TL_Varnish,Eg_Varnish_fit,'b-',linewidth=2,label='Fitted Varnish law')
ax2.tick_params(axis='both',which='both',labelsize=20,direction='in')
ax2.set_xlim([50,350])
ax2.set_xlabel("Temperature [K]",fontsize=25)
ax2.set_ylabel("Energy [eV]",fontsize=25)
ax2.legend(loc='upper right',fontsize=20)


### plot the results
#data_color_plt = np.array([77,97,118,173,223,273,298])
#cb_norm_plt1 = mpl.colors.Normalize(vmin=data_color_plt[0],vmax=data_color_plt[-1])
#sm_plt1 = plt.cm.ScalarMappable(cmap='plasma',norm=cb_norm_plt1)
#sm_plt1.set_array([])
#
#fig1=plt.figure(1)
#ax1=fig1.add_subplot(111)
#ax1.plot(pl_InGaAs_77K[:,0],pl_InGaAs_77K_cor,linestyle='-',linewidth=3,color=convert_colormap_v1(data_color_plt[0],data_color_plt[0],data_color_plt[-1]))
#ax1.plot(pl_InGaAs_97K[:,0],pl_InGaAs_97K_cor,linestyle='-',linewidth=3,color=convert_colormap_v1(data_color_plt[1],data_color_plt[0],data_color_plt[-1]))
#ax1.plot(pl_InGaAs_118K[:,0],pl_InGaAs_118K_cor,linestyle='-',linewidth=3,color=convert_colormap_v1(data_color_plt[2],data_color_plt[0],data_color_plt[-1]))
#ax1.plot(pl_InGaAs_173K[:,0],pl_InGaAs_173K_cor,linestyle='-',linewidth=3,color=convert_colormap_v1(data_color_plt[3],data_color_plt[0],data_color_plt[-1]))
#ax1.plot(pl_InGaAs_223K[:,0],pl_InGaAs_223K_cor,linestyle='-',linewidth=3,color=convert_colormap_v1(data_color_plt[4],data_color_plt[0],data_color_plt[-1]))
#ax1.plot(pl_InGaAs_273K[:,0],pl_InGaAs_273K_cor,linestyle='-',linewidth=3,color=convert_colormap_v1(data_color_plt[5],data_color_plt[0],data_color_plt[-1]))
#ax1.plot(pl_InGaAs_298K[:,0],pl_InGaAs_298K_cor,linestyle='-',linewidth=3,color=convert_colormap_v1(data_color_plt[6],data_color_plt[0],data_color_plt[-1]))
#
#ax1.tick_params(axis='both',which='both',labelsize=20,direction='in')
#ax1.set_xlabel(r'Wavelength [nm]',fontsize=25)
#ax1.set_ylabel(r'Corrected PL Intensity [a.u.]',fontsize=25)
#ax1.set_xlim([800,1700])
#
#fig1.subplots_adjust(right=0.8)
## NOTE: the tuple in 'add_axes' are (left, bottom, width, height)
#cb1_ax = fig1.add_axes([0.85,0.02,0.02,0.85]) #put bound state data at the bottom
#cb1 = fig1.colorbar(sm_plt1,ticks=[77,97,118,173,223,273,298],cax=cb1_ax,format='%i')
#cb1.ax.tick_params(labelsize=16)
#cb1.set_label("Temperature [K]",fontsize=20)
#
#
#fig2=plt.figure(2)
#ax2=fig2.add_subplot(111)
#ax2.plot(pl_InGaAs_77K[:,0], pl_InGaAs_77K[:,1],linestyle='-',linewidth=3,color=convert_colormap_v1(data_color_plt[0],data_color_plt[0],data_color_plt[-1]))
#ax2.plot(pl_InGaAs_97K[:,0], pl_InGaAs_97K[:,1],linestyle='-',linewidth=3,color=convert_colormap_v1(data_color_plt[1],data_color_plt[0],data_color_plt[-1]))
#ax2.plot(pl_InGaAs_118K[:,0],pl_InGaAs_118K[:,1],linestyle='-',linewidth=3,color=convert_colormap_v1(data_color_plt[2],data_color_plt[0],data_color_plt[-1]))
#ax2.plot(pl_InGaAs_173K[:,0],pl_InGaAs_173K[:,1],linestyle='-',linewidth=3,color=convert_colormap_v1(data_color_plt[3],data_color_plt[0],data_color_plt[-1]))
#ax2.plot(pl_InGaAs_223K[:,0],pl_InGaAs_223K[:,1],linestyle='-',linewidth=3,color=convert_colormap_v1(data_color_plt[4],data_color_plt[0],data_color_plt[-1]))
#ax2.plot(pl_InGaAs_273K[:,0],pl_InGaAs_273K[:,1],linestyle='-',linewidth=3,color=convert_colormap_v1(data_color_plt[5],data_color_plt[0],data_color_plt[-1]))
#ax2.plot(pl_InGaAs_298K[:,0],pl_InGaAs_298K[:,1],linestyle='-',linewidth=3,color=convert_colormap_v1(data_color_plt[6],data_color_plt[0],data_color_plt[-1]))
#
#ax2.tick_params(axis='both',which='both',labelsize=20,direction='in')
#ax2.set_xlabel(r'Wavelength [nm]',fontsize=25)
#ax2.set_ylabel(r'Raw PL Intensity [a.u.]',fontsize=25)
#ax2.set_xlim([800,1700])
#
#fig2.subplots_adjust(right=0.8)
## NOTE: the tuple in 'add_axes' are (left, bottom, width, height)
#cb2_ax = fig2.add_axes([0.85,0.02,0.02,0.85]) #put bound state data at the bottom
#cb2 = fig2.colorbar(sm_plt1,ticks=[77,97,118,173,223,273,298],cax=cb2_ax,format='%i')
#cb2.ax.tick_params(labelsize=16)
#cb2.set_label("Temperature [K]",fontsize=20)
#
#fig3=plt.figure(3)
#ax3=fig3.add_subplot(111)
#ax3.plot(pl_InGaAs_77K[:,0], pl_InGaAs_77K[:,1]-pl_InGaAs_77K_cor,linestyle='-',linewidth=3,color=convert_colormap_v1(data_color_plt[0],data_color_plt[0],data_color_plt[-1]))
#ax3.plot(pl_InGaAs_97K[:,0], pl_InGaAs_97K[:,1]-pl_InGaAs_97K_cor,linestyle='-',linewidth=3,color=convert_colormap_v1(data_color_plt[1],data_color_plt[0],data_color_plt[-1]))
#ax3.plot(pl_InGaAs_118K[:,0],pl_InGaAs_118K[:,1]-pl_InGaAs_118K_cor,linestyle='-',linewidth=3,color=convert_colormap_v1(data_color_plt[2],data_color_plt[0],data_color_plt[-1]))
#ax3.plot(pl_InGaAs_173K[:,0],pl_InGaAs_173K[:,1]-pl_InGaAs_173K_cor,linestyle='-',linewidth=3,color=convert_colormap_v1(data_color_plt[3],data_color_plt[0],data_color_plt[-1]))
#ax3.plot(pl_InGaAs_223K[:,0],pl_InGaAs_223K[:,1]-pl_InGaAs_223K_cor,linestyle='-',linewidth=3,color=convert_colormap_v1(data_color_plt[4],data_color_plt[0],data_color_plt[-1]))
#ax3.plot(pl_InGaAs_273K[:,0],pl_InGaAs_273K[:,1]-pl_InGaAs_273K_cor,linestyle='-',linewidth=3,color=convert_colormap_v1(data_color_plt[5],data_color_plt[0],data_color_plt[-1]))
#ax3.plot(pl_InGaAs_298K[:,0],pl_InGaAs_298K[:,1]-pl_InGaAs_298K_cor,linestyle='-',linewidth=3,color=convert_colormap_v1(data_color_plt[6],data_color_plt[0],data_color_plt[-1]))
#
#ax3.tick_params(axis='both',which='both',labelsize=20,direction='in')
#ax3.set_xlabel(r'Wavelength [nm]',fontsize=25)
#ax3.set_ylabel(r'PL Intensity Difference [a.u.]',fontsize=25)
#ax3.set_xlim([800,1700])
#
#fig3.subplots_adjust(right=0.8)
## NOTE: the tuple in 'add_axes' are (left, bottom, width, height)
#cb3_ax = fig3.add_axes([0.85,0.02,0.02,0.85]) #put bound state data at the bottom
#cb3 = fig3.colorbar(sm_plt1,ticks=[77,97,118,173,223,273,298],cax=cb3_ax,format='%i')
#cb3.ax.tick_params(labelsize=16)
#cb3.set_label("Temperature [K]",fontsize=20)


plt.show()
