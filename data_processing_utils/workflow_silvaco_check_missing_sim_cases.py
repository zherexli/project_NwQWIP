import numpy as np
import matplotlib.pyplot as plt

# designed cases
xGa_design = np.linspace(0.203,0.707,num=57)
dEc_design = np.linspace(0.05,0.95,num=19)
# pair up all the combinaitons of xGa and dEc parameters
design_grid = np.meshgrid(xGa_design,dEc_design)
combined_design_grid = np.stack(design_grid,axis=-1).reshape(-1,2)

sim_data = np.genfromtxt("all_simulation_data_processed.txt",skip_header=1)
xGa_sim = sim_data[1:,0]
dEc_sim = sim_data[1:,1]
sim_grid = np.column_stack((xGa_sim,dEc_sim))

tol = 1e-5
#NOTE: the "comparison_grid_src" has a dimension of (1035, 1083, 2)
comparison_grid_src = np.abs(sim_grid[:,None]-combined_design_grid)<tol
# "comparison_grid" below will locate the simualted cases. Its shape is (1035,1083)
comparison_grid = np.logical_and(comparison_grid_src[:,:,0],comparison_grid_src[:,:,1])
sim_done_idx = comparison_grid.any(0)
sim_missing_idx = np.logical_not(sim_done_idx)
sim_missing = combined_design_grid[sim_missing_idx,:]
print(sim_missing)

#--------------------------------- Extra information -------------------------------------#
# 1. find elements of array A inside another array B, with float-number comparison
# https://stackoverflow.com/questions/32513424/find-intersection-of-numpy-float-arrays

# 2. carefully read through the broadcast functionality in NumPy
#https://numpy.org/doc/stable/user/basics.broadcasting.html
#-----------------------------------------------------------------------------------------#

# save the unfinished cases
np.savetxt("unfinished_simulation_cases.txt",sim_missing,fmt='%.6g',header="xGa dEc")

#for xGa_temp in xGa_design:
#    idx_temp = np.isclose(xGa_sim,xGa_temp)
#    print(f"xGa = {xGa_temp}: total {idx_temp.sum()} items: {dEc_sim[idx_temp]}\n")
##end for




#test plot
#plt.figure(1)
#plt.pcolormesh(comparison_grid[:,:,0])
#
#plt.figure(2)
#plt.pcolormesh(comparison_grid[:,:,1])

plt.show()
