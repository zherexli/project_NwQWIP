import numpy as np
import matplotlib.pyplot as plt

sim_data = np.genfromtxt("all_simulation_data_processed.txt",skip_header=1)

xGa_design = np.linspace(0.203,0.707,num=57)
dEc_design = np.linspace(0.05,0.95,num=19)

xGa_sim = sim_data[1:,0]
dEc_sim = sim_data[1:,1]

for xGa_temp in xGa_design:
    idx_temp = np.isclose(xGa_sim,xGa_temp)
    print(f"xGa = {xGa_temp}: total {idx_temp.sum()} items: {dEc_sim[idx_temp]}\n")
#end for








