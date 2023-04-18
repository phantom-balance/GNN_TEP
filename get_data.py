import h5py
import numpy as np
import einops

def get_data(mode="Mode1", IDV=None, intensities=None, runs=None):
    file_path = f"data/TEP_{mode}.h5"
    file = h5py.File(file_path, 'r')


    data_list = []
    for fault_type in (IDV):
        for magnitude in intensities:
            for run in runs:
                print(f"FAULT={fault_type} ### MAGNITUDE={magnitude} #### RUN={run}")
                if (mode=="Mode1" and fault_type == 6 and magnitude == 100):
                    print("simulation was incomplete")
                else:
                    dat1 = np.array(file[mode]["SingleFault"]["SimulationCompleted"][f"IDV{fault_type}"][f"{mode}_IDVInfo_{fault_type}_{magnitude}"][f"Run{run}"]["processdata"])
                    dat = dat1[:,1:-1] # removing 1st column=time and last column=agitatorcap(100_unvaried)

                    if (dat.shape[0]==2001):

                        lab1_ = np.zeros(601)
                        lab2_ = np.ones(dat.shape[0]-601)*fault_type
                        label = np.concatenate((lab1_, lab2_), axis=0)
                        label = einops.rearrange(label, 'i -> i 1')
                        data = np.concatenate((dat, label), axis=1)
                        data_list.append(data)
                        

                    elif (dat.shape[0]==2000):
                        lab1_ = np.zeros(600)
                        lab2_ = np.ones(dat.shape[0]-600)*fault_type
                        label = np.concatenate((lab1_, lab2_), axis=0)
                        label = einops.rearrange(label, 'i -> i 1')
                        data = np.concatenate((dat, label), axis=1)
                        data_list.append(data)
                    else:
                        print(f"size of this data shouldn't exist, FAULT{fault_type}MAGNITUDE{magnitude}RUN{run}")
    
    return data_list

