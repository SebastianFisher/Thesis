import numpy as np
import h5py
f = h5py.File('antenna_network_and_training_debug.mat','r')

# keys that can be accessed (to check that the matlab variables are loaded correctly
# print(f.keys())

# This works for getting the data into python. Have to transpose the last two dimensions for some reason for 
# them to match it seems
"""
data = f.get('XTest')
data = np.array(data) # For converting to a NumPy arrayo
data = np.transpose(data, [0, 1, 3, 2])

print(data[0,:,:,:])
"""

net = f.get('net')
print(net[0])


