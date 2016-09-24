import scipy.io
import numpy as np
from matplotlib import pyplot as plt

mat = scipy.io.loadmat(r'./training_set/p300soft/boostingp300/data.mat')#, variable_names  = ['test', 'train'])
x = mat['x']
y = mat['y']
M = np.array(y, dtype='bool')

oz = x[-1,]
aiml = oz[:,M[0]]
aavg = np.average( aiml, axis = 1)

naiml = oz[:,(np.logical_not(M))[0]]
naavg = np.average( naiml, axis = 1)



# plt.plot(np.arange(np.shape(aavg)[0]), aavg)
# plt.plot(np.arange(np.shape(aavg)[0]), naavg)
# plt.plot(np.arange(np.shape(aavg)[0]), aavg - naavg, linewidth = 10)

# plt.plot(np.arange(np.shape(aavg)[0]), naiml)
# plt.plot(np.arange(np.shape(aavg)[0]), naavg, linewidth=10)
# plt.show()