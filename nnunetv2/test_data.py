import matplotlib.pyplot as plt
import numpy as np

file = "/afs/crc.nd.edu/user/y/ypeng4/data/preprocessed_data/Dataset003_Cirrus/nnUNetPlans_2d/TRAIN001_Cirrus.npz"

data = np.load(file)['data']
seg = np.load(file)['seg']
print(data.shape, seg.shape)

plt.imshow(data[0, 60], cmap="gray")
plt.show()

plt.imshow(seg[0, 60], cmap="gray")
plt.show()