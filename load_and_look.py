from rasterio import open as rasopen
import matplotlib.pyplot as plt
import numpy as np
import sys
raster = sys.argv[1]
with rasopen(raster) as src:
    arr = np.squeeze(src.read())
    src = None  

print(arr.shape)
print(arr.dtype)

arr[arr == -9999] = np.nan

ts = 5000
for i in range(0, arr.shape[0]-ts, ts):
    for j in range(0, arr.shape[1]-ts, ts):
        plt.imshow(arr[i:i+ts, j:j+ts])
        plt.colorbar()
        plt.show()


        


