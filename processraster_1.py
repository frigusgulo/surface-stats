import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from skgstat.models import spherical
from skgstat import DirectionalVariogram
import time as TIME
from scipy import ndimage
from numba import jitclass,jit,njit
import numba
import numpy as np
from joblib import Memory
cachedir = "/tmp"
memory = Memory(cachedir,verbose=0)

spec = [('raster',numba.float32[:,:]),('height', numba.int32),('width', numba.int32),('azis', numba.int64[:]),('grid',numba.int64),('rough',numba.float64[:,:]),('maxrange',numba.float64[:,:]),('aziratio',numba.float64[:,:]),('labels',numba.float64[:,:])]


@jitclass(spec)
class raster_class(object):
    def __init__(self,raster):
        self.raster = raster
        self.height =self.raster.shape[0]
        self.width = self.raster.shape[1]
        self.azis =  np.arange(0,170,10)
        self.grid = 500
        x = np.int(self.height/self.grid)
        y = np.int(self.width/self.grid)
        self.rough = np.zeros((x,y))
        self.maxrange = np.zeros((x,y)) 
        self.aziratio = np.zeros((x,y))
        #self.labels = np.zeros((x,y))

@jit  
def detrend(raster):
    raster.raster = ndimage.gaussian_filter(raster.raster,sigma=40)


@jit
def SR(image):
    return np.nanstd(image)

@memory.cache 
def dirVar(coords,response,azi,tolerance=15,maxlag=200,n_lags=20):
    time_ = TIME.time()
    print("Fitting Variogram for Azimuth {}\n".format(azi))
    DV = DirectionalVariogram(coords,response,azimuth=azi,tolerance=tolerance,maxlag=maxlag,n_lags=n_lags,fit_method='lm')  
    print("Fit Time: {} Minutes".format( (TIME.time() - time_)/60 ))
    return DV  
# Max lag of 2000 m, number of lags around 20

@jit
def getRange(raster,image):
    m,n = image.shape
    vals = image.flatten()[:,np.newaxis]
    coords = []
    for i in range(m):
        for j in range(n):
            coords.append((i,j))
    coords = np.array(coords)
    response = np.hstack((coords,vals))
    #response = response[~np.isnan(response[:,-1])]
    #response = response[response[:,-1] != 0]
    #response += np.random.normal(0,scale=0.25,size=response.shape[0]) #add noise to prevent same values
    inds = np.random.choice(response.shape[0],size=int((0.001)*m*n),replace=False)
    response = response[inds,:]

    azi_r = []
    for azi in raster.azis:
        DV =  dirVar(response[:,:2],response[:,2],azi)
        azi_r.append(DV.cof[0])
    major = np.argmax(azi_r)
    large_range = azi_r[major]
    major = azis[major]

    if major  >= 90:
        perp = major - 90
    else:
        perp = major + 90
    minor = azis.index(perp)
    minor_range = azi_r[minor]
    ratio = large_range/minor_range
    return ratio,large_range



@jit
def iterate(raster):
    for i in range(0,raster.height-raster.grid,raster.grid):
        for j in range(0,raster.width-raster.grid,raster.grid):
            image = raster.raster[i:i+raster.grid,j:j+raster.grid]
            indi = int(i/raster.grid)
            indj = int(j/raster.grid)
            roughness = SR(image)
            ratio,range_ = getRange(raster,image)
            raster.aziratio[indi,indj] = ratio
            raster.maxrange[indi,indj] = range_
            raster.rough[indi,indj] = roughness
            print("Quadrat Complete")


if __name__ == "__main__":
    brooks = np.load("brooks_dem.npy")
    brooks_class = raster_class(brooks)
    time = TIME.time()
    detrend(brooks_class)
    iterate(brooks_class)
    end_time = TIME.time() - time
    hours = end_time/3600
    print("Computation Took {} Hours".format(hours))