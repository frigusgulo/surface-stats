import numpy as np
import pandas as pd
from skimage.feature import greycomatrix,greycoprops
from scipy.ndimage import gaussian_filter



def rasterClass():
	def __init__(raster,labels,grid=500):
		self.res = 10 #m^2/pixel
		self.grid = grid
		self.raster = np.load(raster)
		self.rasterWidth = self.raster.shape[0]
		self.rasterHeight = self.raster.shape[1]
		self.azis = [0,45,90,135]
		self.distances = [0]
		self.textProps = ['contrast','dissimilarity','homogeneity','ASM','energy','correlation']

	def detrend(self,sigma=20*self.res):
		#perform detrending by applying a gaussian filter with a std of 200m, and detrend
		self.raster -= gaussian_filter(self.raster,sigma=sigma)

	def surfRough(self):
		return np.nanstd(self.raster)

	def quantize(self):
		self.raster += (np.min(self.raster) + 1)
		self.raster[self.raster == None] = 0 # set all None values to 0
		self.raster[np.isnan(self.raster)] = 0
		self.raster = int(self.raster)
		self.raster[self.raster > 101] = 0 #remove values greater than 101 for the GLCM

	def greycomattrix(self,image):
		# returns a [(levels,levels),distance,angle] array
		matrices = greycomatrix(image,distances=self.distances,angles=self.azis,levels=102,symmetric=True)
		matrices = matrices[1:,1:,:,:] # remove entries respectice to Nan values
		for i in range(len(self.azis)):
			for j in range(len(self.distances))
				matrices[:,:,j,i] /= np.sum(matrices[:,:,j,i]) # normalize each matrix to sum to 1

	def comatprops(self,image,theta):
		# returns a haralick feature for each image respective to a given azimuth
		features = {}
		for prop in self.textProps:
			features[str(theta) + "_" + prop ] = greycoprops(image,prop=prop)
		return features

	def mergeDicts(dicts):
		main = {}
		for dict in dicts:
			main = {**dict}
		main = sorted(main.items())
		return main




