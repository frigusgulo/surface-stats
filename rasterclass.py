import numpy as np
import pandas as pd
from skimage.feature import greycomatrix,greycoprops
from scipy.ndimage import gaussian_filter
import time
import sys
def rasterClass():



	def __init__(raster,labels,df=None,grid=500):
		self.res = 10 #m^2/pixel
		self.grid = grid
		self.raster = np.load(raster)
		self.labels = np.load(labels)
		self.rasterWidth = self.raster.shape[0]
		self.rasterHeight = self.raster.shape[1]
		self.azis = [0,45,90,135]
		self.distances = [0]
		self.textProps = ['contrast','dissimilarity','homogeneity','ASM','energy','correlation']
		self.detrend = False
		if df is not None:
			self.dataframe = pd.DataFrame.load(df)
		else:
			self.dataframe = pd.DataFrame()

	def detrend(self,sigma=20*self.res):
		#perform detrending by applying a gaussian filter with a std of 200m, and detrend
		self.raster -= gaussian_filter(self.raster,sigma=sigma)
		self.detrend = True

	def surfRough(self,image):
		return np.nanstd(image)

	def quantize(self):
		if self.detrend:
			self.raster += (np.abs(np.min(self.raster)) + 1)
			self.raster[self.raster == None] = 0 # set all None values to 0
			self.raster[np.isnan(self.raster)] = 0
			print("\n\nRaster Range: {}\n\n".format(np.max(self.raster) - np.min(self.raster)))
			self.raster = int(self.raster)
			self.raster[self.raster > 101] = 0 #remove values greater than 101 for the GLCM
		else:
			raise ValueError("Raster Has Not Been Detrended")

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

	def boxPlot(self)
		'''
		Create a boxplot of the response values so as to see if the data falls within a given range upon trend removal.
		The quantization step assumes that the data has been detrended so as to reduce the range of elevation values
		<= 100
		'''
		if self.detrend:
			fig,ax = plt.subplots()
			ax.set_title("Boxplot of Detrended Elevation Values")
			ax.boxplot(np.ndarray.flatten(self.raster))
		else:
			raise ValueError("Raster Has Not Been Detrended")



	def mergeDicts(self,dicts):
		main = {}
		for dict in dicts:
			main = {**dict}
		main = sorted(main.items())
		return main

	def saveDF(self,path):
		print("\n\nSaving DataFrame\n\n")
		print(self.dataframe.head())
		self.dataframe.to_pickle(path=path)

	def iterate():
		counter = 0
		overall = time.time()
		for i in range(0,self.rasterWidth,self.grid):
			for j in range(0,self.rasterHeight,self.grid):
				start = time.time()
				image = self.raster[i:i+self.grid,j:j+self.grid]
				glcm = self.greycomatrix(image)
				featuredicts = []
				for theta in self.azis:
					featuredicts.append(self.comatprops(glcm,theta))
				features = self.mergeDicts(featuredicts)
				features["Srough"] = self.surfRough(image)
				features["label"] = self.labels[i,j] # Not sure if this is the correct indexing, check old renditions of rasterClass()
				self.dataframe = self.dataframe.append(features,ignore_index=True)
				end = time.time() - start
				print("Quadrat {} Done, Elapsed Time: {}".format(counter,end))
				counter += 1
		overall = time.time() - overall
		print("\n\n\nIteration Done, Elapse Time: {} Hours".format(overall/3600))

	def runAll(self,path):
		self.detrend()
		self.boxPlot()
		self.quantize()
		self.iterate()
		self.saveDF(path)

