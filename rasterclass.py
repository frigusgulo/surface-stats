import numpy as np
import pandas as pd
from skimage.feature import greycomatrix,greycoprops
from scipy.ndimage import gaussian_filter
import time
import sys
import matplotlib.pyplot as plt
from joblib import Memory
location = "./cachedir"
memory = Memory(location,verbose=0)
class rasterClass():



	def __init__(self,raster,name=None,labels=None,df=None,grid=500):
		self.res = 10 #m^2/pixel
		self.grid = grid
		self.raster = np.load(raster)
		if isinstance(labels,(int)):
			labels = np.ones(self.raster.shape)*labels
		elif labels is not None:
			self.labels = np.load(labels)

		self.rasterWidth = self.raster.shape[0]
		self.rasterHeight = self.raster.shape[1]
		self.azis = [0,45,90,135]
		self.distances = [0]
		self.textProps = ['contrast','dissimilarity','homogeneity','ASM','energy','correlation']
		self.detrend_ = False
		self.dfpath = df
		if df is not None:
			self.dataframe = pd.DataFrame.load(df)
		else:
			self.dataframe = pd.DataFrame()
		self.name = name

	@memory.cache 
	def detrend(self):
		print("\n Detrending \n")
		#perform detrending by applying a gaussian filter with a std of 200m, and detrend
		trend = gaussian_filter(self.raster,sigma=200)
		self.raster -= trend
		self.detrend_ = True

	def surfRough(self,image):
		return np.nanstd(image)

	@memory.cache
	def quantize(self):
		print("\n Quantizing \n")
		if self.detrend_: 
			self.raster += (np.abs(np.min(self.raster)) + 1)
			mean = np.nanmean(self.raster[self.raster > 0])
			std = np.nanstd(self.raster[self.raster > 0])

			self.raster[self.raster == None] = 0 # set all None values to 0
			self.raster[np.isnan(self.raster)] = 0
			self.raster = np.rint(self.raster)
			self.raster = self.raster.astype(int)

			self.raster[self.raster > (mean + 1.5*std)] = 0
			self.raster[self.raster < (mean - 1.5*std)] = 0 # High pass filter

			self.raster[self.raster > 0] = self.raster[self.raster > 0] - (np.min(self.raster[self.raster > 0]) - 1)

			self.raster[self.raster>101] = 0

			
			flat = np.ndarray.flatten(self.raster[self.raster > 0])
			range = np.max(flat) - np.min(flat)
			print("\n\nRaster Range: {}\n\n".format(range))
			#self.raster[self.raster > 101] = 0 #remove values greater than 101 for the GLCM

		else:
			raise ValueError("Raster Has Not Been Detrended")
	@memory.cache
	def greycomatrix(self,image):
		# returns a [(levels,levels),distance,angle] array
		matrices = greycomatrix(image,distances=self.distances,angles=self.azis,levels=102,symmetric=True)
		matrices = matrices[1:,1:,:,:] # remove entries respectice to Nan values
		print("marix shape: {}".format(matrices.shape))
		for i in range(len(self.azis)):
			for j in range(len(self.distances)):
				matrices[:,:,j,i] = matrices[:,:,j,i]/np.sum(matrices[:,:,j,i]) # normalize each matrix to sum to 1
		return matrices

	@memory.cache
	def comatprops(self,image):
		# returns a haralick feature for each image respective to a given azimuth
		features = {}
		for prop in self.textProps:
			features[prop] = greycoprops(image,prop=prop)
		return features

	def boxPlot(self):
		'''
		Create a boxplot of the response values so as to see if the data falls within a given range upon trend removal.
		The quantization step assumes that the data has been detrended so as to reduce the range of elevation values
		<= 100
		'''
		if self.detrend_:
			fig,ax = plt.subplots()
			ax.set_title("Boxplot of Detrended Elevation Values")
			ax.boxplot(np.ndarray.flatten(self.raster))
		else:
			raise ValueError("Raster Has Not Been Detrended")



	def mergeDicts(self,dicts):
		main = {}
		for dict_ in dicts:
			main = {**dict_}
		#main = sorted(main.items())
		return main

	def saveDF(self,path):
		print("\n\nSaving DataFrame\n\n")
		try:
			print(self.dataframe.head())
		except:
			print("Dataframe not found")

		if self.dataframe == None:
			self.dataframe.to_pickle(path=path)
		elif path==None:
			self.dataframe.to_pickle(path=self.dfpath)



	def iterate(self):
		print("Raster Datatype: {}".format(self.raster.dtype))
		counter = 0
		overall = time.time()
		for i in range(0,self.rasterWidth,self.grid):
			for j in range(0,self.rasterHeight,self.grid):
				start = time.time()
				image = self.raster[i:i+self.grid,j:j+self.grid]
				glcm =  self.greycomatrix(image)
				featuredicts = [self.comatprops(glcm)]
				features = self.mergeDicts(featuredicts)
				features["Srough"] = self.surfRough(image)
				if self.labels is not None:
					features["label"] = self.labels[i,j] # Not sure if this is the correct indexing, check old renditions of rasterClass()
				else:
					features["label"] = None
				if self.name is not None:
					features["Area"] = self.name

				self.dataframe = self.dataframe.append(features,ignore_index=True)
				end = time.time() - start
				print("Quadrat {} Done, Elapsed Time: {}".format(counter,end))
				counter += 1
		overall = time.time() - overall
		print("\n\n\nIteration Done, Elapse Time: {} Hours".format(overall/3600))

	def runAll(self,path=None):
		self.detrend()
		#self.boxPlot()
		self.quantize()
		self.iterate()
		self.saveDF(path)

