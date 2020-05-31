import numpy as np
import pandas as pd
from skimage.feature import greycomatrix,greycoprops
from scipy.ndimage import gaussian_filter
import time
import sys
import matplotlib.pyplot as plt
from joblib import Memory
location = "/tmp"
memory = Memory(location,verbose=0)

#@memory.cache
def detrend(rasterclass):
	print("\n Detrending \n")
	#perform detrending by applying a gaussian filter with a std of 200m, and detrend
	trend = gaussian_filter(rasterclass.raster,sigma=200)
	rasterclass.raster -= trend
	rasterclass.detrend_ = True


#@memory.cache
def quantize(raster):
	print("\n Quantizing \n")
	if raster.detrend_: 
		raster.raster += (np.abs(np.min(raster.raster)) + 1)
		mean = np.nanmean(raster.raster[raster.raster > 0])
		std = np.nanstd(raster.raster[raster.raster > 0])

		raster.raster[raster.raster == None] = 0 # set all None values to 0
		raster.raster[np.isnan(raster.raster)] = 0

		raster.raster[raster.raster > (mean + 1.5*std)] = 0
		raster.raster[raster.raster < (mean - 1.5*std)] = 0 # High pass filter

		raster.raster[raster.raster > 0] = raster.raster[raster.raster > 0] - (np.min(raster.raster[raster.raster > 0]) - 1)
		raster.raster[raster.raster>101] = 0
		raster.raster = np.rint(raster.raster)
		
		

		

		
		flat = np.ndarray.flatten(raster.raster[raster.raster > 0])
		range = np.max(flat) - np.min(flat)
		print("\n\nRaster Range: {}\n\n".format(range))

		raster.raster = raster.raster.astype(np.uint8)
		plt.imshow(raster.raster)
		plt.savefig("quantizedraster")
		plt.clf()
		#self.raster[self.raster > 101] = 0 #remove values greater than 101 for the GLCM

	else:
		raise ValueError("Raster Has Not Been Detrended")


class rasterClass():

	def __init__(self,raster,name=None,labels=None,df=None,grid=500):
		self.res = 10 #m^2/pixel
		self.grid = grid
		self.raster = np.load(raster)
		if isinstance(labels,(int)):
			labels = np.ones( (int(self.raster.shape[0]/grid),int(self.raster.shape[1]/grid)) )*labels
		elif labels is not None:
			self.labels = np.load(labels)

		self.azis = [0]#[0, np.pi/4, np.pi/2, 3*np.pi/4]
		self.distances = [1,2,3,4,5]
		self.textProps = ['contrast','dissimilarity','homogeneity','ASM','energy','correlation']
		self.detrend_ = False
		self.dfpath = df
		if df is not None:
			self.dataframe = pd.read_pickle(df)
		else:
			self.dataframe = pd.DataFrame()
		self.name = name
		self.height =self.raster.shape[0]
		self.width = self.raster.shape[1]

	def surfRough(self,image):
		return np.nanstd(image)


	#@memory.cache
	def greycomatrix(self,image):
		# returns a [(levels,levels),distance,angle] array
	
		matrices = greycomatrix(image,distances=self.distances,levels=100,angles=self.azis,symmetric=True,normed=True)
		matrices = matrices[1:,1:,:,:] # remove entries respectice to Nan values
		
		'''
		for i in range(len(self.azis)):
			for j in range(len(self.distances)):
				cumsum = np.sum(matrices[:,:,j,i])
				matrices[:,:,j,i] = matrices[:,:,j,i] / cumsum
		plt.imshow(matrices)
		plt.show()
		'''
		return matrices

	
	#@memory.cache
	def comatprops(self,image):
		# returns a haralick feature for each image respective to a given azimuth
		image = np.sum(image,axis=2,keepdims=True) # sum all occurences within a given distance
		features = {}
		for prop in self.textProps:
			featvec = greycoprops(image,prop=prop)
			#print("\n",featvec,"\n")
			features[prop] =featvec
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

		if self.dfpath == None:
			self.dataframe.to_pickle(path=path)
		elif path==None:
			self.dataframe.to_pickle(path=self.dfpath)



	def iterate(self):
		print("Raster Datatype: {}".format(self.raster.dtype))
		counter = 0
		overall = time.time()
		for i in range(0,self.height-self.grid,self.grid):
			for j in range(0,self.width-self.grid,self.grid):
				indi = int(i/self.grid)
				indj = int(j/self.grid)
				start = time.time()
				image = self.raster[i:i+self.grid,j:j+self.grid]
				glcm =  self.greycomatrix(image)
				'''
				if i ==0 and j == 0:
					plt.imshow(glcm[:,:,0,0])
					plt.show()
					plt.clf()
				'''
			
				featuredicts = [self.comatprops(glcm)]
				features = self.mergeDicts(featuredicts)
				features["Srough"] = self.surfRough(image)
				if self.labels is not None:
					features["label"] = self.labels[indi,indj] # Not sure if this is the correct indexing, check old renditions of rasterClass()
				else:
					features["label"] = None
				if self.name is not None:
					features["Area"] = self.name

				self.dataframe = self.dataframe.append(features,ignore_index=True)
				end = np.round(time.time() - start)
				#print("Quadrat {} Done, Elapsed Time: {}".format(counter,end))
				counter += 1
		overall = time.time() - overall
		print("\n\n\nIteration Done, Elapse Time: {} for {} Quadrats".format((np.round(overall)),counter))

	def runAll(self,path=None):
		detrend(self)
		quantize(self)
		self.iterate()
		self.saveDF(path)

