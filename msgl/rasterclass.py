import numpy as np
import pandas as pd
from skimage.feature import greycomatrix,greycoprops
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix
import time
import sys
import matplotlib.pyplot as plt
from joblib import Memory
from progress.bar import Bar
location = "/tmp"
memory = Memory(location,verbose=0)

#@memory.cache


#
'''
Reduce distances for greycomatrix
fix status bars
use different haralick features?

'''
def detrend(rasterclass):
	#perform detrending by applying a gaussian filter with a std of 200m, and detrend

	m,n = rasterclass.raster.shape
	step = 2000
	bar = Bar('Detrending',max=int(m/step)*int(n/step))
	for i in range(0,m,step):
		for j in range(0,n,step):
			chunk = rasterclass.raster[i:i+step,j:j+step]
			chunk[np.isnan(chunk)] = 0
			chunk[chunk==None] = 0
			trend = gaussian_filter(chunk,sigma=400)
			chunk = chunk - trend
			rasterclass.raster[i:i+step,j:j+step] = chunk
			chunk = None
			bar.next()
	bar.finish()
	rasterclass.detrend_ = True


#@memory.cache
def quantize(raster):

	print("\n Quantizing \n")
	if raster.detrend_:
		m,n = raster.raster.shape
		step = 1000
		meanlist = []
		stdlist = []
		for i in range(0,m,step):
		    for j in range(0,n,step):
		        chunk = raster.raster[i:i+step,j:j+step]
		        chunk[np.isnan(chunk)] = 0
		        chunk[chunk==None] = 0
		        meanlist.append(np.nanmean(chunk))
		        stdlist.append(np.nanstd(chunk))
		        raster.raster[i:i+step,j:j+step] = chunk
		    mean_ = np.median(meanlist)
		    meanlist = [mean_]
		    std_ = np.median(stdlist)
		    stdlist = [std_]
		    chunk = None

		mean = np.median(meanlist)
		std = np.median(stdlist)

		min_ = np.inf
		for i in range(0,m,step):
			for j in range(0,n,step):
				chunk = raster.raster[i:i+step,j:j+step]
				chunk[chunk > (mean + 2*std)] = 0
				chunk[chunk < (mean - 2*std)] = 0
				chunk = np.rint(chunk)
				chunk[chunk>99] = 0
				raster.raster[i:i+step,j:j+step] = chunk
				if np.min(chunk) is not None and np.min(chunk) < min_ :
					min_ = np.min(chunk)
				chunk = None

		min_ = np.abs(min_) + 1
		raster.raster = raster.raster +  min_
		raster.raster = np.rint(raster.raster).astype(np.uint8)



			
		flat = np.ndarray.flatten(raster.raster[raster.raster > 0])
		range_ = np.max(flat) - np.min(flat)
		print("\n\nRaster Range: {}\n\n".format(range_))

	
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
		else:
			self.labels = None

		self.azis = [0, np.pi/4, np.pi/2, 3*np.pi/4]
		self.distances = np.arange(15) + 1
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
		
		return matrices

	
	#@memory.cache
	def comatprops(self,image):
		
		# returns a haralick feature for each image respective to a given azimuth
		for i in range(len(self.azis)):
			sums = np.sum(image[:,:,:,i],axis=2)
			#sums = sums / np.sum(sums)
			image[:,:,0,i] =  np.reshape(sums,(image[:,:,0,i].shape))
			image = np.delete(image,self.distances,2)
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
		if self.dfpath == None:
			self.dataframe.to_pickle(path=path)
		elif path==None:
			self.dataframe.to_pickle(path=self.dfpath)



	def iterate(self):
		print("{} Raster Datatype: {}".format(self.name,self.raster.dtype))
		counter = 0
		overall = time.time()
		bar = Bar('Extracting Features',max=int(self.height/self.grid)*int(self.width/self.grid))
		for i in range(0,self.height-self.grid,self.grid):
			for j in range(0,self.width-self.grid,self.grid):
				indi = int(i/self.grid)
				indj = int(j/self.grid)
				
				image = self.raster[i:i+self.grid,j:j+self.grid]
				glcm =  self.greycomatrix(image)
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
				counter += 1
				bar.next()
		bar.finish()
		overall = time.time() - overall
		print("\nIteration Done, Elapse Time: {} for {} Quadrats\n".format((np.round(overall)),counter))

	def runAll(self,path=None):
		detrend(self)
		quantize(self)
		self.iterate()
		self.saveDF(path)

