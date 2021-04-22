import numpy as np
import pandas as pd
from skimage.feature import greycomatrix,greycoprops
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix
import time
import sys
import matplotlib.pyplot as plt
#from joblib import Memory,Parallel
from progress.bar import Bar
import matplotlib.pyplot as plt
#location = "/tmp"
#memory = Memory(location,verbose=0)
from numba import jit
#@memory.cache


#
'''
Reduce distances for greycomatrix
fix status bars
use different haralick features?

'''
def detrend(rasterclass):

	rasterclass.raster[rasterclass.raster==0] = np.nan
	V = rasterclass.raster.copy()
	V[np.isnan(V)] = 0
	V[V==None] = 0

	W = 0*rasterclass.raster.copy() + 1
	W[np.isnan(W)] = 0
	VV = gaussian_filter(V,sigma=100)
	WW = gaussian_filter(W,sigma=100)
	trend = VV/WW
	trend = np.rint(trend).astype(np.int)


	rasterclass.raster = rasterclass.raster - trend

	rasterclass.detrend_ = True


#@memory.cache
def quantize(raster):

	print("\n Quantizing \n")
	'''

	raster.raster += np.abs(np.nanmin(raster.raster)) + 1
	raster.raster[raster.raster == None] =np.nan

	mean = np.nanmean(raster.raster)
	std = np.nanstd(raster.raster)

	raster.raster[raster.raster > mean + 2*std] = 0
	raster.raster[raster.raster < mean + 2*std] = 0
	raster.raster[np.isnan(raster.raster)] = 0
	'''
	raster.raster[np.isnan(raster.raster)] = int(0)
	raster.raster = np.rint(raster.raster).astype(np.uint8)

	range_ = np.nanmax(raster.raster) - np.nanmin(raster.raster)
	raster.range = range_+1
	print("\n\nRaster Range: {}\n\n".format(range_))
	#print(f"Raster Datatype: {raster.raster.dtype}\n")


def log(x):
	if x == 0.0:
		return 0.0
	else:
		return np.log(x)

def pxy(tensor,k,range_):
	ng = range_-1
	temp = 0.0
	for i in range(ng):
		for j in range(ng):
			if i+j == k:
				temp += tensor[i,j]
	return temp

def pxminusy(tensor,k,range_):
	ng = range_-1
	temp = 0
	for i in range(ng):
		for j in range(ng):
			if np.abs(i-j) == k:
				temp += tensor[i,j]
	return temp	


def entropy(tensor):
	row, col, dist,azi = tensor.shape
	vec = []
	for i in range(azi):
		temp = 0
		for j in range(row):
			for k in range(col):
				tens_ = tensor[j,k,:,i].astype(np.float64)
				temp += tens_*log(tens_)

		entropy = (-1)*temp
		vec.append(entropy)
	return np.array(vec)

def sum_entropy(tensor,range_):
	row, col, dist,azi = tensor.shape
	vec = []

	assert row == col
	for j in range(azi):
		sumentrop= 0
		for i in range(2,2*range_):
			temp = pxy(tensor[:,:,:,j],i,range_)
			sumentrop += temp*log(temp)
		vec.append(sumentrop)
	return np.array(vec)


def diff_entropy(tensor,range_):
	row, col, dist,azi = tensor.shape
	vec = []
	
	for j in range(azi):
		sumentrop = 0
		for i in range(0,range_-1):
			temp_ = pxminusy(tensor[:,:,:,j],i,range_)
			sumentrop -= temp_*log(temp_)
		vec.append(sumentrop)
	return np.array(vec)


def variance(tensor):
	row, col, dist,azi = tensor.shape
	vec = []
	for i in range(azi):
		var = np.nanstd(tensor[:,:,:,i])
		vec.append(var)
	return np.array(vec)

class rasterClass():

	def __init__(self,raster,name=None,labels=None,df=None,grid=500):
		self.res = 10 #m^2/pixel
		self.grid = grid
		self.raster = np.load(raster)
		if isinstance(labels,(int)):
			self.labels = np.zeros( (int(self.raster.shape[0]/grid),int(self.raster.shape[1]/grid)) )
		elif labels is not None:
			self.labels = np.load(labels)
		else:
			self.labels = None
	
		self.azis =  [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2,2*np.pi/3, 3*np.pi/4,5*np.pi/6]

		self.distance = 5
		self.distances = np.arange(self.distance)
		self.textProps = ['contrast','dissimilarity','correlation','variance','entropy']
		self.detrend_ = False
		self.dfpath = df
		if df is not None:
			self.dataframe = pd.read_pickle(df)
		else:
			self.dataframe = pd.DataFrame()
		self.name = name
		self.height =self.raster.shape[0]
		self.width = self.raster.shape[1]
		self.range = int(np.nanmax(self.raster))


	def surfRough(self,image):
		return np.nanstd(image)


	def greycomatrix(self,image):
		# returns a [(levels,levels),distance,angle] array

		matrices = greycomatrix(image,distances=self.distances,levels=self.range,angles=self.azis)
		matrices = matrices[1:,1:,:,:] # remove entries respectice to Nan values
		
		return matrices

	

	def comatprops(self,image):
	
		# returns a haralick feature for each image respective to a given azimuth
		temp = np.zeros((image.shape[0],image.shape[1],1,image.shape[3]))

		for i in range(len(self.azis)):
			sums = np.sum(image[:,:,:,i],axis=2)
			temp[:,:,0,i] =  np.reshape(sums,(temp[:,:,0,i].shape))

			# Sum freqs along all distances and store in first entry on the 2nd axis
			# remove all other distances
			
			
			#temp[:,:,0,i] = temp[:,:,0,i] + temp[:,:,0,i].transpose() #make symmetric
			temp[:,:,0,i] = temp[:,:,0,i] / np.sum(temp[:,:,0,i]) # normalize
			
			
		features = {}
		
		for prop in self.textProps:
			if prop is 'entropy':
				featvec = entropy(temp[:,:,0,:][:,:,np.newaxis,:])
			elif prop is 'sum_entropy':
				featvec = sum_entropy(temp[:,:,0,:][:,:,np.newaxis,:],self.range)
			elif prop is 'diff_entropy':
				featvec = diff_entropy(temp[:,:,0,:][:,:,np.newaxis,:],self.range)
			elif prop is 'variance':
				featvec = variance( temp[:,:,0,:][:,:,np.newaxis,:] )

			else:
				featvec = greycoprops(temp[:,:,0,:][:,:,np.newaxis,:],prop=prop)
			#print("\n",featvec,"\n")
			#featvec = [self.log(feature) for feature in featvec.tolist()]
			featvec[featvec == np.inf] = 0
			featvec[featvec == -np.inf] = 0

			features[prop] = featvec
		return features


	#@jit()
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
		self.raster = self.raster.astype(np.uint8)
		print("{} Raster Datatype: {}".format(self.name,self.raster.dtype))
		counter = 0
		overall = time.time()
		bar = Bar('Extracting Features',max=int(self.height/self.grid)*int(self.width/self.grid))
		for i in range(0,self.height-self.grid,self.grid):
			for j in range(0,self.width-self.grid,self.grid):
				indi = int(i/self.grid)
				indj = int(j/self.grid)
				
				image = self.raster[i:i+self.grid,j:j+self.grid].astype(np.int)

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


	def runAll(self,path=None,detrend_=True,quantize_=True):
		if detrend_: detrend(self)
		if quantize_: quantize(self)
		self.iterate()
		self.saveDF(path)

