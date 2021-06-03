import numpy as np 
from skimage.feature import greycomatrix,greycoprops
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt 
import time
import progressbar
import pdb
from progress.bar import Bar
import os

class Tile():
	def __init__(self,
		area: str,
		features: dict,
		index: tuple,
		label: bool =0):
		self.area = area
		self.features = features
		self.index = index
		self.label = label


class DEM():
	def __init__(self,
		area: str,
		dem: np.ndarray,
		labels: np.ndarray=None,
		resolution: int =None
		):
		self.dem = np.load(dem,allow_pickle=True)
		self.height,self.width = self.dem.shape[0],self.dem.shape[1]
		self.dem_detrended = None
		self.labels = None
		if labels is not None:
			self.labels = np.load(labels,allow_pickle=True)

		self.resolution = resolution
		self.grid=500
		self.area = area
		self.azis = np.pi*np.arange(0,2,.125)
		self.distances = np.array([1,2,3])

	def mergeDicts(self,dicts):
		main = {}
		for dict_ in dicts:
			main.update(dict_)
			#pdb.set_trace()
		return main

	def quantize(self):
		if np.nanmin(self.dem_detrended) <= 0:
			self.dem_detrended += (np.abs(np.nanmin(self.dem_detrended)) +1)
		else:
			self.dem_detrended[self.dem_detrended>0] -= (np.abs(np.nanmin(self.dem_detrended[self.dem_detrended>0]))-1)
		self.dem_detrended[self.dem_detrended==None] = np.nan
		self.dem_detrended[np.isnan(self.dem_detrended)] = 0
		self.dem_detrended = np.rint(self.dem_detrended).astype(np.uint8)
		self.detrend_range = np.nanmax(self.dem_detrended[self.dem_detrended>0]) - np.nanmin(self.dem_detrended[self.dem_detrended>0])
		print(f"\n Raster Range: {self.detrend_range}\n")


	def detrend_raster(self,sigma: int=1000):
		print(f"\n Detrending {self.area} Raster")
		sigma = sigma/self.resolution
		self.dem[self.dem==0] = np.nan
		V = self.dem.copy()
		V[np.isnan(V)] = 0
		V[V==None] = 0

		W = 0*self.dem.copy() + 1
		W[np.isnan(W)] = 0
		trend = gaussian_filter(V,sigma=sigma) +1e-300
		WW = gaussian_filter(W,sigma=sigma)+1e-300
		trend /= WW
		trend = np.rint(trend).astype(np.int)

		
		self.dem_detrended = self.dem - trend
		trend = None
		WW = None
		W = None
	

	def surface_roughness(self,image):
		return {"srough":np.nanstd(image)}

	def greycomatrix(self,image):
		try:
	
			matrices = greycomatrix(image,distances=self.distances,levels=self.detrend_range+3,angles=self.azis,symmetric=True,normed=True)
			matrices = matrices[1:,1:,:,:] # remove entires related to nan values
			return matrices # remove un-needed axis
		except Exception as e:
			print(f"\ngreycomatrix: {e}")
			
			return None
			#pdb.set_trace()
	def greycofeatures(self,image):
		
		textProps = ['contrast', 'dissimilarity', 'homogeneity', 'energy','ASM','correlation']
		features = {}
		#pdb.set_trace()
		for prop in textProps:
			if image is not None:
				vec = greycoprops(image,prop=prop).flatten()
			else:
				vec = None
			features[prop] = vec
		return features

	def iterate(self):
		tiles = []
		counter = 0
		overall = time.time()
		bar = Bar('Extracting Features',max=int(self.height/self.grid)*int(self.width/self.grid))
		for i in range(0,self.height-self.grid,self.grid):
			for j in range(0,self.width-self.grid,self.grid):
				indi = int(i/self.grid)
				indj = int(j/self.grid)
				
				image = self.dem[i:i+self.grid,j:j+self.grid]
				detrended = self.dem_detrended[i:i+self.grid,j:j+self.grid].astype(np.uint8)

				glcm =  self.greycomatrix(detrended)
				#print(glcm.shape)
				#pdb.set_trace()
				featuredict = self.mergeDicts([self.greycofeatures(glcm),self.surface_roughness(image)])
				#pdb.set_trace()
				if self.labels is None:
					tile = Tile(self.area,featuredict,(indi,indj))
				else:
					tile = Tile(self.area,featuredict,(indi,indj),self.labels[indi,indj])
				tiles.append(tile)
				
				counter += 1
				bar.next()
		bar.finish()
		overall = time.time() - overall
		print("\nIteration Done, Elapse Time: {} for {} Quadrats\n".format((np.round(overall)),counter))
		return tiles

	def run(self,path,load=False):
		if not os.path.isfile(path+".npy"):
			self.detrend_raster()
			self.quantize()
			self.tiles = self.iterate()
			np.save(path,self.tiles)
		elif os.path.isfile(path+".npy") and load:
			self.tiles = np.load(path+"*.npy",allow_pickle=True)


class DEM_Observer():

	def __init__(self,
		dems: list = None,
		areas: list=[]):
		self.dems = dems
		self.tiles = []
		self.areas = areas
		try:
			for dem in dems:
				self.tiles.extend([tile for tile in dem.tiles])
			self.areas = areas
		except:
			pass
		self.azis=np.pi*np.arange(0,2,.125)

	def load_data(self,datapath,features=None):
		try:
			self.tiles = np.hstack([self.tiles,np.load(datapath,allow_pickle=True)])
		except ValueError:
			self.tiles= np.load(datapath,allow_pickle=True)
		if features is not None:
			self.features = features
		[self.areas.append(tile.area) for tile in self.tiles if tile.area not in self.areas]
	def make_training_data(self):

		# Return array of training data from the observer
		# -> DATA[:, label:features]
		#features = list(self.tiles[0].features.keys())
		#print(features)
		#self.features = features
		azis = max(self.azis.shape) # debug
		
		self.training_data = np.zeros((len(self.tiles),290))
	
		for i, tile in enumerate(self.tiles):
			if tile.area in self.areas:
				self.training_data[i,0] = tile.label
				tostack = []
				for key,val in  zip(list(tile.features.keys()),list(tile.features.values())):
					tostack.append(val.flatten())
			


				#pdb.set_trace()
				self.training_data[i,1:] = np.hstack([item for item in tostack])

	def save_td(self,path):
		np.save(path,self.training_data)


	def correllogram(self,path):
	

		self.unitvartraining_data = np.zeros_like(self.training_data[:,1:])
		for j in range(1,self.training_data.shape[-1]-1):

			self.unitvartraining_data[:,j-1] = (self.training_data[:,j] - np.nanmean(self.training_data[:,j]))/np.nanvar(self.training_data[:,j])


		fig,ax = plt.subplots(len(self.features),len(self.features),squeeze=False)
		fig.suptitle("Distribution of Surface Features Across Classes; Red = False, Blue = True",fontsize=18)
		plt.subplots_adjust(left=0.1,
			bottom=0.1,
			right=0.9,
			top=0.9,
			wspace=0.65,
			hspace=0.65)
		labels = [self.training_data[:,0]==0, self.training_data[:,0]==1]
		for i in range(0,len(self.features)):
			for j in range(0,len(self.features)):
				if i < 5 : 
					slice_i = np.s_[i:i+4]
				else:
					slice_i = np.s_[i*4:]

				if j < 5 : 
					slice_j = np.s_[j:j+4]
				else:
					slice_j = np.s_[j*4:]

				if i==j:
					
					wherenans = [np.isnan(self.unitvartraining_data[labels[0],slice_i]),np.isnan(self.unitvartraining_data[labels[1],slice_i])]
					bins = np.histogram(np.hstack((self.unitvartraining_data[labels[0],slice_i][~wherenans[0]],self.unitvartraining_data[labels[1],slice_i][~wherenans[1]])),bins=30)[1]
					ax[i,j].hist(self.unitvartraining_data[labels[0],slice_i][~wherenans[0]],color='r',alpha=0.5,bins=bins)
					ax[i,j].hist(self.unitvartraining_data[labels[1],slice_i][~wherenans[1]],color='b',alpha=0.75,bins=bins)
					#ax[i,j].set_xlabel(self.features[i],fontsize=10)
					ax[i,j].set_ylabel(self.features[i]+" hist.",fontsize=10)
					ax[i,j].grid(True)

				elif i > j:
		
					if self.unitvartraining_data[labels[0],slice_i].shape < self.unitvartraining_data[labels[0],slice_j].shape:
						falsei = np.tile(self.unitvartraining_data[labels[0],slice_i],(1,self.unitvartraining_data[labels[0],slice_j].shape[1]))
					else:
						falsei = self.unitvartraining_data[labels[0],slice_i]

					if self.unitvartraining_data[labels[0],slice_j].shape < self.unitvartraining_data[labels[0],slice_i].shape:
						falsej = np.tile(self.unitvartraining_data[labels[0],slice_j],(1,self.unitvartraining_data[labels[0],slice_i].shape[1]))
					else:
						falsej = self.unitvartraining_data[labels[0],slice_j]

					if self.unitvartraining_data[labels[1],slice_i].shape < self.unitvartraining_data[labels[1],slice_j].shape:
						truei = np.tile(self.unitvartraining_data[labels[1],slice_i],(1,self.unitvartraining_data[labels[1],slice_j].shape[1]))
					else:
						truei = self.unitvartraining_data[labels[1],slice_i]
					if self.unitvartraining_data[labels[1],slice_j].shape < self.unitvartraining_data[labels[1],slice_i].shape:
						truej = np.tile(self.unitvartraining_data[labels[1],slice_j],(1,self.unitvartraining_data[labels[1],slice_i].shape[1]))
					else:
						truej = self.unitvartraining_data[labels[1],slice_j]
					#wherenans_false = np.logical_and(np.isnan(self.unitvartraining_data[labels[0],slice_i]),np.isnan(self.unitvartraining_data[labels[0],slice_j]))
					#wherenans_true = np.logical_and(np.isnan(self.unitvartraining_data[labels[1],slice_i]),np.isnan(self.unitvartraining_data[labels[1],slice_j]))
					ax[i,j].scatter(falsei,falsej,color='r',alpha=0.5,s=.5)
					ax[i,j].scatter(truei,truej,color='b',alpha=0.75,s=.75)
					ax[i,j].set_xlabel(self.features[i],fontsize=10)
					ax[i,j].set_ylabel(self.features[j],fontsize=10)
					ax[i,j].grid(True)
				else:
					ax[i,j].set_visible(False)

		plt.savefig(path)
		plt.show()



