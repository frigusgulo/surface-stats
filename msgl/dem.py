import numpy as np 
from skimage.feature import greycomatrix,greycoprops
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt 
from progress.bar import Bar 



class Tile():
	def __init__(self,
		area: str,
		features: dict,
		index: tuple,
		label: bool):
	self.area = area
	self.features = features
	self.index = index
	self.label = label


class DEM():
	def __init__(self,
		area: str,
		dem: np.ndarray,
		labels: np.ndarray=None,
		name: str,
		resolution: int,
		detrend: bool = True
		):
	self.dem = dem
	self.height,self.width = self.dem.shape[0],self.dem.shape[1]
	self.dem_detrended = None
	self.labels = labels
	self.name = name
	self.resolution = resolution
	self.detrend = detrend
	self.grid=500
	if detrend:
		self.detrend()
		self.quantize(self.dem_detrended)

	def mergeDicts(self,dicts):
		main = {}
		for dict_ in dicts:
			main = {**dict_}
		return main

	def quantize(self,dem:np.ndarray):
		dem[np.isnan(dem)] = 0
		dem = np.rint(dem).astype(np.uint8)
		self.detrend_range = np.nanmax(dem) - np.nanmin(dem) + 1
		print(f"\n Raster Range: {self.detrend_range}\n")


	def detrend(self,sigma: int=1000):
		sigma = sigma/self.resolution
		self.dem[self.dem==0] = np.nan
		V = self.dem.copy()
		V[np.isnan(V)] = 0
		V[V==None] = 0

		W = 0*self.dem.copy() + 1
		W[np.isnan(W)] = 0
		VV = gaussian_filter(V,sigma=sigma)
		WW = gaussian_filter(W,sigma=sigma)
		trend = VV/WW
		trend = np.rint(trend).astype(np.int)

		self.dem[np.isnan(dem)] = 0
		self.dem_detrended = self.dem - trend

	def surface_roughness(self,image):
		return {"srough":np.nanstd(image)}

	def greycomatrix(self,image):
		matrices = greycomatrix(image,distances=1,levels=self.detrend_range,angles=[0, np.pi/4,  np.pi/2, 3*np.pi/4])
		matrices = matrices[1:,1:,:,:] # remove entires related to nan values
		return np.squeeze(matrices) # remove un-needed axis

	def greycofeatures(self,image):
		features = {}
		textProps = ['contrast','dissimilarity','correlation','variance','entropy']
		for prop in textProps:
			vec = greycoprops(image,prop=prop)
			features[prop] = vec

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
				detrended = self.dem_detrended[i:i+self.grid,j:j+self.grid].astype(np.int)

				glcm =  self.greycomatrix(detrended)
				featuredict = self.mergeDicts([self.greycofeatures(glcm),self.surface_roughness(image)])
				tile = Tile(self.area,featuredict,(indi,indj),self.labels[indi,indj])
				tiles.append(tile)
				
				counter += 1
				bar.next()
		bar.finish()
		overall = time.time() - overall
		print("\nIteration Done, Elapse Time: {} for {} Quadrats\n".format((np.round(overall)),counter))
		return tiles

	def run(self):
		self.tiles = self.iterate()


class DEM_Observer():

	def __init__(self,
		dems: list,
		areas: list,
		testareas: str):
	self.dems = dems
	self.tiles = [dem.tiles[:] for dem in self.dems]

	self.tiles = sorted(self.tiles, key = lambda i: i.area, i.index[0],i.index[1])
	self.areas = areas
	self.testarea = testareas



	def training_data(self):
		# Return array of training data from the observer
		# -> DATA[:, label:features]
		features = self.tiles[0].features.keys()
		self.features = features
		self.training_data = np.zeros((len(self.tiles),len(features)+1))

		for i, tile in enumerate(self.tiles):
			if tile.area is in self.areas:
				self.training_data[i,0] = tile.label
				for j,value in enumerate(tile.features.values()):
					self.training_data[i,j+1] = value

	def save_td(self,path):
		np.save("training_data_array",self.training_data)


	def correllogram(self,path):
		fig,ax = plt.subplots((len(self.features),len(self.features)),squeeze=False)
		fig.suptitle("Distribution of Surface Features Across Classes",fontsize=18)
		plt.subplots_adjust(left=0.1,
			bottom=0.1,
			right=0.9,
			top=0.9,
			wspace=0.4,
			hspace=0.4)
		labels = [np.argwhere(self.features[:,0]==0), np.argwhere(self.features[:,0]==1)]
		for i in range(1,len(self.features)+1):
			for j in range(1,len(self.features)+1):

				if i==j:
					ax[i,j].hist(self.training_data[labels[0],i],color='r',alpha=0.5,bins=25)
					ax[i,j].hist(self.training_data[labels[1],i],color='b',alpha=0.5,bins=25)
					ax[i,j].set_xlabel(self.features[i-1],fontsize=12)
					ax[i,j].set_ylabel("Frequency",fontsize=12)
					ax[i,j].grid(True)

				elif i < j:
					ax[i,j].scatter(self.training_data[labels[0],i],self.training_data[labels[0],j],color='r',alpha=0.5,s=.75)
					ax[i,j].scatter(self.training_data[labels[1],i],self.training_data[labels[1],j],color='b',alpha=0.5,s=.75)
					ax[i,j].set_xlabel(self.features[i-1],fontsize=12)
					ax[i,j].set_ylabel(self.features[j-1],fontsize=12)
					ax[i,j].grid(True)
				else:
					ax[i,j].set_visible(False)
		plt.savefig(path)
		plt.show()



