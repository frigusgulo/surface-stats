import numpy as np
from rasterio.transform import rowcol
from rasterio.transform import xy
import rasterio as ro
import matplotlib.pyplot as plt
import geopandas as gp
from rasterio.windows import Window
import cv2
import os
import json
import pandas as pd 
from shapely.geometry import box
from skgstat import DirectionalVariogram

fields = {'extent':[],'msgl':[],'omnivar':[],'roughness':[]}

class raster():
	def __init__(self, raster,outmeta=None):
		if outmeta is not None:
			self.raster = ro.open(raster,**out_meta)
		else:
			self.raster = ro.open(raster)
		self.crs = self.raster.crs
		print(self.crs)
		self.res = 2 #2^m per pixel
		self.raster_width = self.raster.shape[0]
		self.raster_height = self.raster.shape[1]
		self.meta = self.raster.profile
		self.height = self.meta['height']
		self.width = self.meta['width']
		self.dataframe = pd.DataFrame(fields)
		self.extent_x = None
		self.extent_y = None
		self.affine = self.raster.meta['transform']
		self.NoData = False
		
	def writeRaster(self,input_,outName):
		with ro.open(outName,'w',**self.meta) as dst:
			dst.write(input_.astype(ro.float32))
			dst = None

	def changeNoData(self,value=None,outname='new_dem.tiff'):
		array = self.raster.read()
		array[array==-9999] = value
		with ro.open(outname,'w',**self.meta) as dst:
			dst.write(array.astype(ro.float32))
			self.raster = dst
		self.NoData = True

	def convolve(self,image):
		kernel = np.array([[-2, -1,  0],[-1,  0 , 1],[0 ,  1 , 2]])
		(iH, iW) = image.shape[:2]
		(kH, kW) = kernel.shape[:2]
		pad = (kW - 1) // 2
		image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
			cv2.BORDER_REPLICATE)
		output = np.zeros((iH, iW), dtype="float32")
		for y in np.arange(pad, iH + pad):
			for x in np.arange(pad, iW + pad):
				roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
				k = (roi * kernel).sum()
				output[y - pad, x - pad] = k
		return output

	def get_window(self, extent):
		x_ex= extent[0]
		y_ex = extent[1]
		view_window = Window.from_sclices((x_ex[0],x_ex[1]),(y_ex[0],y_ex[1]))
		image = src.read(window=view_window)
		return image

	def get_extent(self):
		coords = np.where(self.raster.read()!=-9999,self.raster.read())
		xmin = np.min(coords[:,0])
		xmax = np.max(coords[:,0])
		ymin = np.min(coords[:,1])
		ymax = np.max(coords[:,1])
		self.extent_x = (xmin,xmax)
		self.extent_y = (ymin,ymax)
		print(self.extent_x)
		print(self.extent_y)

	def pix2world(self,r,c):
		x_y = xy(self.affine,r,c,offset='ul')
		return x_y


	def surfaceRoughness(self,image):
		image = np.flatten(image)
		mean = np.mean(image)
		roughness = (1/(len(image)-1))*((image-mean)**2)
		return np.sqrt(roughness)


	def kmltoshp(self, kmlfile):
		shapefile = kmlfile.strip('.kml')
		output = shapefile + '.shp'
		os.system("ogr2ogr -f 'ESRI Shapefile' " + output + " " + shapefile)


	def tileByextent(self):
		ts = int(3000/2)
		print("Tile by extent")
		path = '/home/dunbar/DEM/tiles'
		#if self.NoData:
		self.tiles = []
		for i in range(0,self.raster_height-ts, ts):
			for j in range(0, self.raster_width-ts,ts):
				arr = np.squeeze(self.raster.read()[i:i+ts,j:j+ts])
				print(arr.shape)
				arr = self.convolve(arr)
				plt.imshow(arr)
				plt.colorbar()
				plt.show()
				outname = str(i+ts) + "_" + str(j+ts)
				output = os.path.join(path,outname)
				with ro.open(output,'w',**self.meta) as dst:
					dst.write(arr.astype(ro.float32))
	def squareBounds(self,data):
		data = ro.open(data,'r')
		bbox = box(self.extent_x[0], self.extent_y[0],self.extent_x[1],self.extent_y[1])
		geo = gp.GeoDataFrame({'geometry': bbox}, index=[0], crs=self.crs)
		geo = geo.to_crs(crs=data.crs.data)
		coords = [json.loads(geo.to_json())['features'][0]['geometry']]
		out_img, out_transform = mask(raster=data, shapes=coords, crop=True)
		out_meta = data.meta.copy()
		epsg_code = int(data.crs.data['init'][5:])
		out_meta.update({"driver": "GTiff",
			"height": out_img.shape[1],
			"width": out_img.shape[2],
			"transform": out_transform,
			"crs": pycrs.parser.from_epsg_code(epsg_code).to_proj4()})
		self.meta = out_meta
		self.writeRaster(data,'squarebounds.tiff')

	def getRange(self,mat,azimuth):
		# fits an anisotropic variogram model and returns the effective range for a given azimuth
		mat = np.squeeze(mat)
		m,n = mat.shape
		coords = []
		for i in range(m):
			for j in range(n):
				coords.append((i,j))
		coords = np.array(coords)
		response = np.ndarray.flatten(mat)
		DV = DirectionalVariogram(coords,response,azimuth,tolerance=15,maxlag=int(m/2),n_lags=int(m/10))
		return DV.cof[0]

	def genCovariate(self,mat):
		# Generates a covariate matrix from response variables
		response = np.flatten(np.squeeze(mat))
		mu = np.mean(reponse)
		n = len(response)
		covar = np.zeros(n,n)
		for i in range(n):
			for j in range(n):
				covar[i,j] = (response[i] - mu)*(response[j] - mu)
		return covar

'''
if __name__ == '__main__':
	import sys
	raster = raster(sys.argv[1])
	xy = raster.pix2world(2,3)
	print(xy)
	raster.changeNoData()

'''
