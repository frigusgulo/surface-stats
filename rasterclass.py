import numpy as np
from rasterio.transform import rowcol
from rasterio.transform import xy
import rasterio as ro
import matplotlib.pyplot as plt
import geopandas as gp
from rasterio.windows import Window
import cv2
import os
import pandas as pd 

fields = {'extent':[],'msgl':[],'omnivar':[],'roughness':[]}

class raster():
	def __init__(self, raster):
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
		coords = np.where(self.raster.read() !=-9999)
		xmin = np.min(coords[:,0])
		xmax = np.max(coords[:,0])
		ymin = np.min(coords[:,1])
		ymax = np.max(coords[:,1])
		self.extent_x = (xmin,xmax)
		self.extent_y = (ymin,ymax)

	def pix2world(self,r,c):
		x_y = xy(self.affine,r,c,offset='ul')
		return x_y


	def surfaceRoughness(self,image):
		image = np.flatten(image)
		mean = np.mean(image)
		roughness = (1/(len(image)-1))*((image-mean)**2)
		return np.sqrt(roughness)


	def omniVariogram(self,image):
		# todo 
		raise NotImplementedError


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



'''
if __name__ == '__main__':
	import sys
	raster = raster(sys.argv[1])
	xy = raster.pix2world(2,3)
	print(xy)
	raster.changeNoData()

'''