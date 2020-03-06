import numpy as np
import rasterio as ro
import matplotlib.pyplot as plt
import geopandas as gp
from rasterio.windows import Window
import cv2
import pandas as pd 

fields = {'extent':[],'msgl':[],'omnivar':[],'roughness':[]}

class raster():
	def __init__(self, raster):
		self.raster = ro.open(raster)
		self.crs = self.raster.crs
		self.transform = self.raster.transform
		self.meta = self.raster.profile
		self.height = self.meta['height']
		self.width = self.meta['width']
		self.dataframe = pd.DataFrame(fields)


	def changeNoData(self,value=None,outname='new_dem.tiff'):
		array = self.raster.read()
		array[array==-9999] = value
		with ro.open(outname,'w',**self.meta) as dst:
			dst.write(array.astype(rasterio.uint8))

	def convolve(image):
		kernel = np.array([[-2, -1,  0],[-1,  0 , 1],[0 ,  1 , 2]])
		# grab the spatial dimensions of the image, along with
		# the spatial dimensions of the kernel
		(iH, iW) = image.shape[:2]
		(kH, kW) = kernel.shape[:2]
		# allocate memory for the output image, taking care to
		# "pad" the borders of the input image so the spatial
		# size (i.e., width and height) are not reduced
		pad = (kW - 1) // 2
		image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
			cv2.BORDER_REPLICATE)
		output = np.zeros((iH, iW), dtype="float32")
		for y in np.arange(pad, iH + pad):
			for x in np.arange(pad, iW + pad):
				# extract the ROI of the image by extracting the
				# *center* region of the current (x, y)-coordinates
				# dimensions
				roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
				# perform the actual convolution by taking the
				# element-wise multiplicate between the ROI and
				# the kernel, then summing the matrix
				k = (roi * kernel).sum()
				# store the convolved value in the output (x,y)-
				# coordinate of the output image
				output[y - pad, x - pad] = k
		return output

	def get_window(self, extent):
		# extent is a list of tuples for the pixel wise extent of the raster data to view
		x_ex= extent[0]
		y_ex = extent[1]
		view_window = Window.from_sclices((x_ex[0],x_ex[1]),(y_ex[0],y_ex[1]))
		image = src.read(window=view_window)
		return image 

	def surfaceRoughness(self,image):
		# TODO

	def omniVariogram(self,image):
		# todo 