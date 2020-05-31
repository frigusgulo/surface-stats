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

class raster_class():

    def __init__(self,raster,msgl):
        self.raster = raster
        self.height = np.shape(self.raster)[0]
        self.width = np.shape(self.raster)[1]
        self.azis = np.arange(0,120,40)
        self.grid = 50
        self.respose_rough = np.zeros((int(self.height/self.grid),int(self.width/self.grid)))
        self.azi_0 = self.respose_rough.copy()
        self.azi_40 = self.respose_rough.copy()
        self.azi_80 = self.respose_rough.copy()
        self.azi_120 = self.respose_rough.copy()
        self.done = False
        if msgl:
            self.msgl = np.ones((int(self.height/self.grid),int(self.width/self.grid)))
        else:
            self.msgl = self.respose_rough.copy()

    def SR(self,image):
        image = np.ndarray.flatten(image)
        mean = np.mean(image)
        roughness = (1/(len(image)-1))*np.sum((image-mean)**2)
        return np.sqrt(roughness)

    def getRange(self,mat):
        # fits an anisotropic variogram model and returns the effective range for a given azimuth
        mat = np.squeeze(mat)

        m,n = mat.shape
        coords = []
        for i in range(m):
            for j in range(n):
                coords.append((i,j))
        coords = np.array(coords)
        response = np.ndarray.flatten(mat)
        azi_r = []
        for azi in self.azis:
            DV = DirectionalVariogram(coords,response,azimuth=45,tolerance=15,maxlag=int(m/2),n_lags=int(m/10) + 1)
            azi_r.append(DV.cof[0])
        
        return azi_r

    def iterate(self):
        for i in range(0,self.width-self.grid,self.grid):
            for j in range(0,self.height-self.grid,self.grid):
                image = self.raster[i:i+self.grid,j:j+self.grid]
                indi = int(i/self.grid)
                indj = int(j/self.grid)
                self.respose_rough[indi,indj] = self.SR(image)
                azi_r = self.getRange(image)
                self.azi_0[indi,indj]   = azi_r[0]
                self.azi_40[indi,indj]  = azi_r[1]
                self.azi_80[indi,indj]  = azi_r[2]
                self.azi_120[indi,indj] = azi_r[3]
                self.done = True
                
    def genCovariate(self,mat):
        # Generates a covariate matrix from response variables
        response = np.ndarray.flatten(np.squeeze(mat))
        m,n = mat.shape
        coords = []
        for i in range(m):
            for j in range(n):
                coords.append((i,j))
        coords = np.array(coords)
        V = Variogram(coords,response)
        dists = squareform(V.distances)
        covars = np.like_zeros(dists)
        covars = spherical(covars,*V.cof)
        return covars

    def saveAll(self,msgl):
        if self.done:
            np.save("roughness" + str(msgl),self.response_rough)
            np.save("azi_0"     + str(msgl),self.azi_0)
            np.save("azi_40"    + str(msgl),self.azi_40)
            np.save("azi_80"    + str(msgl),self.azi_80)
            np.save("azi_120"   + str(msgl),self.azi_120)


